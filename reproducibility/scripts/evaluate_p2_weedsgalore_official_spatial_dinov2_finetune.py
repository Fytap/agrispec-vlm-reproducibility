#!/usr/bin/env python3
"""Eight-GPU paired DINOv2 target fine-tuning on the official spatial split.

Ten prospectively selected DINOv2-coreset label draws are reused from the
frozen baseline.  Repeats are dispatched across the explicitly assigned
physical GPUs 0--7.  The official validation split chooses the best of eight
fixed epochs; the official test split is scored only after that choice.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_commissioning_contract_audit import audit_setting, load_truth  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_proposal_audit import official_split_map  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_queue import metric_record, selected_by_score  # noqa: E402
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import read_tsv, sha256  # noqa: E402


REPEATS = 10
EPOCHS = 8
PHYSICAL_GPUS = tuple(range(8))
KS = (1, 5, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("project-root", "archive", "data-root", "manifest", "candidate-root", "selection-ids", "frozen-selection-results", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    for name in ("archive", "manifest", "candidates", "selection-ids", "frozen-selection-results"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-seed", type=int, default=2026081200)
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_images(manifest: list[dict[str, str]], root: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for row in manifest:
        bands = json.loads(row["source_image_relpaths_json"])
        channels = [np.asarray(Image.open(root / bands[name]), dtype=np.float32) / 65535.0 for name in ("R", "G", "B")]
        result[row["sample_id"]] = np.clip(np.stack(channels, axis=-1) * 255.0, 0, 255).astype(np.uint8)
    return result


def id_hash(ids: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def worker(payload: dict[str, object]) -> dict[str, object]:
    repeat = int(payload["repeat"])
    physical_gpu = int(payload["physical_gpu"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    import timm
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import v2

    seed = int(payload["base_seed"]) + repeat
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError(f"GPU unavailable for repeat {repeat}")
    torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats(0)

    manifest = read_tsv(Path(str(payload["manifest"])))
    split_map = official_split_map(Path(str(payload["archive"])), manifest)
    truth = load_truth(Path(str(payload["archive"])), manifest)
    candidate_root = Path(str(payload["candidate_root"]))
    audit = audit_setting(candidate_root, manifest, truth)
    rows = audit["rows"]
    for row in rows:
        row["official_split"] = split_map[str(row["file"])]
    boxes = {row["candidate_id"]: row for row in read_tsv(candidate_root / "candidate_components.tsv")}
    images = load_images(manifest, Path(str(payload["data_root"])))
    selected_ids = set(payload["selected_ids"])
    train_rows = [row for row in rows if row["official_split"] == "train" and str(row["candidate_id"]) in selected_ids]
    validation_rows = [row for row in rows if row["official_split"] == "val"]
    test_rows = [row for row in rows if row["official_split"] == "test"]
    if len(train_rows) != len(selected_ids) or len(train_rows) != 300:
        raise ValueError(f"Selection coverage mismatch for repeat {repeat}")

    train_transform = v2.Compose([
        v2.ToImage(), v2.Resize((224, 224), antialias=True),
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    score_transform = v2.Compose([
        v2.ToImage(), v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    class CandidateCrops(Dataset):
        def __init__(self, records: list[dict[str, object]], transform: object):
            self.records, self.transform = records, transform
        def __len__(self) -> int:
            return len(self.records)
        def __getitem__(self, index: int):
            row = self.records[index]; box = boxes[str(row["candidate_id"])]
            x0, y0, x1, y1 = (int(float(box[key])) for key in ("x0", "y0", "x1_exclusive", "y1_exclusive"))
            image = images[str(row["file"])]
            crop = image[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
            return self.transform(crop), torch.tensor(row["truth"] == "weed", dtype=torch.float32), str(row["candidate_id"])

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(CandidateCrops(train_rows, train_transform), batch_size=int(payload["batch_size"]), shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(CandidateCrops(validation_rows, score_transform), batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(CandidateCrops(test_rows, score_transform), batch_size=128, shuffle=False, num_workers=0)

    model = timm.create_model(str(payload["model_name"]), pretrained=True, num_classes=0, img_size=224).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.blocks[-1].parameters():
        parameter.requires_grad_(True)
    for parameter in model.norm.parameters():
        parameter.requires_grad_(True)
    head = nn.Linear(model.num_features, 1).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.blocks[-1].parameters(), "lr": 1e-5},
        {"params": model.norm.parameters(), "lr": 1e-5},
        {"params": head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    positives = sum(row["truth"] == "weed" for row in train_rows)
    negatives = len(train_rows) - positives
    positive_weight = torch.tensor([negatives / max(1, positives)], device=device)

    def score(loader: DataLoader) -> tuple[np.ndarray, np.ndarray, list[str]]:
        values, labels, ids = [], [], []
        model.eval(); head.eval()
        with torch.inference_mode():
            for x, y, candidate_ids in loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction = torch.sigmoid(head(model(x.to(device))).squeeze(1))
                values.extend(prediction.float().cpu().tolist()); labels.extend(y.int().tolist()); ids.extend(candidate_ids)
        return np.asarray(values), np.asarray(labels), ids

    history = []
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, object] | None = None
    repeat_started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train(); head.train(); losses = []
        for x, y, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = head(model(x.to(device))).squeeze(1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y.to(device), pos_weight=positive_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_([parameter for parameter in list(model.parameters()) + list(head.parameters()) if parameter.requires_grad], 5.0); optimizer.step()
            losses.append(float(loss.detach()))
        val_scores, val_labels, _ = score(validation_loader)
        from sklearn.metrics import average_precision_score, roc_auc_score
        val_auc = float(roc_auc_score(val_labels, val_scores)); val_ap = float(average_precision_score(val_labels, val_scores))
        history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "validation_all_candidate_roc_auc": val_auc, "validation_all_candidate_average_precision": val_ap})
        key = (val_auc, val_ap, -epoch)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                "model": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items() if name.startswith("blocks.11.") or name.startswith("norm.")},
                "head": {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()},
                "epoch": epoch,
            }
    if best_state is None:
        raise RuntimeError("No validation checkpoint selected")
    current = model.state_dict(); current.update(best_state["model"]); model.load_state_dict(current); head.load_state_dict(best_state["head"])
    test_scores, test_labels, test_ids = score(test_loader)
    from sklearn.metrics import average_precision_score, roc_auc_score
    checkpoint_path = Path(str(payload["checkpoint_dir"])) / f"repeat_{repeat:02d}.pt"
    torch.save({"repeat": repeat, "seed": seed, "physical_gpu": physical_gpu, "model_name": payload["model_name"], "crop_contract": "tight proposal bounding box without padding", "selected_training_ids_sha256": id_hash(selected_ids), "selected_epoch": best_state["epoch"], "trainable_model_state": best_state["model"], "head_state": best_state["head"], "history": history}, checkpoint_path)
    return {
        "repeat": repeat, "seed": seed, "physical_gpu": physical_gpu,
        "selected_training_ids_sha256": id_hash(selected_ids), "training_candidates": len(train_rows),
        "training_weed_candidates": positives, "training_nonweed_candidates": negatives,
        "selected_epoch": best_state["epoch"], "history": history,
        "test_all_candidate_roc_auc": float(roc_auc_score(test_labels, test_scores)),
        "test_all_candidate_average_precision": float(average_precision_score(test_labels, test_scores)),
        "test_ids": test_ids, "test_scores": test_scores.tolist(),
        "checkpoint": checkpoint_path.name, "checkpoint_sha256": sha256(checkpoint_path),
        "wall_seconds": time.perf_counter() - repeat_started,
        "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(0), "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "timm": timm.__version__,
    }


def main() -> None:
    args = parse_args(); started = time.perf_counter()
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    checks = (
        (args.archive, args.expected_archive_sha256), (args.manifest, args.expected_manifest_sha256),
        (args.candidate_root / "candidate_components.tsv", args.expected_candidates_sha256),
        (args.selection_ids, args.expected_selection_ids_sha256),
        (args.frozen_selection_results, args.expected_frozen_selection_results_sha256),
    )
    for path, expected in checks:
        if sha256(path.resolve()) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
    selection_rows = read_tsv(args.selection_ids.resolve())
    selections = {repeat: {row["candidate_id"] for row in selection_rows if row["strategy"] == "prospective_dinov2_coreset" and int(row["repeat"]) == repeat} for repeat in range(REPEATS)}
    if any(len(value) != 300 for value in selections.values()):
        raise ValueError("Expected 300 DINOv2-coreset IDs for each of ten repeats")
    frozen_rows = {int(row["repeat"]): row for row in read_tsv(args.frozen_selection_results.resolve()) if row["strategy"] == "prospective_dinov2_coreset"}
    if set(frozen_rows) != set(range(REPEATS)):
        raise ValueError("Frozen paired baseline repeat coverage mismatch")

    output.mkdir(parents=True); checkpoint_dir = output / "checkpoints"; checkpoint_dir.mkdir()
    base_payload = {
        "base_seed": args.base_seed, "archive": str(args.archive.resolve()), "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()), "candidate_root": str(args.candidate_root.resolve()),
        "model_name": args.model_name, "batch_size": args.batch_size, "checkpoint_dir": str(checkpoint_dir),
    }
    context = mp.get_context("spawn")
    payloads = [{**base_payload, "repeat": repeat, "physical_gpu": PHYSICAL_GPUS[repeat % len(PHYSICAL_GPUS)], "selected_ids": sorted(selections[repeat])} for repeat in range(REPEATS)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(PHYSICAL_GPUS), mp_context=context) as executor:
        futures = [executor.submit(worker, payload) for payload in payloads]
        results = [future.result() for future in futures]
    results.sort(key=lambda row: int(row["repeat"]))

    manifest = read_tsv(args.manifest.resolve()); split_map = official_split_map(args.archive.resolve(), manifest)
    truth = load_truth(args.archive.resolve(), manifest); audit = audit_setting(args.candidate_root.resolve(), manifest, truth)
    rows = audit["rows"]
    for row in rows: row["official_split"] = split_map[str(row["file"])]
    index_by_id = {str(row["candidate_id"]): index for index, row in enumerate(rows)}
    test_indices = [index for index, row in enumerate(rows) if row["official_split"] == "test"]
    test_instances = [row for row in audit["instances"] if split_map[str(row["sample_id"])] == "test"]
    result_rows, queue_rows, prediction_rows, paired_rows, history_rows = [], [], [], [], []
    for result in results:
        scores = np.zeros(len(rows), dtype=np.float64)
        for candidate_id, value in zip(result["test_ids"], result["test_scores"], strict=True):
            scores[index_by_id[candidate_id]] = float(value)
            row = rows[index_by_id[candidate_id]]
            prediction_rows.append({"repeat": result["repeat"], "candidate_id": candidate_id, "file": row["file"], "acquisition_date": row["date"], "dataset_role_target": row["truth"], "score": f"{float(value):.12f}"})
        primary_metrics = None
        for k in KS:
            selected = selected_by_score(rows, scores, test_indices, k)
            metrics = metric_record(rows, selected, test_instances, audit["spatial"], audit["qualified"], "pooled_test", "ALL")
            queue_rows.append({"repeat": result["repeat"], "k": k, **metrics})
            if k == 20: primary_metrics = metrics
        result_rows.append({key: value for key, value in result.items() if key not in {"test_ids", "test_scores", "history"}})
        history_rows.extend({"repeat": result["repeat"], **row} for row in result["history"])
        frozen = frozen_rows[int(result["repeat"])]
        paired_rows.append({
            "repeat": result["repeat"], "selected_training_ids_sha256": result["selected_training_ids_sha256"],
            "frozen_all_candidate_roc_auc": frozen["test_all_candidate_roc_auc"], "finetuned_all_candidate_roc_auc": result["test_all_candidate_roc_auc"], "paired_auc_difference_finetuned_minus_frozen": float(result["test_all_candidate_roc_auc"]) - float(frozen["test_all_candidate_roc_auc"]),
            "frozen_all_candidate_average_precision": frozen["test_all_candidate_average_precision"], "finetuned_all_candidate_average_precision": result["test_all_candidate_average_precision"], "paired_ap_difference_finetuned_minus_frozen": float(result["test_all_candidate_average_precision"]) - float(frozen["test_all_candidate_average_precision"]),
            "frozen_k20_queue_precision": frozen["all_candidate_queue_precision"], "finetuned_k20_queue_precision": primary_metrics["all_candidate_queue_precision"], "paired_k20_precision_difference": float(primary_metrics["all_candidate_queue_precision"]) - float(frozen["all_candidate_queue_precision"]),
            "frozen_k20_role_qualified_recall": frozen["queued_role_qualified_weed_instance_recall"], "finetuned_k20_role_qualified_recall": primary_metrics["queued_role_qualified_weed_instance_recall"], "paired_k20_recall_difference": float(primary_metrics["queued_role_qualified_weed_instance_recall"]) - float(frozen["queued_role_qualified_weed_instance_recall"]),
        })

    files = {"repeats": output / "finetune_repeat_results.tsv", "history": output / "finetune_epoch_history.tsv", "queue": output / "finetune_official_test_queue.tsv", "predictions": output / "finetune_official_test_predictions.tsv", "paired": output / "paired_frozen_finetuned_differences.tsv"}
    for key, values in (("repeats", result_rows), ("history", history_rows), ("queue", queue_rows), ("predictions", prediction_rows), ("paired", paired_rows)): write_tsv(files[key], values)
    paired_auc = np.asarray([float(row["paired_auc_difference_finetuned_minus_frozen"]) for row in paired_rows])
    paired_precision = np.asarray([float(row["paired_k20_precision_difference"]) for row in paired_rows])
    paired_recall = np.asarray([float(row["paired_k20_recall_difference"]) for row in paired_rows])
    summary = {
        "run_id": args.run_id, "status": "completed_official_spatial_dinov2_last_block_finetune",
        "scope": {"public_data_only": True, "sealed_test_read": False, "official_train_labels_only_for_fitting": True, "official_validation_only_for_epoch_selection": True, "official_test_scoring_once_after_selection": True, "same_prospective_coreset_draws_as_frozen_baseline": True},
        "method": {"backbone": args.model_name, "pretraining": "DINOv2 LVD-142M", "crop": "tight candidate bounding box without padding", "trainable": ["last transformer block", "final norm", "binary head"], "optimizer": "AdamW", "learning_rates": {"last_block_and_norm": 1e-5, "head": 1e-3}, "weight_decay": 1e-4, "epochs": EPOCHS, "best_epoch_selection": "validation AUROC, then AP, then earlier epoch", "batch_size": args.batch_size, "augmentations": ["random horizontal flip", "random vertical flip"], "loss": "class-weighted binary cross entropy", "labels_per_repeat": 300, "repeats": REPEATS, "physical_gpu_ids": list(PHYSICAL_GPUS)},
        "paired_differences_finetuned_minus_frozen": {"auc_values": paired_auc.tolist(), "auc_mean": float(paired_auc.mean()), "auc_empirical_quantiles_2.5_50_97.5": np.percentile(paired_auc, [2.5, 50, 97.5]).tolist(), "k20_precision_values": paired_precision.tolist(), "k20_precision_mean": float(paired_precision.mean()), "k20_role_qualified_recall_values": paired_recall.tolist(), "k20_role_qualified_recall_mean": float(paired_recall.mean())},
        "inputs": {"archive_sha256": sha256(args.archive.resolve()), "manifest_sha256": sha256(args.manifest.resolve()), "candidates_sha256": sha256(args.candidate_root / "candidate_components.tsv"), "selection_ids_sha256": sha256(args.selection_ids.resolve()), "frozen_selection_results_sha256": sha256(args.frozen_selection_results.resolve())},
        "outputs": {key + "_sha256": sha256(path) for key, path in files.items()},
        "checkpoints": {row["checkpoint"]: row["checkpoint_sha256"] for row in result_rows},
        "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version, "numpy": np.__version__, "per_repeat": [{key: row[key] for key in ("repeat", "physical_gpu", "wall_seconds", "peak_cuda_memory_allocated_bytes", "gpu_name", "torch", "timm")} for row in result_rows]},
        "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(args.config.resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
