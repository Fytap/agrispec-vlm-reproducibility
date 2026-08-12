#!/usr/bin/env python3
"""Train a standard Mask R-CNN instance-segmentation queue baseline.

All 104 official WeedsGalore training images provide the public instance and
semantic masks.  Epoch selection uses only the official validation split and
the same fixed K=20, role-qualified, one-to-one instance endpoint as the
semantic baselines.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import train_evaluate_p2_weedsgalore_target_semantic_seed_stability_h200 as common
except ModuleNotFoundError:  # local publication-backup layout
    backup_scripts = (
        Path(__file__).resolve().parents[1]
        / "reproducibility_backups"
        / "H200_seed_stability_20260812_v3"
        / "scripts"
    )
    sys.path.insert(0, str(backup_scripts))
    import train_evaluate_p2_weedsgalore_target_semantic_seed_stability_h200 as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--physical-gpu-id", type=int, choices=(6, 7), required=True)
    parser.add_argument("--coco-weights", type=Path, required=True)
    parser.add_argument("--expected-coco-sha256", required=True)
    return parser.parse_args()


class DetectionDataset:
    @staticmethod
    def build(stems: list[str], root: Path, train: bool):
        import torch
        from torch.utils.data import Dataset

        class Impl(Dataset):
            def __len__(self) -> int:
                return len(stems)

            def __getitem__(self, index: int):
                stem = stems[index]
                paths = common.paths_for_stem(root, stem)
                rgb = common.load_rgb(paths).astype(np.float32) / 255.0
                semantic = common.load_semantic(paths["semantic"])
                with Image.open(paths["instances"]) as image:
                    instance_map = np.asarray(image)
                masks: list[np.ndarray] = []
                labels: list[int] = []
                boxes: list[list[float]] = []
                for value in np.unique(instance_map):
                    if int(value) == 0:
                        continue
                    mask = instance_map == value
                    ys, xs = np.nonzero(mask)
                    if len(xs) == 0:
                        continue
                    crop = int((semantic[mask] == 1).sum())
                    weed = int((semantic[mask] == 2).sum())
                    if crop + weed == 0:
                        continue
                    masks.append(mask)
                    labels.append(1 if crop >= weed else 2)
                    boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
                image_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
                stacked_masks = np.stack(masks) if masks else np.zeros((0, 600, 600), dtype=np.uint8)
                target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.int64),
                    "masks": torch.from_numpy(np.ascontiguousarray(stacked_masks)).to(torch.uint8),
                    "image_id": torch.tensor([index], dtype=torch.int64),
                    "area": torch.tensor([float(mask.sum()) for mask in masks], dtype=torch.float32),
                    "iscrowd": torch.zeros(len(masks), dtype=torch.int64),
                }
                if train and torch.rand(()) < 0.5:
                    image_tensor = image_tensor.flip(-1)
                    target["masks"] = target["masks"].flip(-1)
                    x0 = 600.0 - target["boxes"][:, 2].clone()
                    x1 = 600.0 - target["boxes"][:, 0].clone()
                    target["boxes"][:, 0] = x0
                    target["boxes"][:, 2] = x1
                return image_tensor, target, stem

        return Impl()


def build_model(coco_weights: Path, expected_sha256: str):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        num_classes=91,
        min_size=600,
        max_size=600,
        box_score_thresh=0.05,
        box_detections_per_img=100,
    )
    if common.sha256(coco_weights) != expected_sha256:
        raise ValueError("Frozen Mask R-CNN COCO weight hash mismatch")
    # The official torchvision checkpoint is accepted only after the frozen
    # SHA-256 check above.  Explicit ``weights_only=False`` also supports the
    # legacy tar serialization used by older model-zoo files.
    model.load_state_dict(torch.load(coco_weights, map_location="cpu", weights_only=False), strict=True)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 3)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, 3)
    return model


def deterministic_matching(
    accepted: list[dict[str, Any]], edges: dict[str, set[str]]
) -> list[tuple[str, str]]:
    """Lexicographically deterministic maximum-cardinality bipartite matching."""
    owner: dict[str, str] = {}

    def augment(candidate_id: str, visited: set[str]) -> bool:
        for truth_id in sorted(edges.get(candidate_id, set())):
            if truth_id in visited:
                continue
            visited.add(truth_id)
            previous = owner.get(truth_id)
            if previous is None or augment(previous, visited):
                owner[truth_id] = candidate_id
                return True
        return False

    for row in sorted(accepted, key=lambda item: str(item["candidate_id"])):
        augment(str(row["candidate_id"]), set())
    return sorted((candidate_id, truth_id) for truth_id, candidate_id in owner.items())


def evaluate_predictions(
    predictions: dict[str, list[dict[str, Any]]],
    root: Path,
    include_candidates: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    totals = Counter()
    dates: dict[str, Counter] = defaultdict(Counter)
    tile_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    for stem in sorted(predictions):
        semantic, instance_map, instances = common.truth_instances(stem, root)
        by_value = {record.value: record for record in instances}
        weed_ids = {record.instance_id for record in instances if record.role == "weed"}
        crop_ids = {record.instance_id for record in instances if record.role == "crop"}
        spatial_edges: dict[str, set[str]] = defaultdict(set)
        qualified_edges: dict[str, set[str]] = defaultdict(set)
        candidates: list[dict[str, Any]] = []
        for index, prediction in enumerate(predictions[stem], 1):
            mask = np.asarray(prediction["mask"], dtype=bool)
            area = int(mask.sum())
            if area == 0:
                continue
            ys, xs = np.nonzero(mask)
            crop_pixels = int((semantic[mask] == 1).sum())
            weed_pixels = int((semantic[mask] == 2).sum())
            labeled = crop_pixels + weed_pixels
            labeled_coverage = labeled / max(1, area)
            crop_purity = crop_pixels / max(1, labeled)
            weed_purity = weed_pixels / max(1, labeled)
            truth_role = (
                "weed"
                if labeled_coverage >= common.ROLE_LABELED_COVERAGE and weed_purity >= common.ROLE_PURITY
                else "crop"
                if labeled_coverage >= common.ROLE_LABELED_COVERAGE and crop_purity >= common.ROLE_PURITY
                else "ambiguous_or_background"
            )
            predicted_label = int(prediction["label"])
            detection_score = float(prediction["score"])
            queue_score = detection_score if predicted_label == 2 else -detection_score
            candidate_id = f"{stem}:maskrcnn:C{index:05d}"
            row = {
                "candidate_id": candidate_id,
                "file": stem,
                "date": stem[:10],
                "area_pixels": area,
                "x0": int(xs.min()),
                "y0": int(ys.min()),
                "x1_exclusive": int(xs.max()) + 1,
                "y1_exclusive": int(ys.max()) + 1,
                "score": queue_score,
                "detection_score": detection_score,
                "predicted_role": "weed" if predicted_label == 2 else "crop",
                "candidate_truth": truth_role,
                "candidate_labeled_coverage": labeled_coverage,
                "candidate_crop_purity": crop_purity,
                "candidate_weed_purity": weed_purity,
                "crop_fraction_candidate": crop_pixels / max(1, area),
                "weed_fraction_candidate": weed_pixels / max(1, area),
            }
            candidates.append(row)
            values, counts = np.unique(instance_map[mask], return_counts=True)
            for value, overlap in zip(values.tolist(), counts.tolist()):
                record = by_value.get(int(value))
                if record is None or record.role != "weed":
                    continue
                coverage = int(overlap) / max(1, record.area)
                if coverage >= common.ROLE_INSTANCE_COVERAGE:
                    spatial_edges[candidate_id].add(record.instance_id)
                    if labeled_coverage >= common.ROLE_LABELED_COVERAGE and weed_purity >= common.ROLE_PURITY:
                        qualified_edges[candidate_id].add(record.instance_id)
                    edge_rows.append(
                        {
                            "record_type": "qualified_edge_candidate",
                            "file": stem,
                            "candidate_id": candidate_id,
                            "truth_instance_id": record.instance_id,
                            "instance_coverage": coverage,
                            "role_qualified": int(record.instance_id in qualified_edges[candidate_id]),
                        }
                    )
        candidates.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
        accepted = candidates[: common.QUEUE_K]
        spatial = set().union(*(spatial_edges.get(str(row["candidate_id"]), set()) for row in accepted)) if accepted else set()
        qualified = set().union(*(qualified_edges.get(str(row["candidate_id"]), set()) for row in accepted)) if accepted else set()
        matches = deterministic_matching(accepted, qualified_edges)
        for candidate_id, truth_id in matches:
            matching_rows.append({
                "record_type": "selected_maximum_matching",
                "file": stem,
                "candidate_id": candidate_id,
                "truth_instance_id": truth_id,
                "instance_coverage": "",
                "role_qualified": 1,
            })
        eligible = sum(str(row["candidate_truth"]) == "weed" for row in accepted)
        crop_overlap = [float(row["crop_fraction_candidate"]) for row in accepted]
        tile = {
            "file": stem,
            "date": stem[:10],
            "weed_instances": len(weed_ids),
            "crop_instances": len(crop_ids),
            "candidates": len(candidates),
            "accepted_candidates": len(accepted),
            "eligible_weed_candidates": eligible,
            "all_candidate_precision": eligible / max(1, len(accepted)),
            "many_to_one_spatial_recalled": len(spatial),
            "many_to_one_role_qualified_recalled": len(qualified),
            "one_to_one_role_qualified_recalled": len(matches),
            "crop_overlap_scene": int(any(value > 0 for value in crop_overlap)),
            "maximum_crop_fraction": max(crop_overlap, default=0.0),
        }
        tile_rows.append(tile)
        for key in (
            "weed_instances", "crop_instances", "candidates", "accepted_candidates",
            "eligible_weed_candidates", "many_to_one_spatial_recalled",
            "many_to_one_role_qualified_recalled", "one_to_one_role_qualified_recalled",
            "crop_overlap_scene",
        ):
            totals[key] += int(tile[key])
            dates[stem[:10]][key] += int(tile[key])
        totals["images"] += 1
        dates[stem[:10]]["images"] += 1
        if include_candidates:
            for rank, row in enumerate(candidates, 1):
                candidate_rows.append({**row, "rank_within_image": rank, "accepted_K20": int(rank <= common.QUEUE_K)})
    summary = {
        "images": totals["images"],
        "weed_instances": totals["weed_instances"],
        "crop_instances": totals["crop_instances"],
        "candidates": totals["candidates"],
        "candidates_per_image": totals["candidates"] / max(1, totals["images"]),
        "accepted_candidates": totals["accepted_candidates"],
        "eligible_weed_candidates": totals["eligible_weed_candidates"],
        "K20_all_candidate_precision": totals["eligible_weed_candidates"] / max(1, totals["accepted_candidates"]),
        "K20_many_to_one_spatial_recall": totals["many_to_one_spatial_recalled"] / max(1, totals["weed_instances"]),
        "K20_many_to_one_role_qualified_recall": totals["many_to_one_role_qualified_recalled"] / max(1, totals["weed_instances"]),
        "K20_one_to_one_role_qualified_recall": totals["one_to_one_role_qualified_recalled"] / max(1, totals["weed_instances"]),
        "K20_crop_overlap_scene_frequency": totals["crop_overlap_scene"] / max(1, totals["images"]),
    }
    date_rows = []
    for date, values in sorted(dates.items()):
        date_rows.append({
            "date": date,
            "images": values["images"],
            "weed_instances": values["weed_instances"],
            "candidates": values["candidates"],
            "accepted_candidates": values["accepted_candidates"],
            "precision": values["eligible_weed_candidates"] / max(1, values["accepted_candidates"]),
            "many_to_one_spatial_recall": values["many_to_one_spatial_recalled"] / max(1, values["weed_instances"]),
            "many_to_one_role_qualified_recall": values["many_to_one_role_qualified_recalled"] / max(1, values["weed_instances"]),
            "one_to_one_role_qualified_recall": values["one_to_one_role_qualified_recalled"] / max(1, values["weed_instances"]),
            "crop_overlap_scene_frequency": values["crop_overlap_scene"] / max(1, values["images"]),
        })
    return summary, tile_rows, candidate_rows, date_rows, edge_rows + matching_rows


def predict(model, stems: list[str], root: Path, device) -> tuple[dict[str, list[dict[str, Any]]], float, int]:
    import torch

    model.eval()
    output: dict[str, list[dict[str, Any]]] = {}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for stem in stems:
            rgb = common.load_rgb(common.paths_for_stem(root, stem)).astype(np.float32) / 255.0
            x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model([x])[0]
            records = []
            for label, score, mask in zip(prediction["labels"], prediction["scores"], prediction["masks"]):
                records.append({
                    "label": int(label),
                    "score": float(score),
                    "mask": (mask[0] >= 0.5).cpu().numpy(),
                })
            output[stem] = records
    return output, time.perf_counter() - started, int(torch.cuda.max_memory_allocated(device))


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise RuntimeError("Frozen GPU assignment mismatch")
    import torch
    from torch.utils.data import DataLoader

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    root = args.project_root.resolve()
    archive = args.archive.resolve()
    config = args.config.resolve()
    output = args.output_dir.resolve()
    cache = args.cache_dir.resolve()
    if common.sha256(archive) != args.expected_archive_sha256:
        raise ValueError("Frozen public archive hash mismatch")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    dataset_root = common.extract_public_archive(archive, cache, args.expected_archive_sha256)
    train_stems = common.read_split(dataset_root, "train")
    val_stems = common.read_split(dataset_root, "val")
    test_stems = common.read_split(dataset_root, "test")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    dataset = DetectionDataset.build(train_stems, dataset_root, train=True)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    coco_weights = args.coco_weights.resolve()
    model = build_model(coco_weights, args.expected_coco_sha256).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    history: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    best_key: tuple[Any, ...] | None = None
    best_row: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for images, targets, _ in loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        validation_predictions, seconds, peak = predict(model, val_stems, dataset_root, device)
        summary, _, _, _, _ = evaluate_predictions(validation_predictions, dataset_root)
        row = {"epoch": epoch, **summary}
        validation_rows.append(row)
        key = (
            float(summary["K20_one_to_one_role_qualified_recall"]),
            float(summary["K20_many_to_one_role_qualified_recall"]),
            float(summary["K20_all_candidate_precision"]),
            -float(summary["candidates_per_image"]),
            -int(epoch),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_row = row
            torch.save(
                {
                    "architecture": "Mask-RCNN-ResNet50-FPN-v2",
                    "model_state": model.state_dict(),
                    "selection": row,
                    "seed": args.seed,
                },
                output / "best_validation_queue.pt",
            )
        history_row = {
            "epoch": epoch,
            "mean_train_loss": total_loss / max(1, batches),
            "validation_seconds": seconds,
            "validation_peak_memory_bytes": peak,
            "validation_K20_one_to_one_role_qualified_recall": summary["K20_one_to_one_role_qualified_recall"],
            "global_best_epoch_so_far": int(best_row["epoch"] if best_row else epoch),
        }
        history.append(history_row)
        print(json.dumps(history_row), flush=True)
    if best_row is None:
        raise RuntimeError("No validation checkpoint selected")
    checkpoint = torch.load(output / "best_validation_queue.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_predictions, test_seconds, test_peak = predict(model, test_stems, dataset_root, device)
    test_summary, tile_rows, candidate_rows, date_rows, audit_rows = evaluate_predictions(
        test_predictions, dataset_root, include_candidates=True
    )
    common.write_tsv(output / "training_history.tsv", history)
    common.write_tsv(output / "validation_epoch_selection.tsv", validation_rows)
    common.write_tsv(output / "official_test_tile_metrics.tsv", tile_rows)
    common.write_tsv(output / "official_test_date_metrics.tsv", date_rows)
    common.write_tsv(output / "official_test_candidates.tsv", candidate_rows)
    if audit_rows:
        common.write_tsv(output / "official_test_matching_audit.tsv", audit_rows)
    governance = {
        "analysis_class": "post_test_strong_baseline_extension",
        "selection_role": "official validation only",
        "test_use": "offline scoring after epoch selection",
        "external_data_used": False,
    }
    (output / "test_governance.json").write_text(json.dumps(governance, indent=2) + "\n", encoding="utf-8")
    result = {
        "run_id": args.run_id,
        "status": "completed_post_test_strong_baseline_extension",
        "git_commit": common.git_commit(root),
        "inputs": {
            "archive_sha256": common.sha256(archive),
            "config": str(config),
            "config_sha256": common.sha256(config),
            "script_sha256": common.sha256(Path(__file__).resolve()),
            "coco_weights": str(coco_weights),
            "coco_weights_sha256": common.sha256(coco_weights),
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "model": {
            "architecture": "Mask-RCNN-ResNet50-FPN-v2",
            "initialization": "torchvision COCO V1 public weights",
            "checkpoint_sha256": common.sha256(output / "best_validation_queue.pt"),
        },
        "selected_on_validation": best_row,
        "official_test_queue_metrics": test_summary,
        "runtime": {
            "test_total_seconds": test_seconds,
            "test_seconds_per_tile": test_seconds / len(test_stems),
            "test_peak_gpu_memory_bytes": test_peak,
            "run_wall_seconds": time.perf_counter() - started,
            "physical_gpu_id": args.physical_gpu_id,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "governance": governance,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "SHA256SUMS.tsv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("path\tsha256\n")
        for path in sorted(output.iterdir()):
            if path.name != "SHA256SUMS.tsv":
                stream.write(f"{path.name}\t{common.sha256(path)}\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
