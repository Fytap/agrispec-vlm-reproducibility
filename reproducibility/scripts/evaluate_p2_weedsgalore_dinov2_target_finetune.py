#!/usr/bin/env python3
"""Last-block DINOv2 target fine-tuning on commissioned candidate crops."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_frozen_encoder_linear_probe import DATES, audit_setting, load_truth, stratified_sample, write_tsv  # noqa: E402
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import average_precision, read_tsv, roc_auc, sha256  # noqa: E402

REPEATS, BUDGET, EPOCHS = 3, 100, 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for name in ("project-root", "archive", "data-root", "manifest", "grid-root", "commissioning-summary", "config", "output-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--physical-gpu-id", type=int, required=True)
    p.add_argument("--expected-archive-sha256", required=True); p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--expected-commissioning-summary-sha256", required=True)
    p.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    return p.parse_args()


def load_images(manifest: list[dict[str, str]], root: Path) -> dict[str, np.ndarray]:
    result = {}
    for row in manifest:
        bands = json.loads(row["source_image_relpaths_json"])
        arrays = [np.asarray(Image.open(root / bands[c]), dtype=np.float32) / 65535. for c in ("R", "G", "B")]
        result[row["sample_id"]] = np.clip(np.stack(arrays, -1) * 255., 0, 255).astype(np.uint8)
    return result


def main() -> None:
    args = parse_args(); started = time.perf_counter()
    import torch
    import timm
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import v2
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id) or not torch.cuda.is_available(): raise RuntimeError("GPU assignment mismatch")
    torch.cuda.get_device_properties(0); device = torch.device("cuda:0")
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    if sha256(args.archive.resolve()) != args.expected_archive_sha256 or sha256(args.manifest.resolve()) != args.expected_manifest_sha256 or sha256(args.commissioning_summary.resolve()) != args.expected_commissioning_summary_sha256:
        raise ValueError("Input hash mismatch")
    manifest = read_tsv(args.manifest.resolve()); truth = load_truth(args.archive.resolve(), manifest)
    images = load_images(manifest, args.data_root.resolve())
    commission = json.loads(args.commissioning_summary.read_text(encoding="utf-8")); settings = {d: commission["per_outer_date"][d]["setting"] for d in DATES}
    audits = {}; candidates = {}
    for setting in sorted(set(settings.values())):
        path = args.grid_root.resolve() / setting; audits[setting] = audit_setting(path, manifest, truth, True)
        candidates[setting] = {r["candidate_id"]: r for r in read_tsv(path / "candidate_components.tsv")}
    train_tf = v2.Compose([v2.ToImage(), v2.Resize((224, 224), antialias=True), v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
                           v2.ToDtype(torch.float32, scale=True), v2.Normalize((.485,.456,.406),(.229,.224,.225))])
    test_tf = v2.Compose([v2.ToImage(), v2.Resize((224, 224), antialias=True), v2.ToDtype(torch.float32, scale=True),
                          v2.Normalize((.485,.456,.406),(.229,.224,.225))])
    class Crops(Dataset):
        def __init__(self, rows, box_map, transform): self.rows, self.box_map, self.transform = rows, box_map, transform
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            row = self.rows[i]; b = self.box_map[row["candidate_id"]]; im = images[row["file"]]
            x0,y0,x1,y1 = (int(float(b[k])) for k in ("x0","y0","x1_exclusive","y1_exclusive"))
            crop = im[max(0,y0):max(y0+1,y1), max(0,x0):max(x0+1,x1)]
            return self.transform(crop), torch.tensor(row["truth"] == "weed", dtype=torch.float32), row["candidate_id"]
    results, prediction_rows, checkpoint_hashes = [], [], {}
    out.mkdir(parents=True); checkpoint_dir = out / "checkpoints"; checkpoint_dir.mkdir()
    for held in DATES:
        setting = settings[held]; rows = audits[setting]["rows"]; boxes = candidates[setting]; training = tuple(d for d in DATES if d != held)
        all_test_rows = [r for r in rows if r["date"] == held]
        test_rows = [r for r in all_test_rows if r["eligible"]]
        for repeat in range(REPEATS):
            seed = 2026081000 + 100 * DATES.index(held) + repeat; torch.manual_seed(seed); np.random.seed(seed)
            ids = stratified_sample(rows, training, BUDGET, seed); train_rows = [r for r in rows if r["eligible"] and r["date"] in training and r["candidate_id"] in ids]
            model = timm.create_model(args.model_name, pretrained=True, num_classes=0, img_size=224).to(device)
            for p in model.parameters(): p.requires_grad_(False)
            for p in model.blocks[-1].parameters(): p.requires_grad_(True)
            for p in model.norm.parameters(): p.requires_grad_(True)
            head = nn.Linear(model.num_features, 1).to(device)
            optimizer = torch.optim.AdamW([{"params": model.blocks[-1].parameters(), "lr": 1e-5},
                                           {"params": model.norm.parameters(), "lr": 1e-5},
                                           {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
            loader = DataLoader(Crops(train_rows, boxes, train_tf), batch_size=64, shuffle=True, num_workers=0)
            losses = []
            model.train(); head.train()
            for _ in range(EPOCHS):
                epoch = []
                for x,y,_ in loader:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = head(model(x.to(device))).squeeze(1); loss = nn.functional.binary_cross_entropy_with_logits(logits, y.to(device))
                    loss.backward(); optimizer.step(); epoch.append(float(loss.detach()))
                losses.append(float(np.mean(epoch)))
            model.eval(); head.eval(); pred, labels, ids_out = [], [], []
            with torch.inference_mode():
                for x,y,cids in DataLoader(Crops(all_test_rows, boxes, test_tf), batch_size=128, shuffle=False, num_workers=0):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16): score = torch.sigmoid(head(model(x.to(device))).squeeze(1))
                    pred.extend(score.float().cpu().tolist()); labels.extend(y.int().tolist()); ids_out.extend(cids)
            ckpt = checkpoint_dir / f"{held}_repeat{repeat}.pt"
            torch.save({"model_name": args.model_name, "held_out_date": held, "repeat": repeat, "seed": seed,
                        "trainable_model_state": {k: v.cpu() for k,v in model.state_dict().items() if k.startswith("blocks.11.") or k.startswith("norm.")},
                        "head_state": {k: v.cpu() for k,v in head.state_dict().items()}, "sampled_candidate_ids_sha256": hashlib.sha256(("\n".join(sorted(ids))+"\n").encode()).hexdigest()}, ckpt)
            checkpoint_hashes[ckpt.name] = sha256(ckpt)
            row_by_id = {r["candidate_id"]: r for r in all_test_rows}
            eligible_indices = [i for i,cid in enumerate(ids_out) if row_by_id[cid]["eligible"]]
            y_np, p_np = np.asarray(labels)[eligible_indices], np.asarray(pred)[eligible_indices]
            prediction_rows.extend({"held_out_date": held, "selected_setting": setting, "repeat": repeat,
                                    "candidate_id": cid, "dataset_role_target": row["truth"],
                                    "score": f"{score:.10f}"}
                                   for cid, row, score in zip(ids_out, all_test_rows, pred, strict=True))
            results.append({"held_out_date": held, "selected_setting": setting, "repeat": repeat, "seed": seed,
                            "training_labels": len(train_rows), "test_candidates": len(test_rows), "epochs": EPOCHS,
                            "final_training_loss": losses[-1], "roc_auc": roc_auc(p_np, y_np),
                            "average_precision": average_precision(p_np, y_np, [ids_out[i] for i in eligible_indices]), "checkpoint": ckpt.name,
                            "checkpoint_sha256": checkpoint_hashes[ckpt.name]})
            del model, head, optimizer; torch.cuda.empty_cache()
    table = out / "target_finetune_by_date_repeat.tsv"; prediction_table = out / "outer_predictions.tsv"
    write_tsv(table, results); write_tsv(prediction_table, prediction_rows)
    repeat_macro = [float(np.mean([float(r["roc_auc"]) for r in results if int(r["repeat"]) == rep])) for rep in range(REPEATS)]
    summary = {"run_id": args.run_id, "status": "completed_dinov2_last_block_target_finetune",
               "scope": {"public_data_only": True, "sealed_test_read": False, "held_out_date_labels_used_for_scoring_only": True,
                         "training_date_candidate_roles_used_for_stratified_sampling_and_training": True},
               "method": {"backbone": args.model_name, "trainable": ["last_transformer_block", "final_norm", "binary_head"],
                          "labels_per_training_date": BUDGET, "repeats": REPEATS, "epochs": EPOCHS},
               "date_macro_roc_auc_by_repeat": repeat_macro, "mean_date_macro_roc_auc": float(np.mean(repeat_macro)),
               "range_date_macro_roc_auc": [min(repeat_macro), max(repeat_macro)],
               "per_date_mean": {d: {m: float(np.mean([float(r[m]) for r in results if r["held_out_date"] == d])) for m in ("roc_auc", "average_precision")} for d in DATES},
               "inputs": {"archive_sha256": sha256(args.archive.resolve()), "manifest_sha256": sha256(args.manifest.resolve()),
                         "commissioning_summary_sha256": sha256(args.commissioning_summary.resolve())},
               "outputs": {"results_tsv_sha256": sha256(table), "outer_predictions_sha256": sha256(prediction_table),
                           "checkpoint_sha256": checkpoint_hashes},
               "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version, "numpy": np.__version__,
                           "torch": torch.__version__, "timm": timm.__version__, "physical_gpu_id": args.physical_gpu_id,
                           "gpu_name": torch.cuda.get_device_name(0)},
               "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(args.config.resolve()),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
