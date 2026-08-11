#!/usr/bin/env python3
"""Compare modern frozen, source-trained frozen, and structured features."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_frozen_encoder_linear_probe import (  # noqa: E402
    DATES, FEATURES, audit_setting, load_truth, score_linear, fit_linear, stratified_sample, write_tsv,
)
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import (  # noqa: E402
    average_precision, read_tsv, roc_auc, sha256,
)

L2_GRID = (1e-4, 1e-3, 1e-2)
REPEATS = 10
BUDGET = 100
KINDS = ("dinov2_frozen", "source_semantic_frozen", "role_plus_local")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for name in ("project-root", "archive", "manifest", "grid-root", "commissioning-summary",
                 "dinov2-root", "source-embedding-root", "config", "output-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--physical-gpu-id", type=int, required=True)
    p.add_argument("--expected-archive-sha256", required=True)
    p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--expected-commissioning-summary-sha256", required=True)
    return p.parse_args()


def load_embedding(path: Path) -> dict[str, np.ndarray]:
    a = np.load(path); ids = [str(v) for v in a["candidate_ids"]]; matrix = a["embeddings"].astype(np.float32, copy=False)
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or len(ids) != len(set(ids)): raise ValueError({"path": str(path), "shape": matrix.shape})
    return dict(zip(ids, matrix, strict=True))


def select(rows: list[dict[str, object]], dates: tuple[str, ...], kind: str,
           dino: dict[str, np.ndarray], source: dict[str, np.ndarray], ids: set[str] | None = None):
    chosen = [r for r in rows if r["eligible"] and r["date"] in dates and (ids is None or r["candidate_id"] in ids)]
    if kind == "dinov2_frozen": x = np.stack([dino[r["candidate_id"]] for r in chosen])
    elif kind == "source_semantic_frozen": x = np.stack([source[r["candidate_id"]] for r in chosen])
    else: x = np.asarray([[float(r[f]) for f in FEATURES] for r in chosen], dtype=np.float32)
    y = np.asarray([r["truth"] == "weed" for r in chosen], dtype=np.int64)
    return x, y, chosen


def main() -> None:
    args = parse_args(); started = time.perf_counter()
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id) or not torch.cuda.is_available(): raise RuntimeError("GPU assignment mismatch")
    torch.cuda.get_device_properties(0); torch.manual_seed(20260810); device = "cuda:0"
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    if sha256(args.archive.resolve()) != args.expected_archive_sha256 or sha256(args.manifest.resolve()) != args.expected_manifest_sha256 or sha256(args.commissioning_summary.resolve()) != args.expected_commissioning_summary_sha256:
        raise ValueError("Input hash mismatch")
    manifest = read_tsv(args.manifest.resolve()); truth = load_truth(args.archive.resolve(), manifest)
    commission = json.loads(args.commissioning_summary.read_text(encoding="utf-8"))
    settings = {d: commission["per_outer_date"][d]["setting"] for d in DATES}; unique = sorted(set(settings.values()))
    audits = {s: audit_setting(args.grid_root.resolve() / s, manifest, truth, True) for s in unique}
    dino = {s: load_embedding(args.dinov2_root.resolve() / s / "dinov2_roi_embeddings.npz") for s in unique}
    source = {s: load_embedding(args.source_embedding_root.resolve() / s / "frozen_encoder_roi_embeddings.npz") for s in unique}
    full, budget = [], []
    for held in DATES:
        setting = settings[held]; rows = audits[setting]["rows"]; training = tuple(d for d in DATES if d != held)
        for kind in KINDS:
            sensitivity = []
            for l2 in L2_GRID:
                vals = []
                for val in training:
                    x, y, _ = select(rows, tuple(d for d in training if d != val), kind, dino[setting], source[setting])
                    vx, vy, _ = select(rows, (val,), kind, dino[setting], source[setting])
                    vals.append(float(roc_auc(score_linear(fit_linear(x, y, l2, device), vx, device), vy)))
                sensitivity.append({"l2": l2, "mean": float(np.mean(vals))})
            chosen = max(sensitivity, key=lambda r: (r["mean"], -r["l2"]))
            x, y, _ = select(rows, training, kind, dino[setting], source[setting]); tx, ty, test_rows = select(rows, (held,), kind, dino[setting], source[setting])
            pred = score_linear(fit_linear(x, y, float(chosen["l2"]), device), tx, device)
            full.append({"held_out_date": held, "selected_setting": setting, "feature_kind": kind,
                         "training_labels": len(y), "test_candidates": len(ty), "selected_l2": chosen["l2"],
                         "roc_auc": roc_auc(pred, ty), "average_precision": average_precision(pred, ty, [r["candidate_id"] for r in test_rows])})
        for repeat in range(REPEATS):
            sampled = stratified_sample(rows, training, BUDGET, 2026081000 + 100 * DATES.index(held) + repeat)
            for kind in KINDS:
                x, y, _ = select(rows, training, kind, dino[setting], source[setting], sampled)
                tx, ty, test_rows = select(rows, (held,), kind, dino[setting], source[setting])
                pred = score_linear(fit_linear(x, y, 1e-3, device), tx, device)
                budget.append({"held_out_date": held, "selected_setting": setting, "repeat": repeat, "feature_kind": kind,
                               "labels_per_training_date": BUDGET, "training_labels": len(y), "test_candidates": len(ty),
                               "roc_auc": roc_auc(pred, ty), "average_precision": average_precision(pred, ty, [r["candidate_id"] for r in test_rows])})
    paired = []
    for repeat in range(REPEATS):
        means = {k: float(np.mean([float(r["roc_auc"]) for r in budget if r["feature_kind"] == k and int(r["repeat"]) == repeat])) for k in KINDS}
        paired.append({"repeat": repeat, **{k + "_date_macro_roc_auc": v for k, v in means.items()},
                       "role_plus_local_minus_dinov2": means["role_plus_local"] - means["dinov2_frozen"],
                       "dinov2_minus_source_semantic": means["dinov2_frozen"] - means["source_semantic_frozen"]})
    out.mkdir(parents=True); full_p, budget_p, paired_p = out / "full_label_baselines.tsv", out / "same_label_budget.tsv", out / "paired_auc_differences.tsv"
    write_tsv(full_p, full); write_tsv(budget_p, budget); write_tsv(paired_p, paired)
    def summarize(kind: str):
        vals = [r[kind + "_date_macro_roc_auc"] for r in paired]
        return {"mean": float(np.mean(vals)), "q0.025": float(np.quantile(vals, .025)), "q0.975": float(np.quantile(vals, .975))}
    summary = {"run_id": args.run_id, "status": "completed_modern_frozen_baseline_comparison",
               "scope": {"public_data_only": True, "sealed_test_read": False, "held_out_labels_used_for_scoring_only": True,
                         "class_stratified_sampling_requires_known_training_candidate_roles": True},
               "full_label_date_macro": {k: {m: float(np.mean([float(r[m]) for r in full if r["feature_kind"] == k])) for m in ("roc_auc", "average_precision")} for k in KINDS},
               "same_label_budget": {k: summarize(k) for k in KINDS},
               "paired_differences": {name: {"mean": float(np.mean([r[name] for r in paired])), "q0.025": float(np.quantile([r[name] for r in paired], .025)), "q0.975": float(np.quantile([r[name] for r in paired], .975))} for name in ("role_plus_local_minus_dinov2", "dinov2_minus_source_semantic")},
               "inputs": {"archive_sha256": sha256(args.archive.resolve()), "manifest_sha256": sha256(args.manifest.resolve()),
                         "commissioning_summary_sha256": sha256(args.commissioning_summary.resolve()),
                         "dinov2_embeddings_sha256": {s: sha256(args.dinov2_root.resolve() / s / "dinov2_roi_embeddings.npz") for s in unique},
                         "source_embeddings_sha256": {s: sha256(args.source_embedding_root.resolve() / s / "frozen_encoder_roi_embeddings.npz") for s in unique}},
               "outputs": {"full_label_sha256": sha256(full_p), "same_label_budget_sha256": sha256(budget_p), "paired_differences_sha256": sha256(paired_p)},
               "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "physical_gpu_id": args.physical_gpu_id, "gpu_name": torch.cuda.get_device_name(0)},
               "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(args.config.resolve()),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
