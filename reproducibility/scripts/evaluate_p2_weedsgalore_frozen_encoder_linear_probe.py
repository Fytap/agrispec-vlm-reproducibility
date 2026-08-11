#!/usr/bin/env python3
"""Nested frozen-encoder linear-probe and same-label-budget baselines."""
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
from evaluate_p2_weedsgalore_nested_proposal_commissioning import (  # noqa: E402
    DATES,
    FEATURES,
    audit_setting,
    average_precision,
    load_truth,
    read_tsv,
    roc_auc,
    sha256,
    write_tsv,
)


L2_GRID = (1e-4, 1e-3, 1e-2)
LABEL_BUDGET_PER_TRAINING_DATE = 100
REPEATS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("project-root", "archive", "manifest", "grid-root", "commissioning-summary", "embedding-root", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-commissioning-summary-sha256", required=True)
    return parser.parse_args()


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path)
    ids = [str(value) for value in archive["candidate_ids"]]
    matrix = archive["embeddings"].astype(np.float32, copy=False)
    if matrix.shape != (len(ids), 512) or len(ids) != len(set(ids)):
        raise ValueError({"path": str(path), "shape": matrix.shape, "ids": len(ids), "unique": len(set(ids))})
    return dict(zip(ids, matrix, strict=True))


def fit_linear(train_x: np.ndarray, train_y: np.ndarray, l2: float, device: str):
    import torch
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-5)
    x = (x - mean) / scale
    weight = torch.zeros(x.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    positives = y.sum().clamp_min(1.0)
    negatives = (1.0 - y).sum().clamp_min(1.0)
    positive_weight = negatives / positives
    optimizer = torch.optim.LBFGS([weight, bias], lr=0.75, max_iter=50, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=positive_weight)
        loss = loss + float(l2) * weight.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return mean.detach(), scale.detach(), weight.detach(), bias.detach()


def score_linear(model, matrix: np.ndarray, device: str) -> np.ndarray:
    import torch
    mean, scale, weight, bias = model
    with torch.inference_mode():
        x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        return torch.sigmoid(((x - mean) / scale) @ weight + bias).cpu().numpy()


def xy(rows: list[dict[str, object]], dates: tuple[str, ...], embeddings: dict[str, np.ndarray], feature_kind: str):
    selected = [row for row in rows if row["eligible"] and row["date"] in dates]
    if feature_kind == "frozen_encoder":
        matrix = np.stack([embeddings[str(row["candidate_id"])] for row in selected])
    else:
        matrix = np.asarray([[float(row[name]) for name in FEATURES] for row in selected], dtype=np.float32)
    labels = np.asarray([row["truth"] == "weed" for row in selected], dtype=np.int64)
    return matrix, labels, selected


def stratified_sample(rows: list[dict[str, object]], training_dates: tuple[str, ...], budget: int, seed: int) -> set[str]:
    rng = np.random.default_rng(seed)
    chosen: set[str] = set()
    for date in training_dates:
        date_rows = [row for row in rows if row["eligible"] and row["date"] == date]
        positives = [row for row in date_rows if row["truth"] == "weed"]
        negatives = [row for row in date_rows if row["truth"] == "crop"]
        positive_n = min(len(positives), max(1, budget // 2))
        negative_n = min(len(negatives), max(1, budget - positive_n))
        if positive_n + negative_n < budget:
            positive_n = min(len(positives), budget - negative_n)
        for pool, count in ((positives, positive_n), (negatives, negative_n)):
            indices = rng.choice(len(pool), size=count, replace=False)
            chosen.update(str(pool[int(index)]["candidate_id"]) for index in indices)
    return chosen


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id) or not torch.cuda.is_available():
        raise RuntimeError("GPU assignment mismatch")
    torch.manual_seed(20260810)
    device = "cuda:0"
    root, output_dir = args.project_root.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256(args.archive.resolve()) != args.expected_archive_sha256 or sha256(args.manifest.resolve()) != args.expected_manifest_sha256 or sha256(args.commissioning_summary.resolve()) != args.expected_commissioning_summary_sha256:
        raise ValueError("input hash mismatch")

    manifest = read_tsv(args.manifest.resolve())
    truth = load_truth(args.archive.resolve(), manifest)
    commissioning = json.loads(args.commissioning_summary.read_text(encoding="utf-8"))
    selected_settings = {date: commissioning["per_outer_date"][date]["setting"] for date in DATES}
    unique_settings = sorted(set(selected_settings.values()))
    audits = {setting: audit_setting(args.grid_root.resolve() / setting, manifest, truth, True) for setting in unique_settings}
    embedding_maps = {
        setting: load_embeddings(args.embedding_root.resolve() / setting / "frozen_encoder_roi_embeddings.npz")
        for setting in unique_settings
    }
    full_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    for held_out in DATES:
        setting = selected_settings[held_out]
        rows = audits[setting]["rows"]
        embeddings = embedding_maps[setting]
        training_dates = tuple(date for date in DATES if date != held_out)
        inner_records = []
        for l2 in L2_GRID:
            fold_values = []
            for validation in training_dates:
                inner_train = tuple(date for date in training_dates if date != validation)
                train_x, train_y, _ = xy(rows, inner_train, embeddings, "frozen_encoder")
                val_x, val_y, val_rows = xy(rows, (validation,), embeddings, "frozen_encoder")
                prediction = score_linear(fit_linear(train_x, train_y, l2, device), val_x, device)
                fold_values.append(roc_auc(prediction, val_y))
            inner_records.append({"l2": l2, "mean_inner_date_auroc": float(np.mean(fold_values))})
        chosen = max(inner_records, key=lambda row: (row["mean_inner_date_auroc"], -row["l2"]))
        train_x, train_y, _ = xy(rows, training_dates, embeddings, "frozen_encoder")
        test_x, test_y, test_rows = xy(rows, (held_out,), embeddings, "frozen_encoder")
        prediction = score_linear(fit_linear(train_x, train_y, float(chosen["l2"]), device), test_x, device)
        full_rows.append({
            "held_out_date": held_out,
            "selected_setting": setting,
            "eligible_test_candidates": len(test_y),
            "selected_l2": chosen["l2"],
            "mean_inner_date_auroc": chosen["mean_inner_date_auroc"],
            "frozen_encoder_linear_probe_auroc": roc_auc(prediction, test_y),
            "frozen_encoder_linear_probe_average_precision": average_precision(prediction, test_y, [str(row["candidate_id"]) for row in test_rows]),
        })
        for repeat in range(REPEATS):
            selected_ids = stratified_sample(rows, training_dates, LABEL_BUDGET_PER_TRAINING_DATE, 2026081000 + 100 * DATES.index(held_out) + repeat)
            for feature_kind in ("frozen_encoder", "role_plus_local"):
                train_all_x, train_all_y, train_rows = xy(rows, training_dates, embeddings, feature_kind)
                keep = np.asarray([str(row["candidate_id"]) in selected_ids for row in train_rows])
                test_x, test_y, test_rows = xy(rows, (held_out,), embeddings, feature_kind)
                prediction = score_linear(fit_linear(train_all_x[keep], train_all_y[keep], 1e-3, device), test_x, device)
                budget_rows.append({
                    "held_out_date": held_out,
                    "selected_setting": setting,
                    "repeat": repeat,
                    "feature_kind": feature_kind,
                    "labels_per_training_date": LABEL_BUDGET_PER_TRAINING_DATE,
                    "training_labels": int(keep.sum()),
                    "test_candidates": len(test_y),
                    "auroc": roc_auc(prediction, test_y),
                    "average_precision": average_precision(prediction, test_y, [str(row["candidate_id"]) for row in test_rows]),
                })

    output_dir.mkdir(parents=True)
    full_path, budget_path = output_dir / "full_label_nested_linear_probe.tsv", output_dir / "same_label_budget_comparison.tsv"
    write_tsv(full_path, full_rows)
    write_tsv(budget_path, budget_rows)
    budget_summary = {}
    for kind in ("frozen_encoder", "role_plus_local"):
        kind_rows = [row for row in budget_rows if row["feature_kind"] == kind]
        repeat_means = [float(np.mean([row["auroc"] for row in kind_rows if int(row["repeat"]) == repeat])) for repeat in range(REPEATS)]
        budget_summary[kind] = {
            "mean_date_auroc_across_repeats": float(np.mean(repeat_means)),
            "median_repeat_mean_date_auroc": float(np.median(repeat_means)),
            "quantile_0p025": float(np.quantile(repeat_means, 0.025)),
            "quantile_0p975": float(np.quantile(repeat_means, 0.975)),
            "per_date_mean_auroc": {date: float(np.mean([row["auroc"] for row in kind_rows if row["held_out_date"] == date])) for date in DATES},
        }
    summary = {
        "run_id": args.run_id,
        "status": "completed_nested_frozen_encoder_linear_probe_and_same_label_budget_comparison",
        "scope": {"public_data_only": True, "sealed_test_read": False, "outer_date_truth_used_for_offline_scoring_only": True, "encoder_frozen": True, "formal_risk_certificate": False},
        "selection": {"outer": "leave_one_provisional_date_out", "inner_l2_grid": list(L2_GRID), "commissioned_setting_selected_without_held_out_date": True, "same_label_budget_l2_fixed": 1e-3},
        "full_label_frozen_encoder": {
            "mean_date_auroc": float(np.mean([row["frozen_encoder_linear_probe_auroc"] for row in full_rows])),
            "mean_date_average_precision": float(np.mean([row["frozen_encoder_linear_probe_average_precision"] for row in full_rows])),
            "per_date": full_rows,
        },
        "same_label_budget": {"labels_per_training_date": LABEL_BUDGET_PER_TRAINING_DATE, "repeats": REPEATS, "sampling": "same deterministic stratified candidate IDs for both feature sets", "summary": budget_summary},
        "inputs": {"archive_sha256": sha256(args.archive.resolve()), "manifest_sha256": sha256(args.manifest.resolve()), "commissioning_summary_sha256": sha256(args.commissioning_summary.resolve()), "embedding_artifact_sha256": {setting: sha256(args.embedding_root.resolve() / setting / "frozen_encoder_roi_embeddings.npz") for setting in unique_settings}},
        "outputs": {"full_label_tsv_sha256": sha256(full_path), "same_label_budget_tsv_sha256": sha256(budget_path)},
        "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "physical_gpu_id": args.physical_gpu_id, "gpu_name": torch.cuda.get_device_name(0), "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(0)},
        "limitations": ["The four outer groups are acquisition dates from one public source, not independent farms.", "Sampling-repeat quantiles measure label-subsampling variability, not population confidence intervals.", "The frozen encoder was source-trained and the comparison does not represent generic self-supervised embeddings."],
        "script_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
