#!/usr/bin/env python3
"""Revision-10 robustness and allocation audit for the fixed WeedsGalore queue.

This script consumes only already-versioned prediction and matching summaries. It
does not regenerate candidates, change thresholds, select a ranker, or access
unseen data. All outputs are explicitly post-test descriptive diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


RUN_ID = "P2_WEEDSGALORE_REVISION10_ROBUST_QUEUE_20260811_V1"
DEFAULT_CONFIG = Path(
    "configs/experiments/p2_weedsgalore_revision10_robust_queue_v1.yaml"
)
DEFAULT_QUEUE_RUN = Path(
    "results/p2_development/"
    "P2_WEEDSGALORE_REVISION8_QUEUE_POLICY_TILE_AUDIT_20260810_v1"
)
DEFAULT_SEMANTIC_RUN = Path(
    "results/p2_development/"
    "P2_WEEDSGALORE_TARGET_SEMANTIC_QUEUE_BASELINE_20260810_v3"
)
DEFAULT_OUTPUT = Path(
    "results/p2_development/P2_WEEDSGALORE_REVISION10_ROBUST_QUEUE_20260811_v1"
)

BENEFIT_METRICS = [
    "all_candidate_queue_precision",
    "queued_spatial_weed_instance_recall",
    "queued_role_qualified_weed_instance_recall",
    "one_to_one_role_qualified_recall",
]
COST_METRICS = ["crop_overlap_scene_frequency_0.10"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return value or "UNCOMMITTED_NO_HEAD"
    except (OSError, subprocess.CalledProcessError):
        return "UNCOMMITTED_NO_HEAD"


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or float(values.sum()) == 0.0:
        return 0.0
    return float(np.abs(values[:, None] - values[None, :]).sum() / (2 * values.size * values.sum()))


def safe_metric(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, str]:
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return math.nan, math.nan, "undefined_single_class"
    return (
        float(roc_auc_score(y_true, y_score)),
        float(average_precision_score(y_true, y_score)),
        "defined",
    )


def discrimination_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(aggregation: str, stratum: str, frame: pd.DataFrame) -> None:
        y_true = (frame["dataset_role_target"] == "weed").astype(int).to_numpy()
        y_score = frame["score"].astype(float).to_numpy()
        auc, ap, status = safe_metric(y_true, y_score)
        rows.append(
            {
                "aggregation": aggregation,
                "stratum": stratum,
                "candidates": int(len(frame)),
                "weed_candidates": int(y_true.sum()),
                "nonweed_candidates": int(len(y_true) - y_true.sum()),
                "auc": auc,
                "average_precision": ap,
                "status": status,
            }
        )

    add("pooled_test", "ALL", predictions)
    for date, frame in predictions.groupby("date", sort=True):
        add("acquisition_date", str(date), frame)
    for tile, frame in predictions.groupby("file", sort=True):
        add("tile", str(tile), frame)
    return pd.DataFrame(rows)


def discrimination_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for aggregation in ["acquisition_date", "tile"]:
        frame = rows[(rows["aggregation"] == aggregation) & (rows["status"] == "defined")]
        for metric in ["auc", "average_precision"]:
            values = frame[metric].astype(float)
            output.append(
                {
                    "aggregation": aggregation,
                    "metric": metric,
                    "eligible_strata": int(len(values)),
                    "total_strata": int((rows["aggregation"] == aggregation).sum()),
                    "macro_mean": float(values.mean()) if len(values) else math.nan,
                    "worst_stratum": float(values.min()) if len(values) else math.nan,
                    "best_stratum": float(values.max()) if len(values) else math.nan,
                    "interpretation": "descriptive_post_test_no_population_inference",
                }
            )
    return pd.DataFrame(output)


def policy_macro_worst(date_patch: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for aggregation in ["acquisition_date", "test_patch_proxy"]:
        subset = date_patch[date_patch["aggregation"] == aggregation]
        for policy, frame in subset.groupby("policy", sort=True):
            for metric in BENEFIT_METRICS + COST_METRICS:
                values = frame[metric].astype(float)
                lower_is_better = metric in COST_METRICS
                worst = float(values.max() if lower_is_better else values.min())
                output.append(
                    {
                        "policy": policy,
                        "aggregation": aggregation,
                        "metric": metric,
                        "strata": int(len(values)),
                        "macro_mean": float(values.mean()),
                        "worst_stratum_value": worst,
                        "best_stratum_value": float(values.min() if lower_is_better else values.max()),
                        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                        "interpretation": "descriptive_post_test_no_population_inference",
                    }
                )
    return pd.DataFrame(output)


def allocation_audit(per_tile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    concentration: list[dict[str, object]] = []
    by_date: list[dict[str, object]] = []
    for policy, frame in per_tile.groupby("policy", sort=True):
        counts = frame["accepted_candidates"].astype(int).to_numpy()
        date_counts = frame.groupby("date", sort=True)["accepted_candidates"].sum()
        total = int(counts.sum())
        shares = date_counts / total if total else date_counts.astype(float)
        concentration.append(
            {
                "policy": policy,
                "tiles": int(len(counts)),
                "accepted_candidates": total,
                "tiles_with_zero": int((counts == 0).sum()),
                "tiles_below_5": int((counts < 5).sum()),
                "tiles_below_10": int((counts < 10).sum()),
                "minimum_per_tile": int(counts.min()),
                "q25_per_tile": float(np.quantile(counts, 0.25)),
                "median_per_tile": float(np.median(counts)),
                "q75_per_tile": float(np.quantile(counts, 0.75)),
                "maximum_per_tile": int(counts.max()),
                "allocation_gini": gini(counts),
                "allocation_coefficient_of_variation": float(np.std(counts, ddof=0) / np.mean(counts)) if np.mean(counts) else math.nan,
                "maximum_date_share": float(shares.max()) if len(shares) else math.nan,
                "maximum_date": str(shares.idxmax()) if len(shares) else "",
                "interpretation": "descriptive_batch_allocation_diagnostic",
            }
        )
        for date, value in date_counts.items():
            by_date.append(
                {
                    "policy": policy,
                    "date": str(date),
                    "accepted_candidates": int(value),
                    "share_of_policy_budget": float(value / total) if total else math.nan,
                }
            )
    return pd.DataFrame(concentration), pd.DataFrame(by_date)


def semantic_robustness(date_metrics: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    benefit = [
        "precision",
        "many_to_one_spatial_recall",
        "many_to_one_role_qualified_recall",
        "one_to_one_role_qualified_recall",
    ]
    cost = ["crop_overlap_scene_frequency"]
    for metric in benefit + cost:
        values = date_metrics[metric].astype(float)
        lower_is_better = metric in cost
        output.append(
            {
                "model": "target_semantic_resnet18_unet_single_run",
                "aggregation": "acquisition_date",
                "metric": metric,
                "dates": int(len(values)),
                "macro_mean": float(values.mean()),
                "worst_date_value": float(values.max() if lower_is_better else values.min()),
                "best_date_value": float(values.min() if lower_is_better else values.max()),
                "direction": "lower_is_better" if lower_is_better else "higher_is_better",
                "training_variability": "not_estimated_one_completed_training_run",
            }
        )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--queue-run", type=Path, default=DEFAULT_QUEUE_RUN)
    parser.add_argument("--semantic-run", type=Path, default=DEFAULT_SEMANTIC_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    inputs = {
        "config": args.config,
        "queue_comparison": args.queue_run / "official_test_queue_policy_comparison.tsv",
        "queue_per_tile": args.queue_run / "official_test_queue_policy_per_tile.tsv",
        "queue_date_patch": args.queue_run / "official_test_queue_policy_date_patch.tsv",
        "queue_predictions": args.queue_run / "official_test_queue_selected_predictions.tsv",
        "semantic_date": args.semantic_run / "official_test_date_metrics.tsv",
        "semantic_tile": args.semantic_run / "official_test_tile_metrics.tsv",
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing input {name}: {path}")

    comparison = pd.read_csv(inputs["queue_comparison"], sep="\t")
    per_tile = pd.read_csv(inputs["queue_per_tile"], sep="\t")
    date_patch = pd.read_csv(inputs["queue_date_patch"], sep="\t")
    predictions = pd.read_csv(inputs["queue_predictions"], sep="\t")
    semantic_date = pd.read_csv(inputs["semantic_date"], sep="\t")

    macro = policy_macro_worst(date_patch)
    concentration, allocation_by_date = allocation_audit(per_tile)
    discrimination = discrimination_rows(predictions)
    discrimination_agg = discrimination_summary(discrimination)
    semantic = semantic_robustness(semantic_date)

    comparison.to_csv(args.output / "queue_policy_pooled.tsv", sep="\t", index=False)
    date_patch.to_csv(args.output / "queue_policy_date_patch.tsv", sep="\t", index=False)
    macro.to_csv(args.output / "queue_policy_macro_worst.tsv", sep="\t", index=False)
    per_tile.to_csv(args.output / "queue_policy_per_tile.tsv", sep="\t", index=False)
    concentration.to_csv(args.output / "queue_policy_allocation_concentration.tsv", sep="\t", index=False)
    allocation_by_date.to_csv(args.output / "queue_policy_allocation_by_date.tsv", sep="\t", index=False)
    discrimination.to_csv(args.output / "candidate_discrimination_by_stratum.tsv", sep="\t", index=False)
    discrimination_agg.to_csv(args.output / "candidate_discrimination_summary.tsv", sep="\t", index=False)
    semantic_date.to_csv(args.output / "target_semantic_date_metrics.tsv", sep="\t", index=False)
    semantic.to_csv(args.output / "target_semantic_macro_worst.tsv", sep="\t", index=False)

    summary = {
        "run_id": RUN_ID,
        "git_commit": git_commit(),
        "analysis_class": "reviewer_requested_post_test_diagnostic",
        "governance": {
            "changes_frozen_protocol": False,
            "opens_new_test_data": False,
            "uses_test_for_selection": False,
            "test_use": "descriptive_post_test_diagnostic_only",
            "population_p_values": False,
        },
        "primary_policy": "bounded_global_min5_max50_N520",
        "unconstrained_global_policy_role": "diagnostic_upper_bound_only",
        "inputs": {
            name: {"path": str(path).replace("\\", "/"), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "headline_allocation": concentration.set_index("policy").to_dict(orient="index"),
        "notes": [
            "Macro and worst-stratum values are descriptive across four acquisition dates or two patch proxies.",
            "Candidate AUC/AP are defined only for strata containing both weed and non-weed candidates.",
            "The target-semantic baseline has one completed training run; training-seed variability is not estimated.",
            "The official split contains one field and two held-out spatial patches.",
        ],
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({"run_id": RUN_ID, "output": str(args.output), "files": len(list(args.output.iterdir()))}, indent=2))


if __name__ == "__main__":
    main()
