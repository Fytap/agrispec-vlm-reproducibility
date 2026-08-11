#!/usr/bin/env python3
"""Replay the revision-7 evidence summaries from archived TSV files.

This script uses only the Python standard library. It does not require or read
dataset imagery, model checkpoints, embeddings, or target masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def main() -> None:
    args = arguments()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    base = root / "results" / "p2_development"

    proposal_path = base / "P2_WEEDSGALORE_OFFICIAL_SPATIAL_PROPOSAL_AUDIT_20260810_v1" / "official_split_proposal_layers.tsv"
    matched_path = base / "P2_WEEDSGALORE_OFFICIAL_SPATIAL_MATCHED_PROPOSAL_QUEUE_20260810_v1" / "matched_weight_queue_curves.tsv"
    ranking_path = base / "P2_WEEDSGALORE_OFFICIAL_SPATIAL_QUEUE_20260810_v2" / "official_test_candidate_ranking.tsv"
    queue_path = base / "P2_WEEDSGALORE_OFFICIAL_SPATIAL_QUEUE_20260810_v2" / "official_test_queue_curves.tsv"
    labels_path = base / "P2_WEEDSGALORE_OFFICIAL_SPATIAL_QUEUE_20260810_v2" / "prospective_label_selection_results.tsv"
    paired_path = base / "P2_WEEDSGALORE_OFFICIAL_SPATIAL_DINOV2_FINETUNE_20260810_v1" / "paired_frozen_finetuned_differences.tsv"
    sensitivity_path = base / "P2_WEEDSGALORE_OFFICIAL_QUEUE_CONTRACT_SENSITIVITY_20260810_v1" / "queued_matching_contract_sensitivity.tsv"

    inputs = [proposal_path, matched_path, ranking_path, queue_path, labels_path, paired_path, sensitivity_path]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing replay inputs:\n" + "\n".join(missing))

    proposal = [
        row for row in read_tsv(proposal_path)
        if row["official_split"] == "test" and row["acquisition_date"] == "ALL" and row["truth_role"] == "weed"
    ]
    proposal_rows = [{
        "variant": row["variant"],
        "truth_instances": row["truth_instances"],
        "spatial_recalled_instances": row["spatial_recalled_instances"],
        "spatial_proposal_recall": row["spatial_proposal_recall"],
        "role_qualified_recalled_instances": row["role_qualified_recalled_instances"],
        "role_qualified_proposal_recall": row["role_qualified_proposal_recall"],
    } for row in proposal]

    matched = [
        row for row in read_tsv(matched_path)
        if row["aggregation"] == "pooled_test" and row["k"] == "20"
    ]
    matched_rows = [{key: row[key] for key in (
        "variant", "accepted_candidates", "weed_candidates", "all_candidate_queue_precision",
        "weed_truth_instances", "spatial_recalled_weed_instances", "queued_spatial_weed_instance_recall",
        "role_qualified_recalled_weed_instances", "queued_role_qualified_weed_instance_recall",
        "one_to_one_role_qualified_matches", "one_to_one_role_qualified_recall",
        "crop_overlap_scene_numerator_0.00", "crop_overlap_scene_frequency_0.00",
    )} for row in matched]

    ranking_rows = [{key: row[key] for key in (
        "model", "candidates", "weed_candidates", "nonweed_candidates", "roc_auc", "average_precision"
    )} for row in read_tsv(ranking_path) if row["test_subset"] == "all_candidates"]

    masked_rows = [{key: row[key] for key in (
        "aggregation", "stratum", "images", "accepted_candidates", "weed_candidates",
        "all_candidate_queue_precision", "weed_truth_instances", "spatial_recalled_weed_instances",
        "queued_spatial_weed_instance_recall", "role_qualified_recalled_weed_instances",
        "queued_role_qualified_weed_instance_recall", "one_to_one_role_qualified_matches",
        "one_to_one_role_qualified_recall", "crop_overlap_scene_numerator_0.00",
        "crop_overlap_scene_frequency_0.00",
    )} for row in read_tsv(queue_path)
        if row["model"] == "all_candidate_component_masked_dinov2_linear" and row["k"] == "20"]

    label_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(labels_path):
        label_groups[row["strategy"]].append(row)
    label_rows = []
    for strategy in sorted(label_groups):
        rows = label_groups[strategy]
        label_rows.append({
            "strategy": strategy,
            "repeats": len(rows),
            "mean_test_auc": mean([float(row["test_all_candidate_roc_auc"]) for row in rows]),
            "mean_test_ap": mean([float(row["test_all_candidate_average_precision"]) for row in rows]),
            "mean_k20_precision": mean([float(row["all_candidate_queue_precision"]) for row in rows]),
            "mean_k20_qualified_recall": mean([float(row["queued_role_qualified_weed_instance_recall"]) for row in rows]),
        })

    paired = read_tsv(paired_path)
    paired_rows = []
    for column, metric in (
        ("paired_auc_difference_finetuned_minus_frozen", "AUC"),
        ("paired_k20_precision_difference", "K20_precision"),
        ("paired_k20_recall_difference", "K20_qualified_recall"),
    ):
        values = [float(row[column]) for row in paired]
        paired_rows.append({
            "metric": metric,
            "repeats": len(values),
            "mean_finetuned_minus_frozen": mean(values),
            "empirical_q025": quantile(values, 0.025),
            "empirical_q975": quantile(values, 0.975),
        })

    sensitivity = [
        row for row in read_tsv(sensitivity_path)
        if row["model"] == "all_candidate_component_masked_dinov2_linear"
        and row["k"] == "20" and row["stratum"] == "ALL"
    ]
    many = [float(row["many_to_one_queued_recall"]) for row in sensitivity]
    one = [float(row["one_to_one_queued_recall"]) for row in sensitivity]
    sensitivity_rows = [{
        "model": "all_candidate_component_masked_dinov2_linear",
        "k": 20,
        "contract_cells": len(sensitivity),
        "many_to_one_min": min(many),
        "many_to_one_max": max(many),
        "one_to_one_min": min(one),
        "one_to_one_max": max(one),
    }]

    outputs = {
        "official_test_proposal_summary.tsv": proposal_rows,
        "matched_weight_k20.tsv": matched_rows,
        "all_candidate_ranking.tsv": ranking_rows,
        "masked_dinov2_k20_by_date.tsv": masked_rows,
        "label_strategy_summary.tsv": label_rows,
        "finetune_paired_summary.tsv": paired_rows,
        "contract_sensitivity_range.tsv": sensitivity_rows,
    }
    output_hashes: dict[str, str] = {}
    for filename, rows in outputs.items():
        path = out / filename
        write_tsv(path, rows)
        output_hashes[filename] = sha256(path)

    summary = {
        "status": "replayed_from_archived_revision7_tables",
        "inputs": {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in inputs},
        "outputs": output_hashes,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
