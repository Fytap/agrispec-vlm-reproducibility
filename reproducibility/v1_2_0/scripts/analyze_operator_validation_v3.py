#!/usr/bin/env python3
"""Run aggregate two-operator agreement analysis without timing outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    categories = sorted(set(a) | set(b))
    observed = sum(left == right for left, right in zip(a, b, strict=True)) / len(a)
    expected = sum((a.count(cat) / len(a)) * (b.count(cat) / len(b)) for cat in categories)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def agreement(a: list[str], b: list[str]) -> float:
    return sum(left == right for left, right in zip(a, b, strict=True)) / len(a)


def finite_percentiles(values: list[float]) -> list[float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    return [float(value) for value in np.percentile(finite, [2.5, 50, 97.5])] if finite.size else [None, None, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-a", type=Path, required=True)
    parser.add_argument("--operator-b", type=Path, required=True)
    parser.add_argument("--stimulus-key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")

    rows_a, rows_b, key_rows = read_csv(args.operator_a), read_csv(args.operator_b), read_tsv(args.stimulus_key)
    by_a = {row["stimulus_id"]: row for row in rows_a}
    by_b = {row["stimulus_id"]: row for row in rows_b}
    key = {row["stimulus_id"]: row for row in key_rows}
    if len(by_a) != 518 or set(by_a) != set(by_b) or set(by_a) != set(key):
        raise ValueError({"operator_a": len(by_a), "operator_b": len(by_b), "key": len(key)})

    ids = sorted(by_a)
    fields = ("weed_reviewable", "primary_content", "crop_present", "localization_usability")
    observed: dict[str, dict[str, object]] = {}
    for field in fields:
        left, right = [by_a[i][field] for i in ids], [by_b[i][field] for i in ids]
        pair_counts = Counter(zip(left, right, strict=True))
        observed[field] = {
            "agreement": agreement(left, right),
            "cohen_kappa": kappa(left, right),
            "paired_category_counts": {
                f"{left_value}|{right_value}": count
                for (left_value, right_value), count in sorted(pair_counts.items())
            },
        }

    both_yes = sum(by_a[i]["weed_reviewable"] == by_b[i]["weed_reviewable"] == "yes" for i in ids)
    both_no = sum(by_a[i]["weed_reviewable"] == by_b[i]["weed_reviewable"] == "no" for i in ids)
    truth_positive = {i for i in ids if key[i]["candidate_truth"] == "weed"}
    consensus_positive = {i for i in ids if by_a[i]["weed_reviewable"] == by_b[i]["weed_reviewable"] == "yes"}
    clusters: dict[str, list[str]] = defaultdict(list)
    for item in ids:
        clusters[key[item]["scene_code"]].append(item)
    cluster_names = sorted(clusters)

    rng = np.random.default_rng(args.seed)
    bootstrap = {
        "weed_reviewable_kappa": [],
        "weed_reviewable_agreement": [],
        "consensus_yes_fraction": [],
    }
    for _ in range(args.bootstrap_repeats):
        sampled_clusters = rng.choice(cluster_names, size=len(cluster_names), replace=True)
        sampled_ids = [item for cluster in sampled_clusters for item in clusters[str(cluster)]]
        left = [by_a[i]["weed_reviewable"] for i in sampled_ids]
        right = [by_b[i]["weed_reviewable"] for i in sampled_ids]
        bootstrap["weed_reviewable_kappa"].append(kappa(left, right))
        bootstrap["weed_reviewable_agreement"].append(agreement(left, right))
        bootstrap["consensus_yes_fraction"].append(
            sum(l == r == "yes" for l, r in zip(left, right, strict=True)) / len(sampled_ids)
        )

    summary = {
        "status": "completed_registered_two_operator_analysis",
        "analysis_revision": "v3_no_timing_outcome",
        "items": len(ids),
        "scenes": len(cluster_names),
        "analysis_excluded_fields": ["elapsed_ms", "submitted_at_utc"],
        "agreement": observed,
        "weed_reviewability": {
            "both_yes": both_yes,
            "both_no": both_no,
            "disputed_or_uncertain": len(ids) - both_yes - both_no,
            "consensus_yes_fraction": both_yes / len(ids),
        },
        "mask_contract_correspondence": {
            "mask_qualified_weed_candidates": len(truth_positive),
            "consensus_positive_candidates": len(consensus_positive),
            "consensus_positive_and_mask_positive": len(consensus_positive & truth_positive),
            "consensus_positive_mask_positive_fraction": len(consensus_positive & truth_positive) / max(1, len(consensus_positive)),
        },
        "scene_cluster_bootstrap_95_percentile_intervals": {
            name: finite_percentiles(values) for name, values in bootstrap.items()
        },
        "inference_boundary": "two fixed pilot operators; no operator-population inference",
        "privacy_boundary": "aggregate outputs only; no item-level response table exported",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
