#!/usr/bin/env python3
"""Create anonymous extended agreement and human--mask correspondence tables."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


FIELDS = {
    "primary_content": ["background", "crop_dominant", "weed_dominant", "uncertain"],
    "weed_reviewable": ["no", "yes", "uncertain"],
    "crop_present": ["no", "yes", "uncertain"],
    "localization_usability": ["unusable", "partial", "adequate", "uncertain"],
    "confidence": ["1", "2", "3", "4", "5"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator-a", type=Path, required=True)
    p.add_argument("--operator-b", type=Path, required=True)
    p.add_argument("--stimulus-key", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260813)
    return p.parse_args()


def read_tsv(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def confusion(a: list[str], b: list[str], categories: list[str]) -> np.ndarray:
    index = {value: i for i, value in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=int)
    for left, right in zip(a, b, strict=True):
        matrix[index[left], index[right]] += 1
    return matrix


def kappa(matrix: np.ndarray, weight: str = "nominal") -> float:
    n = float(matrix.sum())
    if n == 0:
        return float("nan")
    q = matrix.shape[0]
    distance = np.abs(np.arange(q)[:, None] - np.arange(q)[None, :]) / max(1, q - 1)
    if weight == "nominal":
        weights = np.eye(q)
    elif weight == "linear":
        weights = 1.0 - distance
    elif weight == "quadratic":
        weights = 1.0 - distance**2
    else:
        raise ValueError(weight)
    observed = float((weights * matrix).sum() / n)
    expected_matrix = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / n
    expected = float((weights * expected_matrix).sum() / n)
    return float((observed - expected) / (1.0 - expected)) if expected < 1 else float("nan")


def gwet_ac1(matrix: np.ndarray) -> float:
    """Gwet AC1 for two raters and nominal categories."""
    n = float(matrix.sum())
    q = matrix.shape[0]
    if n == 0 or q < 2:
        return float("nan")
    observed = float(np.trace(matrix) / n)
    marginal = (matrix.sum(axis=0) + matrix.sum(axis=1)) / (2.0 * n)
    expected = float(np.sum(marginal * (1.0 - marginal)) / (q - 1))
    return float((observed - expected) / (1.0 - expected)) if expected < 1 else float("nan")


def interval(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return tuple(float(v) for v in np.quantile(finite, [0.025, 0.975]))


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    rows_a = {row["stimulus_id"]: row for row in read_tsv(args.operator_a, delimiter=",")}
    rows_b = {row["stimulus_id"]: row for row in read_tsv(args.operator_b, delimiter=",")}
    stimuli = {row["stimulus_id"]: row for row in read_tsv(args.stimulus_key)}
    ids = sorted(set(rows_a) & set(rows_b) & set(stimuli))
    if len(ids) != 518 or set(ids) != set(rows_a) or set(ids) != set(rows_b):
        raise ValueError("The two response files and stimulus key must contain the same 518 IDs")
    scene_to_ids: dict[str, list[str]] = defaultdict(list)
    for stimulus_id in ids:
        scene_to_ids[stimuli[stimulus_id]["scene_code"]].append(stimulus_id)

    rng = np.random.default_rng(args.seed)
    sampled_scene_ids: list[list[str]] = []
    scene_codes = np.asarray(sorted(scene_to_ids))
    for _ in range(args.replicates):
        draw = rng.choice(scene_codes, size=len(scene_codes), replace=True)
        sampled_scene_ids.append([stimulus_id for scene in draw for stimulus_id in scene_to_ids[str(scene)]])

    summary_rows: list[dict[str, Any]] = []
    contingency_rows: list[dict[str, Any]] = []
    for field, categories in FIELDS.items():
        values_a = [rows_a[stimulus_id][field] for stimulus_id in ids]
        values_b = [rows_b[stimulus_id][field] for stimulus_id in ids]
        matrix = confusion(values_a, values_b, categories)
        for i, category_a in enumerate(categories):
            for j, category_b in enumerate(categories):
                contingency_rows.append({
                    "field": field,
                    "operator_a_category": category_a,
                    "operator_b_category": category_b,
                    "count": int(matrix[i, j]),
                })

        exact_distribution: list[float] = []
        kappa_distribution: list[float] = []
        ac1_distribution: list[float] = []
        for sampled_ids in sampled_scene_ids:
            sampled_a = [rows_a[value][field] for value in sampled_ids]
            sampled_b = [rows_b[value][field] for value in sampled_ids]
            sampled_matrix = confusion(sampled_a, sampled_b, categories)
            exact_distribution.append(float(np.trace(sampled_matrix) / sampled_matrix.sum()))
            kappa_distribution.append(kappa(sampled_matrix))
            ac1_distribution.append(gwet_ac1(sampled_matrix))
        exact_lower, exact_upper = interval(exact_distribution)
        kappa_lower, kappa_upper = interval(kappa_distribution)
        ac1_lower, ac1_upper = interval(ac1_distribution)
        summary_rows.append({
            "field": field,
            "analysis": "all_categories_nominal",
            "n": int(matrix.sum()),
            "exact_agreement": float(np.trace(matrix) / matrix.sum()),
            "exact_lower_95_scene_bootstrap": exact_lower,
            "exact_upper_95_scene_bootstrap": exact_upper,
            "cohen_kappa": kappa(matrix),
            "kappa_lower_95_scene_bootstrap": kappa_lower,
            "kappa_upper_95_scene_bootstrap": kappa_upper,
            "gwet_ac1": gwet_ac1(matrix),
            "ac1_lower_95_scene_bootstrap": ac1_lower,
            "ac1_upper_95_scene_bootstrap": ac1_upper,
            "linear_weighted_kappa": "",
            "quadratic_weighted_kappa": "",
        })

    ordered_specs = {
        "localization_usability_determinate": ("localization_usability", ["unusable", "partial", "adequate"]),
        "confidence": ("confidence", ["1", "2", "3", "4", "5"]),
    }
    for name, (field, categories) in ordered_specs.items():
        chosen = [
            stimulus_id for stimulus_id in ids
            if rows_a[stimulus_id][field] in categories and rows_b[stimulus_id][field] in categories
        ]
        matrix = confusion(
            [rows_a[value][field] for value in chosen],
            [rows_b[value][field] for value in chosen],
            categories,
        )
        summary_rows.append({
            "field": name,
            "analysis": "ordered_determinate_pairs" if field == "localization_usability" else "ordered_all_pairs",
            "n": int(matrix.sum()),
            "exact_agreement": float(np.trace(matrix) / matrix.sum()),
            "exact_lower_95_scene_bootstrap": "",
            "exact_upper_95_scene_bootstrap": "",
            "cohen_kappa": kappa(matrix),
            "kappa_lower_95_scene_bootstrap": "",
            "kappa_upper_95_scene_bootstrap": "",
            "gwet_ac1": gwet_ac1(matrix),
            "ac1_lower_95_scene_bootstrap": "",
            "ac1_upper_95_scene_bootstrap": "",
            "linear_weighted_kappa": kappa(matrix, "linear"),
            "quadratic_weighted_kappa": kappa(matrix, "quadratic"),
        })

    confidence_rows: list[dict[str, Any]] = []
    for operator, source in (("OP-A", rows_a), ("OP-B", rows_b)):
        values = np.asarray([int(source[stimulus_id]["confidence"]) for stimulus_id in ids])
        confidence_rows.append({
            "operator": operator,
            "n": len(values),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        })
    confidence_a = np.asarray([int(rows_a[value]["confidence"]) for value in ids])
    confidence_b = np.asarray([int(rows_b[value]["confidence"]) for value in ids])
    confidence_rows.append({
        "operator": "paired_absolute_difference",
        "n": len(ids),
        "mean": float(np.abs(confidence_a - confidence_b).mean()),
        "median": float(np.median(np.abs(confidence_a - confidence_b))),
        "q25": float(np.quantile(np.abs(confidence_a - confidence_b), 0.25)),
        "q75": float(np.quantile(np.abs(confidence_a - confidence_b), 0.75)),
    })

    human_mask = Counter()
    for stimulus_id in ids:
        left, right = rows_a[stimulus_id]["weed_reviewable"], rows_b[stimulus_id]["weed_reviewable"]
        if left == right == "yes":
            human = "consensus_yes"
        elif left == right == "no":
            human = "consensus_no"
        else:
            human = "disputed_or_uncertain"
        mask = "mask_positive" if stimuli[stimulus_id]["candidate_truth"] == "weed" else "mask_negative"
        human_mask[(human, mask)] += 1
    human_mask_rows = [
        {"human_status": human, "mask_status": mask, "count": human_mask[(human, mask)]}
        for human in ("consensus_yes", "consensus_no", "disputed_or_uncertain")
        for mask in ("mask_positive", "mask_negative")
    ]
    # Binary human--mask correspondence is defined only where the two operators
    # reached a determinate consensus.  Disputed or uncertain items remain in
    # the complete three-by-two contingency table and are reported as excluded;
    # they are not silently counted as human-negative observations.
    true_positive = human_mask[("consensus_yes", "mask_positive")]
    false_positive = human_mask[("consensus_no", "mask_positive")]
    false_negative = human_mask[("consensus_yes", "mask_negative")]
    true_negative = human_mask[("consensus_no", "mask_negative")]
    eligible_items = true_positive + false_positive + false_negative + true_negative
    unresolved_items = sum(
        human_mask[("disputed_or_uncertain", mask)]
        for mask in ("mask_positive", "mask_negative")
    )
    human_positive = true_positive + false_negative
    mask_positive = true_positive + false_positive
    human_mask_metrics = [{
        "analysis_population": "determinate_operator_consensus_only",
        "eligible_items": eligible_items,
        "eligible_fraction_of_all_items": eligible_items / len(ids),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "intersection": true_positive,
        "human_consensus_positive": human_positive,
        "mask_positive": mask_positive,
        "mask_precision_against_human_consensus": true_positive / mask_positive,
        "mask_recall_against_human_consensus": true_positive / human_positive,
        "f1": 2 * true_positive / (2 * true_positive + false_positive + false_negative),
        "jaccard": true_positive / (true_positive + false_positive + false_negative),
        "accuracy": (true_positive + true_negative) / eligible_items,
        "unresolved_human_items": unresolved_items,
        "unresolved_items_excluded_from_binary_metrics": True,
    }]

    write_tsv(output / "operator_agreement_extended.tsv", summary_rows)
    write_tsv(output / "operator_contingency_long.tsv", contingency_rows)
    write_tsv(output / "confidence_summary.tsv", confidence_rows)
    write_tsv(output / "human_mask_contingency.tsv", human_mask_rows)
    write_tsv(output / "human_mask_metrics.tsv", human_mask_metrics)
    summary = {
        "status": "completed_anonymous_operator_agreement_extension",
        "items": len(ids),
        "scenes": len(scene_codes),
        "timing_used": False,
        "operator_population_inference": False,
        "raw_item_level_responses_public": False,
        "ordered_localization_rule": "exclude_pairs_with_uncertain_before_weighted_kappa",
        "human_mask_binary_rule": "compute_binary_metrics_on_determinate_consensus_yes_or_no_only",
        "human_mask_unresolved_rule": "retain_in_three_by_two_contingency_and_exclude_from_binary_metrics",
        "bootstrap": {"unit": "scene", "replicates": args.replicates, "seed": args.seed},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
