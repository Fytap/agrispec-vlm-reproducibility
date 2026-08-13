#!/usr/bin/env python3
"""Crossed seed-by-image uncertainty for fixed-budget queue metrics.

Input files are named ``MODEL__SEED.tsv`` and contain one row per test image.
The crossed bootstrap resamples a common image set across models and resamples
training seeds independently within each model.  A second sensitivity analysis
resamples acquisition-date blocks rather than treating images as field-level
independent units.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


COUNT_METRICS = {
    "one_to_one_recall": ("one_to_one_role_qualified_recalled", "weed_instances"),
    "spatial_recall": ("many_to_one_spatial_recalled", "weed_instances"),
    "qualified_recall": ("many_to_one_role_qualified_recalled", "weed_instances"),
    "candidate_precision": ("eligible_weed_candidates", "accepted_candidates"),
}
MEAN_METRICS = {"crop_exposure": "crop_overlap_scene"}
PRIMARY_CONTRASTS = (("maskrcnn", "unet"), ("deeplabv3plus", "unet"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--replicates", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260813)
    return p.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return tuple(float(v) for v in np.quantile(values, [alpha / 2, 1 - alpha / 2]))


def aggregate(arrays: dict[str, np.ndarray], image_indices: np.ndarray, seed_indices: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric, (numerator, denominator) in COUNT_METRICS.items():
        num = arrays[numerator][seed_indices][:, image_indices].mean(axis=0).sum()
        den = arrays[denominator][seed_indices][:, image_indices].mean(axis=0).sum()
        result[metric] = float(num / max(1e-12, den))
    for metric, column in MEAN_METRICS.items():
        result[metric] = float(arrays[column][seed_indices][:, image_indices].mean())
    return result


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    discovered: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(args.input_dir.resolve().glob("*__*.tsv")):
        model, raw_seed = path.stem.rsplit("__", 1)
        discovered[model].append((int(raw_seed), path))
    required = {"unet", "deeplabv3plus", "maskrcnn"}
    if set(discovered) != required:
        raise ValueError(f"Expected {sorted(required)}, found {sorted(discovered)}")

    models: dict[str, dict[str, Any]] = {}
    reference_files: list[str] | None = None
    reference_dates: list[str] | None = None
    combined_rows: list[dict[str, Any]] = []
    columns = sorted({value for pair in COUNT_METRICS.values() for value in pair} | set(MEAN_METRICS.values()))
    for model, specifications in sorted(discovered.items()):
        runs: list[list[dict[str, str]]] = []
        seeds: list[int] = []
        for seed, path in sorted(specifications):
            rows = sorted(read_tsv(path), key=lambda row: row["file"])
            files = [row["file"] for row in rows]
            dates = [row["date"] for row in rows]
            if reference_files is None:
                reference_files, reference_dates = files, dates
            if files != reference_files or dates != reference_dates:
                raise ValueError(f"Image/date mismatch in {path}")
            runs.append(rows)
            seeds.append(seed)
            for row in rows:
                combined_rows.append({"model": model, "seed": seed, **row})
        arrays = {column: np.asarray([[float(row[column]) for row in run] for run in runs]) for column in columns}
        models[model] = {"seeds": seeds, "arrays": arrays}

    assert reference_files is not None and reference_dates is not None
    n_images = len(reference_files)
    if n_images != 26:
        raise ValueError(f"Expected 26 official-test images, found {n_images}")
    date_to_indices = {
        date: np.asarray([i for i, value in enumerate(reference_dates) if value == date], dtype=int)
        for date in sorted(set(reference_dates))
    }

    points: dict[str, dict[str, float]] = {}
    for model, content in models.items():
        points[model] = aggregate(
            content["arrays"], np.arange(n_images), np.arange(len(content["seeds"]))
        )

    rng = np.random.default_rng(args.seed)
    crossed = {model: {metric: np.empty(args.replicates) for metric in points[model]} for model in models}
    date_block = {model: {metric: np.empty(args.replicates) for metric in points[model]} for model in models}
    dates = np.asarray(sorted(date_to_indices))
    for replicate in range(args.replicates):
        image_draw = rng.integers(0, n_images, n_images)
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        block_draw = np.concatenate([date_to_indices[str(date)] for date in sampled_dates])
        for model, content in models.items():
            n_seeds = len(content["seeds"])
            seed_draw = rng.integers(0, n_seeds, n_seeds)
            for metric, value in aggregate(content["arrays"], image_draw, seed_draw).items():
                crossed[model][metric][replicate] = value
            for metric, value in aggregate(content["arrays"], block_draw, seed_draw).items():
                date_block[model][metric][replicate] = value

    estimate_rows: list[dict[str, Any]] = []
    for model, content in sorted(models.items()):
        for metric, point in points[model].items():
            lower, upper = percentile(crossed[model][metric])
            block_lower, block_upper = percentile(date_block[model][metric])
            seed_values = []
            for seed_index in range(len(content["seeds"])):
                seed_values.append(aggregate(content["arrays"], np.arange(n_images), np.asarray([seed_index]))[metric])
            estimate_rows.append({
                "model": model,
                "metric": metric,
                "point": point,
                "crossed_seed_image_lower_95": lower,
                "crossed_seed_image_upper_95": upper,
                "date_block_seed_lower_95": block_lower,
                "date_block_seed_upper_95": block_upper,
                "training_seeds": len(content["seeds"]),
                "seed_sample_sd": float(np.std(seed_values, ddof=1)),
            })

    contrast_rows: list[dict[str, Any]] = []
    for left, right in PRIMARY_CONTRASTS:
        for metric in points[left]:
            distribution = crossed[left][metric] - crossed[right][metric]
            block_distribution = date_block[left][metric] - date_block[right][metric]
            lower, upper = percentile(distribution)
            # Bonferroni family-wise interval for the two primary model contrasts.
            family_lower, family_upper = percentile(distribution, alpha=0.025)
            block_lower, block_upper = percentile(block_distribution)
            contrast_rows.append({
                "contrast": f"{left}_minus_{right}",
                "metric": metric,
                "point": points[left][metric] - points[right][metric],
                "crossed_seed_image_lower_95": lower,
                "crossed_seed_image_upper_95": upper,
                "bonferroni_family_lower_97_5": family_lower,
                "bonferroni_family_upper_97_5": family_upper,
                "date_block_seed_lower_95": block_lower,
                "date_block_seed_upper_95": block_upper,
                "bootstrap_probability_above_zero": float(np.mean(distribution > 0)),
            })

    write_tsv(output / "per_image_seed_metrics.tsv", combined_rows)
    write_tsv(output / "model_estimates.tsv", estimate_rows)
    write_tsv(output / "model_contrasts.tsv", contrast_rows)
    summary = {
        "status": "completed_crossed_seed_image_uncertainty",
        "analysis_class": "post_test_exploratory_uncertainty_analysis",
        "official_test_images": n_images,
        "acquisition_dates": {key: len(value) for key, value in date_to_indices.items()},
        "training_seeds": {model: content["seeds"] for model, content in models.items()},
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "image_estimand": "conditional_on_the_26_image_official_test_split",
        "date_block_sensitivity": "resample_four_acquisition_date_blocks_with_replacement",
        "multiplicity": "Bonferroni_familywise_intervals_for_two_primary_model_contrasts",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
