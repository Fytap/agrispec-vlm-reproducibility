#!/usr/bin/env python3
"""Paired image-cluster bootstrap for the matched-burden K=20 audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ORIGINAL = "original_T095_A032"
COMMISSIONED = "commissioned_T070_A032"
VARIANTS = (ORIGINAL, COMMISSIONED)
METRICS = (
    "all_candidate_precision",
    "many_to_one_spatial_recall",
    "many_to_one_role_qualified_recall",
    "one_to_one_role_qualified_recall",
    "crop_overlap_scene_frequency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, str]], indices: np.ndarray) -> dict[str, float]:
    accepted = sum(int(rows[index]["accepted_candidates"]) for index in indices)
    eligible = sum(int(rows[index]["eligible_weed_candidates"]) for index in indices)
    weeds = sum(int(rows[index]["weed_instances"]) for index in indices)
    spatial = sum(int(rows[index]["spatial_recalled"]) for index in indices)
    qualified = sum(int(rows[index]["qualified_recalled"]) for index in indices)
    one_to_one = sum(int(rows[index]["one_to_one_recalled"]) for index in indices)
    crop_scenes = sum(int(rows[index]["crop_overlap_scene"]) for index in indices)
    return {
        "all_candidate_precision": eligible / accepted if accepted else float("nan"),
        "many_to_one_spatial_recall": spatial / weeds if weeds else float("nan"),
        "many_to_one_role_qualified_recall": qualified / weeds if weeds else float("nan"),
        "one_to_one_role_qualified_recall": one_to_one / weeds if weeds else float("nan"),
        "crop_overlap_scene_frequency": crop_scenes / len(indices) if len(indices) else float("nan"),
    }


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def tile_values(rows: list[dict[str, str]], metric: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        if metric == "all_candidate_precision":
            denominator = int(row["accepted_candidates"])
            numerator = int(row["eligible_weed_candidates"])
        elif metric == "crop_overlap_scene_frequency":
            denominator = 1
            numerator = int(row["crop_overlap_scene"])
        else:
            denominator = int(row["weed_instances"])
            numerator_key = {
                "many_to_one_spatial_recall": "spatial_recalled",
                "many_to_one_role_qualified_recall": "qualified_recalled",
                "one_to_one_role_qualified_recall": "one_to_one_recalled",
            }[metric]
            numerator = int(row[numerator_key])
        if denominator:
            values.append(numerator / denominator)
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    args = parse_args()
    input_path = args.input_tsv.resolve()
    config_path = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source = read_rows(input_path)
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in source:
        variant = row["variant"]
        if variant not in VARIANTS:
            raise ValueError(f"Unexpected variant {variant}")
        file_id = row["file"]
        if file_id in grouped[variant]:
            raise ValueError(f"Duplicate {variant}/{file_id}")
        grouped[variant][file_id] = row
    files = sorted(set(grouped[ORIGINAL]) & set(grouped[COMMISSIONED]))
    if not files or set(files) != set(grouped[ORIGINAL]) or set(files) != set(grouped[COMMISSIONED]):
        raise ValueError("Paired file contract failed")
    rows_by_variant = {variant: [grouped[variant][file_id] for file_id in files] for variant in VARIANTS}
    for index, file_id in enumerate(files):
        left = rows_by_variant[ORIGINAL][index]
        right = rows_by_variant[COMMISSIONED][index]
        if left["weed_instances"] != right["weed_instances"] or left["common_k"] != right["common_k"]:
            raise ValueError(f"Pair denominator/burden mismatch for {file_id}")

    rng = np.random.default_rng(20260812)
    replicates = 10000
    point_indices = np.arange(len(files), dtype=np.int64)
    point = {variant: aggregate(rows_by_variant[variant], point_indices) for variant in VARIANTS}
    bootstrap = {variant: {metric: np.empty(replicates) for metric in METRICS} for variant in VARIANTS}
    for replicate in range(replicates):
        sampled = rng.integers(0, len(files), size=len(files))
        for variant in VARIANTS:
            estimate = aggregate(rows_by_variant[variant], sampled)
            for metric in METRICS:
                bootstrap[variant][metric][replicate] = estimate[metric]

    estimate_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        for variant in VARIANTS:
            low, high = interval(bootstrap[variant][metric])
            estimate_rows.append({
                "variant": variant,
                "metric": metric,
                "estimate": point[variant][metric],
                "ci95_low": low,
                "ci95_high": high,
                "cluster_unit": "image",
                "clusters": len(files),
                "bootstrap_replicates": replicates,
            })
            values = tile_values(rows_by_variant[variant], metric)
            distribution_rows.append({
                "variant": variant,
                "metric": metric,
                "tiles_with_defined_metric": len(values),
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            })
        differences = bootstrap[COMMISSIONED][metric] - bootstrap[ORIGINAL][metric]
        low, high = interval(differences)
        contrast_rows.append({
            "contrast": "commissioned_minus_original",
            "metric": metric,
            "estimate": point[COMMISSIONED][metric] - point[ORIGINAL][metric],
            "ci95_low": low,
            "ci95_high": high,
            "cluster_unit": "image",
            "clusters": len(files),
            "bootstrap_replicates": replicates,
        })

    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "variant_metric_cluster_bootstrap.tsv", estimate_rows)
    write_rows(output / "paired_contrast_cluster_bootstrap.tsv", contrast_rows)
    write_rows(output / "per_tile_distribution_summary.tsv", distribution_rows)
    summary = {
        "run_id": args.run_id,
        "status": "completed",
        "paired_clusters": len(files),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": 20260812,
        "primary_endpoint": next(row for row in contrast_rows if row["metric"] == "one_to_one_role_qualified_recall"),
        "input_sha256": sha256(input_path),
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "SHA256SUMS.tsv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("path\tsha256\n")
        for path in sorted(output.iterdir()):
            if path.name != "SHA256SUMS.tsv":
                stream.write(f"{path.name}\t{sha256(path)}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
