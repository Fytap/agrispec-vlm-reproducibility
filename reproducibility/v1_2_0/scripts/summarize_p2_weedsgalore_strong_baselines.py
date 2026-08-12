#!/usr/bin/env python3
"""Aggregate strong-baseline seeds with image-cluster uncertainty."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "all_candidate_precision",
    "many_to_one_spatial_recall",
    "many_to_one_role_qualified_recall",
    "one_to_one_role_qualified_recall",
    "crop_overlap_scene_frequency",
    "mean_candidates_per_image",
)
RUNS = {
    "ResNet18-UNet": [
        f"P2_WEEDSGALORE_TARGET_SEMANTIC_H200_SEED_{seed}_20260812_v1"
        for seed in (20260810, 20260811, 20260812, 20260813, 20260814)
    ],
    "DeepLabV3Plus-ResNet50": [
        "P2_WEEDSGALORE_DEEPLABV3PLUS_H200_SEED_20260820_20260812_v2",
        "P2_WEEDSGALORE_DEEPLABV3PLUS_H200_SEED_20260821_20260812_v1",
        "P2_WEEDSGALORE_DEEPLABV3PLUS_H200_SEED_20260822_20260812_v1",
    ],
    "Mask-RCNN-ResNet50-FPN-v2": [
        f"P2_WEEDSGALORE_MASKRCNN_H200_SEED_{seed}_20260812_v1"
        for seed in (20260820, 20260821, 20260822)
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, str]], indices: np.ndarray) -> dict[str, float]:
    selected = [rows[int(index)] for index in indices]
    weeds = sum(int(row["weed_instances"]) for row in selected)
    accepted = sum(int(row["accepted_candidates"]) for row in selected)
    eligible = sum(int(row["eligible_weed_candidates"]) for row in selected)
    return {
        "all_candidate_precision": eligible / max(1, accepted),
        "many_to_one_spatial_recall": sum(int(row["many_to_one_spatial_recalled"]) for row in selected) / max(1, weeds),
        "many_to_one_role_qualified_recall": sum(int(row["many_to_one_role_qualified_recalled"]) for row in selected) / max(1, weeds),
        "one_to_one_role_qualified_recall": sum(int(row["one_to_one_role_qualified_recalled"]) for row in selected) / max(1, weeds),
        "crop_overlap_scene_frequency": sum(int(row["crop_overlap_scene"]) for row in selected) / max(1, len(selected)),
        "mean_candidates_per_image": accepted / max(1, len(selected)),
    }


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values[np.isfinite(values)], [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    root = args.results_root.resolve()
    config = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    run_rows: list[dict[str, Any]] = []
    tiles: dict[str, dict[str, list[dict[str, str]]]] = {}
    input_rows: list[dict[str, Any]] = []
    files: list[str] | None = None
    for model, run_names in RUNS.items():
        tiles[model] = {}
        for run_name in run_names:
            run = root / run_name
            summary_path = run / "summary.json"
            tile_path = run / "official_test_tile_metrics.tsv"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not str(summary["status"]).startswith("completed_post_test_"):
                raise ValueError(f"Incomplete run {run_name}")
            rows = sorted(read_tsv(tile_path), key=lambda row: row["file"])
            current_files = [row["file"] for row in rows]
            if len(current_files) != 26:
                raise ValueError(f"Expected 26 image clusters in {run_name}")
            if files is None:
                files = current_files
            elif files != current_files:
                raise ValueError(f"Image order/set mismatch in {run_name}")
            indices = np.arange(len(rows), dtype=np.int64)
            point = aggregate(rows, indices)
            archived = summary["official_test_queue_metrics"]
            aliases = {
                "all_candidate_precision": "K20_all_candidate_precision",
                "many_to_one_spatial_recall": "K20_many_to_one_spatial_recall",
                "many_to_one_role_qualified_recall": "K20_many_to_one_role_qualified_recall",
                "one_to_one_role_qualified_recall": "K20_one_to_one_role_qualified_recall",
                "crop_overlap_scene_frequency": "K20_crop_overlap_scene_frequency",
            }
            for metric, archived_key in aliases.items():
                if abs(point[metric] - float(archived[archived_key])) > 1e-12:
                    raise ValueError(f"Recomputed metric mismatch for {run_name}/{metric}")
            seed = int(summary["training"]["seed"])
            run_rows.append({"model": model, "run_id": run_name, "seed": seed, **point})
            tiles[model][run_name] = rows
            input_rows.append({
                "model": model,
                "run_id": run_name,
                "seed": seed,
                "summary_sha256": sha256(summary_path),
                "tile_metrics_sha256": sha256(tile_path),
                "checkpoint_sha256": summary["model"]["checkpoint_sha256"],
            })
    if files is None:
        raise RuntimeError("No runs")

    model_rows: list[dict[str, Any]] = []
    for model in RUNS:
        rows = [row for row in run_rows if row["model"] == model]
        record: dict[str, Any] = {"model": model, "training_seeds": len(rows)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in rows])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sample_sd"] = float(values.std(ddof=1))
        model_rows.append(record)

    rng = np.random.default_rng(20260812)
    replicates = 10000
    model_bootstrap = {model: {metric: np.empty(replicates) for metric in METRICS} for model in RUNS}
    for replicate in range(replicates):
        sampled = rng.integers(0, len(files), size=len(files))
        for model, run_names in RUNS.items():
            estimates = [aggregate(tiles[model][run_name], sampled) for run_name in run_names]
            for metric in METRICS:
                model_bootstrap[model][metric][replicate] = float(np.mean([row[metric] for row in estimates]))
    uncertainty_rows: list[dict[str, Any]] = []
    for model in RUNS:
        point = next(row for row in model_rows if row["model"] == model)
        for metric in METRICS:
            low, high = interval(model_bootstrap[model][metric])
            uncertainty_rows.append({
                "model": model,
                "metric": metric,
                "estimate": point[f"{metric}_mean"],
                "ci95_low": low,
                "ci95_high": high,
                "cluster_unit": "image",
                "clusters": len(files),
                "bootstrap_replicates": replicates,
            })
    contrast_rows: list[dict[str, Any]] = []
    for comparator in ("DeepLabV3Plus-ResNet50", "Mask-RCNN-ResNet50-FPN-v2"):
        for metric in METRICS:
            differences = model_bootstrap[comparator][metric] - model_bootstrap["ResNet18-UNet"][metric]
            low, high = interval(differences)
            left = next(row for row in model_rows if row["model"] == comparator)[f"{metric}_mean"]
            right = next(row for row in model_rows if row["model"] == "ResNet18-UNet")[f"{metric}_mean"]
            contrast_rows.append({
                "contrast": f"{comparator}_minus_ResNet18-UNet",
                "metric": metric,
                "estimate": left - right,
                "ci95_low": low,
                "ci95_high": high,
                "cluster_unit": "image",
                "clusters": len(files),
                "bootstrap_replicates": replicates,
            })

    output.mkdir(parents=True, exist_ok=False)
    write_tsv(output / "input_run_audit.tsv", input_rows)
    write_tsv(output / "per_run_metrics.tsv", run_rows)
    write_tsv(output / "model_mean_sample_sd.tsv", model_rows)
    write_tsv(output / "model_image_cluster_bootstrap.tsv", uncertainty_rows)
    write_tsv(output / "paired_model_contrasts_image_cluster_bootstrap.tsv", contrast_rows)
    primary = [row for row in contrast_rows if row["metric"] == "one_to_one_role_qualified_recall"]
    summary = {
        "run_id": args.run_id,
        "status": "completed_strong_baseline_aggregate",
        "models": list(RUNS),
        "image_clusters": len(files),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": 20260812,
        "primary_contrasts": primary,
        "config_sha256": sha256(config),
        "script_sha256": sha256(Path(__file__).resolve()),
        "governance": {
            "analysis_class": "post_test_strong_baseline_extension",
            "same_fixed_queue_capacity": 20,
            "test_labels_for_training_or_selection": False,
            "training_seed_sd_not_field_uncertainty": True,
        },
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
