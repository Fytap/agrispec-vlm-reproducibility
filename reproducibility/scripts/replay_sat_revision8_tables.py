#!/usr/bin/env python3
"""Replay revision-8 evidence summaries from archived, non-image outputs.

The script uses only the Python standard library. It first replays the
revision-7 core tables, then recomputes the revision-8 queue-policy ratios and
extracts the registered target-semantic comparison from its completed run.
It never reads imagery, masks, embeddings, checkpoints, or test labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def ratio(numerator: str, denominator: str) -> float:
    return int(numerator) / max(1, int(denominator))


def assert_close(observed: str, recomputed: float, label: str) -> None:
    if abs(float(observed) - recomputed) > 1e-12:
        raise ValueError(f"Archived ratio mismatch for {label}: {observed} != {recomputed}")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> None:
    args = arguments()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    replay7 = root / "scripts" / "replay_sat_revision7_tables.py"
    if not replay7.is_file():
        raise FileNotFoundError(replay7)
    core_out = out / "revision7_core"
    subprocess.run(
        [sys.executable, str(replay7), "--project-root", str(root), "--output-dir", str(core_out)],
        check=True,
    )

    base = root / "results" / "p2_development"
    queue_dir = base / "P2_WEEDSGALORE_REVISION8_QUEUE_POLICY_TILE_AUDIT_20260810_v1"
    semantic_dir = base / "P2_WEEDSGALORE_TARGET_SEMANTIC_QUEUE_BASELINE_20260810_v3"
    queue_path = queue_dir / "official_test_queue_policy_comparison.tsv"
    crop_path = queue_dir / "official_test_crop_exposure_severity.tsv"
    tile_path = queue_dir / "official_test_queue_policy_per_tile.tsv"
    semantic_summary_path = semantic_dir / "summary.json"
    semantic_dates_path = semantic_dir / "official_test_date_metrics.tsv"
    semantic_tiles_path = semantic_dir / "official_test_tile_metrics.tsv"
    inputs = [
        queue_path,
        crop_path,
        tile_path,
        semantic_summary_path,
        semantic_dates_path,
        semantic_tiles_path,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing revision-8 replay inputs:\n" + "\n".join(missing))

    policy_rows: list[dict[str, Any]] = []
    for row in read_tsv(queue_path):
        if row["aggregation"] != "pooled_test" or row["stratum"] != "ALL":
            continue
        precision = ratio(row["weed_candidates"], row["accepted_candidates"])
        many = ratio(row["role_qualified_recalled_weed_instances"], row["weed_truth_instances"])
        one = ratio(row["one_to_one_role_qualified_matches"], row["weed_truth_instances"])
        crop = ratio(row["crop_overlap_scene_numerator_0.00"], row["images"])
        assert_close(row["all_candidate_queue_precision"], precision, row["policy"] + ":precision")
        assert_close(row["queued_role_qualified_weed_instance_recall"], many, row["policy"] + ":many")
        assert_close(row["one_to_one_role_qualified_recall"], one, row["policy"] + ":one")
        assert_close(row["crop_overlap_scene_frequency_0.00"], crop, row["policy"] + ":crop")
        policy_rows.append({
            "policy": row["policy"],
            "accepted_candidates": row["accepted_candidates"],
            "weed_candidates": row["weed_candidates"],
            "precision_recomputed": precision,
            "weed_truth_instances": row["weed_truth_instances"],
            "many_to_one_qualified_recalled": row["role_qualified_recalled_weed_instances"],
            "many_to_one_qualified_recall_recomputed": many,
            "one_to_one_qualified_matches": row["one_to_one_role_qualified_matches"],
            "one_to_one_qualified_recall_recomputed": one,
            "crop_overlap_scenes": row["crop_overlap_scene_numerator_0.00"],
            "crop_overlap_scene_frequency_recomputed": crop,
        })
    write_tsv(out / "revision8_queue_policy_summary.tsv", policy_rows)

    crop_rows: list[dict[str, Any]] = []
    for row in read_tsv(crop_path):
        recomputed = ratio(row["crop_instances_covered_at_0.50"], row["crop_truth_instances"])
        assert_close(row["crop_instance_coverage_rate_at_0.50"], recomputed, row["policy"] + ":crop50")
        crop_rows.append({
            "policy": row["policy"],
            "accepted_candidates": row["accepted_candidates"],
            "crop_truth_instances": row["crop_truth_instances"],
            "crop_instances_covered_at_0.50": row["crop_instances_covered_at_0.50"],
            "crop_instance_coverage_rate_at_0.50_recomputed": recomputed,
            "crop_fraction_q95": row["crop_fraction_q95"],
            "crop_fraction_maximum": row["crop_fraction_maximum"],
        })
    write_tsv(out / "revision8_crop_exposure_summary.tsv", crop_rows)

    per_tile: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_tsv(tile_path):
        per_tile[(row["policy"], row["file"])] = row
    tile_rows: list[dict[str, Any]] = []
    fixed_files = sorted(file for policy, file in per_tile if policy == "fixed_K20_per_image")
    for metric in ("queued_role_qualified_weed_instance_recall", "one_to_one_role_qualified_recall"):
        deltas = [
            float(per_tile[("global_top_N520", file)][metric])
            - float(per_tile[("fixed_K20_per_image", file)][metric])
            for file in fixed_files
        ]
        tile_rows.append({
            "comparator_minus_fixed": "global_top_N520",
            "metric": metric,
            "tiles": len(deltas),
            "median_delta": quantile(deltas, 0.50),
            "q25_delta": quantile(deltas, 0.25),
            "q75_delta": quantile(deltas, 0.75),
            "minimum_delta": min(deltas),
            "maximum_delta": max(deltas),
            "improved_tiles": sum(value > 1e-15 for value in deltas),
            "unchanged_tiles": sum(abs(value) <= 1e-15 for value in deltas),
            "worsened_tiles": sum(value < -1e-15 for value in deltas),
        })
    write_tsv(out / "revision8_paired_tile_delta_summary.tsv", tile_rows)

    semantic = json.loads(semantic_summary_path.read_text(encoding="utf-8"))
    if not str(semantic.get("status", "")).startswith("completed"):
        raise ValueError("Target-semantic run is not completed")
    selected = semantic["selected_on_validation"]
    segmentation = semantic["official_test_semantic_metrics"]
    queue = semantic["official_test_queue_metrics"]
    runtime = semantic["runtime"]
    semantic_tiles = read_tsv(semantic_tiles_path)
    semantic_totals = {
        field: sum(int(row[field]) for row in semantic_tiles)
        for field in (
            "weed_instances",
            "candidates",
            "accepted_candidates",
            "eligible_weed_candidates",
            "many_to_one_spatial_recalled",
            "many_to_one_role_qualified_recalled",
            "one_to_one_role_qualified_recalled",
            "crop_overlap_scene",
        )
    }
    semantic_recomputed = {
        "candidates_per_image": semantic_totals["candidates"] / len(semantic_tiles),
        "K20_all_candidate_precision": semantic_totals["eligible_weed_candidates"]
        / max(1, semantic_totals["accepted_candidates"]),
        "K20_many_to_one_spatial_recall": semantic_totals["many_to_one_spatial_recalled"]
        / max(1, semantic_totals["weed_instances"]),
        "K20_many_to_one_role_qualified_recall": semantic_totals["many_to_one_role_qualified_recalled"]
        / max(1, semantic_totals["weed_instances"]),
        "K20_one_to_one_role_qualified_recall": semantic_totals["one_to_one_role_qualified_recalled"]
        / max(1, semantic_totals["weed_instances"]),
        "K20_crop_overlap_scene_frequency": semantic_totals["crop_overlap_scene"] / len(semantic_tiles),
    }
    for field, value in semantic_recomputed.items():
        assert_close(str(queue[field]), value, "target_semantic:" + field)
    for field in ("candidates", "accepted_candidates", "eligible_weed_candidates", "weed_instances"):
        if int(queue[field]) != semantic_totals[field]:
            raise ValueError(f"Target-semantic pooled count mismatch for {field}")
    semantic_row = {
        "selected_epoch": selected["epoch"],
        "selected_threshold": selected["threshold"],
        "selected_minimum_area_pixels": selected["minimum_area_pixels"],
        "test_background_iou": segmentation["background_iou"],
        "test_crop_iou": segmentation["crop_iou"],
        "test_weed_iou": segmentation["weed_iou"],
        "test_mean_iou": segmentation["mean_iou"],
        "test_plant_role_mean_iou": segmentation["plant_role_mean_iou"],
        "test_candidates": semantic_totals["candidates"],
        "test_candidates_per_image_recomputed": semantic_recomputed["candidates_per_image"],
        "K20_eligible_weed_candidates": semantic_totals["eligible_weed_candidates"],
        "K20_accepted_candidates": semantic_totals["accepted_candidates"],
        "K20_precision_recomputed": semantic_recomputed["K20_all_candidate_precision"],
        "K20_many_to_one_spatial_recalled": semantic_totals["many_to_one_spatial_recalled"],
        "K20_many_to_one_spatial_recall_recomputed": semantic_recomputed["K20_many_to_one_spatial_recall"],
        "K20_many_to_one_qualified_recalled": semantic_totals["many_to_one_role_qualified_recalled"],
        "K20_many_to_one_qualified_recall_recomputed": semantic_recomputed["K20_many_to_one_role_qualified_recall"],
        "K20_one_to_one_qualified_recalled": semantic_totals["one_to_one_role_qualified_recalled"],
        "K20_one_to_one_qualified_recall_recomputed": semantic_recomputed["K20_one_to_one_role_qualified_recall"],
        "K20_AR_IoU_0.25": queue["K20_AR_IoU_0.25"],
        "K20_AR_IoU_0.50": queue["K20_AR_IoU_0.50"],
        "K20_AR_IoU_0.75": queue["K20_AR_IoU_0.75"],
        "K20_crop_overlap_scenes": semantic_totals["crop_overlap_scene"],
        "K20_crop_overlap_scene_frequency_recomputed": semantic_recomputed["K20_crop_overlap_scene_frequency"],
        "test_seconds_per_tile": runtime["test_seconds_per_tile"],
        "test_peak_gpu_memory_bytes": runtime["test_peak_gpu_memory_bytes"],
    }
    write_tsv(out / "revision8_target_semantic_summary.tsv", [semantic_row])

    date_rows = read_tsv(semantic_dates_path)
    semantic_by_date: dict[str, list[dict[str, str]]] = {}
    for row in semantic_tiles:
        semantic_by_date.setdefault(row["date"], []).append(row)
    for row in date_rows:
        tiles_for_date = semantic_by_date[row["date"]]
        weed = sum(int(tile["weed_instances"]) for tile in tiles_for_date)
        accepted = sum(int(tile["accepted_candidates"]) for tile in tiles_for_date)
        eligible = sum(int(tile["eligible_weed_candidates"]) for tile in tiles_for_date)
        spatial = sum(int(tile["many_to_one_spatial_recalled"]) for tile in tiles_for_date)
        qualified = sum(int(tile["many_to_one_role_qualified_recalled"]) for tile in tiles_for_date)
        one_to_one = sum(int(tile["one_to_one_role_qualified_recalled"]) for tile in tiles_for_date)
        crop_scenes = sum(int(tile["crop_overlap_scene"]) for tile in tiles_for_date)
        assert_close(row["precision"], eligible / max(1, accepted), row["date"] + ":precision")
        assert_close(row["many_to_one_spatial_recall"], spatial / max(1, weed), row["date"] + ":spatial")
        assert_close(row["many_to_one_role_qualified_recall"], qualified / max(1, weed), row["date"] + ":qualified")
        assert_close(row["one_to_one_role_qualified_recall"], one_to_one / max(1, weed), row["date"] + ":one")
        assert_close(row["crop_overlap_scene_frequency"], crop_scenes / len(tiles_for_date), row["date"] + ":crop")
    write_tsv(out / "revision8_target_semantic_by_date.tsv", date_rows)

    outputs = [
        out / "revision8_queue_policy_summary.tsv",
        out / "revision8_crop_exposure_summary.tsv",
        out / "revision8_paired_tile_delta_summary.tsv",
        out / "revision8_target_semantic_summary.tsv",
        out / "revision8_target_semantic_by_date.tsv",
    ]
    summary = {
        "status": "replayed_from_archived_revision8_tables",
        "revision7_core_summary_sha256": sha256(core_out / "summary.json"),
        "inputs": {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in inputs},
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
