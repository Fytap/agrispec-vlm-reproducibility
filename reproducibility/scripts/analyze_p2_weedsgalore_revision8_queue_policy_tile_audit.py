#!/usr/bin/env python3
"""Reviewer-requested queue-policy, tile, crop-exposure, and governance audit.

The component-masked frozen DINOv2 ranker is refitted using the official train
split.  Regularization is selected on validation by the final fixed-K20 queue
metric, then the model is refitted on train+validation.  Test labels are used
only for post-selection offline scoring.  Equal-total-budget queue policies do
not use test truth; they use scores and candidate/image identities only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_commissioning_contract_audit import audit_setting, load_truth  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_proposal_audit import official_split_map  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_queue import (  # noqa: E402
    C_GRID,
    align_embeddings,
    exact_oracle_indices,
    fit_model,
    metric_record,
    predict,
    reconstruct_test_patch_proxy,
    selected_by_score,
)
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import read_tsv, sha256  # noqa: E402


PRIMARY_MODEL = "all_candidate_component_masked_dinov2_queue_selected"
EQUAL_BUDGET = 520
MIN_QUOTA = 5
MAX_QUOTA = 50
K = 20
CROP_COVERAGE_THRESHOLDS = (0.10, 0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project-root",
        "archive",
        "manifest",
        "candidate-root",
        "masked-dino-embeddings",
        "config",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-candidates-sha256", required=True)
    parser.add_argument("--expected-masked-dino-embeddings-sha256", required=True)
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def global_top_n(rows: list[dict[str, Any]], scores: np.ndarray, indices: list[int], budget: int) -> list[int]:
    return sorted(indices, key=lambda i: (-float(scores[i]), str(rows[i]["candidate_id"])))[:budget]


def bounded_global(
    rows: list[dict[str, Any]], scores: np.ndarray, indices: list[int], budget: int,
    minimum: int = MIN_QUOTA, maximum: int = MAX_QUOTA,
) -> list[int]:
    by_file: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_file[str(rows[index]["file"])].append(index)
    ranked = {
        file_name: sorted(values, key=lambda i: (-float(scores[i]), str(rows[i]["candidate_id"])))
        for file_name, values in by_file.items()
    }
    selected = [index for file_name in sorted(ranked) for index in ranked[file_name][:minimum]]
    selected_set = set(selected)
    remainder = [
        index
        for file_name in sorted(ranked)
        for index in ranked[file_name][minimum:maximum]
        if index not in selected_set
    ]
    remainder.sort(key=lambda i: (-float(scores[i]), str(rows[i]["candidate_id"])))
    needed = budget - len(selected)
    if needed < 0 or needed > len(remainder):
        raise ValueError("Bounded allocation cannot satisfy the frozen total budget")
    return selected + remainder[:needed]


def queue_selection_metrics(
    rows: list[dict[str, Any]], selected: list[int], instances: list[dict[str, Any]],
    audit: dict[str, Any], aggregation: str, stratum: str,
) -> dict[str, Any]:
    return metric_record(
        rows, selected, instances, audit["spatial"], audit["qualified"], aggregation, stratum
    )


def tune_regularization(
    rows: list[dict[str, Any]], embeddings: np.ndarray, audit: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    labels = np.asarray([row["truth"] == "weed" for row in rows], dtype=np.int64)
    train = [i for i, row in enumerate(rows) if row["official_split"] == "train"]
    validation = [i for i, row in enumerate(rows) if row["official_split"] == "val"]
    validation_instances = [row for row in audit["instances"] if row["official_split"] == "val"]
    records: list[dict[str, Any]] = []
    for c_value in C_GRID:
        fitted = fit_model(embeddings[train], labels[train], float(c_value))
        scores = predict(fitted, embeddings)
        selected = selected_by_score(rows, scores, validation, K)
        metrics = queue_selection_metrics(
            rows, selected, validation_instances, audit, "validation", "ALL"
        )
        records.append({"C": c_value, **metrics})
    chosen = max(
        records,
        key=lambda row: (
            float(row["one_to_one_role_qualified_recall"]),
            float(row["queued_role_qualified_weed_instance_recall"]),
            float(row["all_candidate_queue_precision"]),
            -abs(math.log10(float(row["C"]))),
        ),
    )
    return float(chosen["C"]), records


def select_score_threshold(
    rows: list[dict[str, Any]], scores: np.ndarray, validation: list[int], maximum: int
) -> float:
    ranked = sorted(validation, key=lambda i: (-float(scores[i]), str(rows[i]["candidate_id"])))
    if len(ranked) <= maximum:
        return float("-inf")
    boundary = float(scores[ranked[maximum - 1]])
    following = float(scores[ranked[maximum]])
    return (boundary + following) / 2.0 if boundary > following else boundary


def per_tile_rows(
    rows: list[dict[str, Any]], selected: list[int], instances: list[dict[str, Any]],
    audit: dict[str, Any], policy: str,
) -> list[dict[str, Any]]:
    by_file_selected: dict[str, list[int]] = defaultdict(list)
    for index in selected:
        by_file_selected[str(rows[index]["file"])].append(index)
    all_files = sorted({str(row["sample_id"]) for row in instances})
    output = []
    for file_name in all_files:
        local_instances = [row for row in instances if str(row["sample_id"]) == file_name]
        metric = queue_selection_metrics(
            rows, by_file_selected.get(file_name, []), local_instances, audit, "tile", file_name
        )
        output.append({"policy": policy, "file": file_name, "date": str(local_instances[0]["date"]), **metric})
    return output


def paired_delta_summary(tile_rows: list[dict[str, Any]], comparator: str) -> list[dict[str, Any]]:
    by_key = {(str(row["policy"]), str(row["file"])): row for row in tile_rows}
    files = sorted({str(row["file"]) for row in tile_rows if row["policy"] == "fixed_K20_per_image"})
    output = []
    metrics = (
        "queued_role_qualified_weed_instance_recall",
        "one_to_one_role_qualified_recall",
        "all_candidate_queue_precision",
    )
    for metric in metrics:
        values = np.asarray([
            float(by_key[(comparator, file_name)][metric])
            - float(by_key[("fixed_K20_per_image", file_name)][metric])
            for file_name in files
        ])
        output.append({
            "comparator_minus_fixed": comparator,
            "metric": metric,
            "tiles": len(values),
            "median_delta": float(np.median(values)),
            "q25_delta": float(np.quantile(values, 0.25)),
            "q75_delta": float(np.quantile(values, 0.75)),
            "minimum_delta": float(values.min()),
            "maximum_delta": float(values.max()),
            "improved_tiles": int((values > 1e-12).sum()),
            "unchanged_tiles": int((np.abs(values) <= 1e-12).sum()),
            "worsened_tiles": int((values < -1e-12).sum()),
        })
    return output


def crop_exposure_summary(
    rows: list[dict[str, Any]], selected: list[int], matches: list[dict[str, Any]],
    instances: list[dict[str, Any]], policy: str,
) -> dict[str, Any]:
    candidate_ids = {str(rows[index]["candidate_id"]) for index in selected}
    fractions = np.asarray([float(rows[index]["crop_fraction_candidate"]) for index in selected])
    crop_pixels = np.asarray([
        float(rows[index]["crop_fraction_candidate"]) * (math.exp(float(rows[index]["log1p_area_pixels"])) - 1.0)
        for index in selected
    ])
    crop_matches = [
        row for row in matches
        if row["truth_role"] == "crop" and str(row["candidate_id"]) in candidate_ids
    ]
    best: dict[str, float] = defaultdict(float)
    for row in crop_matches:
        iid = str(row["instance_id"])
        best[iid] = max(best[iid], float(row["instance_coverage"]))
    crop_truth_ids = {str(row["instance_id"]) for row in instances if row["truth_role"] == "crop"}
    record: dict[str, Any] = {
        "policy": policy,
        "accepted_candidates": len(selected),
        "candidates_with_any_crop_overlap": int((fractions > 0).sum()),
        "crop_fraction_median": float(np.median(fractions)),
        "crop_fraction_q75": float(np.quantile(fractions, 0.75)),
        "crop_fraction_q95": float(np.quantile(fractions, 0.95)),
        "crop_fraction_maximum": float(fractions.max()),
        "crop_overlap_pixels_median": float(np.median(crop_pixels)),
        "crop_overlap_pixels_q95": float(np.quantile(crop_pixels, 0.95)),
        "crop_overlap_pixels_maximum": float(crop_pixels.max()),
        "crop_truth_instances": len(crop_truth_ids),
    }
    for threshold in CROP_COVERAGE_THRESHOLDS:
        count = sum(value >= threshold for value in best.values())
        record[f"crop_instances_covered_at_{threshold:.2f}"] = count
        record[f"crop_instance_coverage_rate_at_{threshold:.2f}"] = count / max(1, len(crop_truth_ids))
    return record


def descriptive_groups(
    rows: list[dict[str, Any]], scores: np.ndarray, selected: list[int], instances: list[dict[str, Any]],
    audit: dict[str, Any], policy: str, patch_map: dict[str, str]
) -> list[dict[str, Any]]:
    del scores
    output = []
    for date in sorted({str(row["date"]) for row in instances}):
        local_selected = [index for index in selected if str(rows[index]["date"]) == date]
        local_instances = [row for row in instances if str(row["date"]) == date]
        output.append({"policy": policy, **queue_selection_metrics(
            rows, local_selected, local_instances, audit, "acquisition_date", date
        )})
    for patch in sorted(set(patch_map.values())):
        local_selected = [index for index in selected if patch_map[str(rows[index]["file"])] == patch]
        local_instances = [row for row in instances if patch_map[str(row["sample_id"])] == patch]
        output.append({"policy": policy, **queue_selection_metrics(
            rows, local_selected, local_instances, audit, "test_patch_proxy", patch
        )})
    return output


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("This audit is CPU-only")
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    checks = (
        (args.archive, args.expected_archive_sha256),
        (args.manifest, args.expected_manifest_sha256),
        (args.candidate_root / "candidate_components.tsv", args.expected_candidates_sha256),
        (args.masked_dino_embeddings, args.expected_masked_dino_embeddings_sha256),
    )
    for path, expected in checks:
        if sha256(path.resolve()) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
    started = time.perf_counter()
    manifest = read_tsv(args.manifest.resolve())
    split_map = official_split_map(args.archive.resolve(), manifest)
    truth = load_truth(args.archive.resolve(), manifest)
    audit = audit_setting(args.candidate_root.resolve(), manifest, truth)
    rows = audit["rows"]
    for row in rows:
        row["official_split"] = split_map[str(row["file"])]
    for row in audit["instances"]:
        row["official_split"] = split_map[str(row["sample_id"])]
    embeddings = align_embeddings(rows, args.masked_dino_embeddings.resolve())
    selected_c, sensitivity_rows = tune_regularization(rows, embeddings, audit)
    labels = np.asarray([row["truth"] == "weed" for row in rows], dtype=np.int64)
    fit_indices = [i for i, row in enumerate(rows) if row["official_split"] in {"train", "val"}]
    model = fit_model(embeddings[fit_indices], labels[fit_indices], selected_c)
    scores = predict(model, embeddings)
    validation_indices = [i for i, row in enumerate(rows) if row["official_split"] == "val"]
    test_indices = [i for i, row in enumerate(rows) if row["official_split"] == "test"]
    test_instances = [row for row in audit["instances"] if row["official_split"] == "test"]
    if len({str(rows[index]["file"]) for index in test_indices}) != 26:
        raise ValueError("Expected 26 official test tiles")
    fixed = selected_by_score(rows, scores, test_indices, K)
    if len(fixed) != EQUAL_BUDGET:
        raise ValueError(f"Fixed K20 accepted {len(fixed)}, not {EQUAL_BUDGET}")
    global_selected = global_top_n(rows, scores, test_indices, EQUAL_BUDGET)
    bounded = bounded_global(rows, scores, test_indices, EQUAL_BUDGET)
    threshold = select_score_threshold(rows, scores, validation_indices, EQUAL_BUDGET)
    threshold_selected = [index for index in test_indices if float(scores[index]) >= threshold]
    policies = {
        "fixed_K20_per_image": fixed,
        "global_top_N520": global_selected,
        "bounded_global_min5_max50_N520": bounded,
        "validation_score_threshold_transferred": threshold_selected,
    }
    patch_map, patch_rows = reconstruct_test_patch_proxy(manifest, split_map)
    pooled_rows = []
    tile_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    crop_rows = []
    for policy, selected in policies.items():
        pooled_rows.append({"model": PRIMARY_MODEL, "policy": policy, **queue_selection_metrics(
            rows, selected, test_instances, audit, "pooled_test", "ALL"
        )})
        tile_rows.extend(per_tile_rows(rows, selected, test_instances, audit, policy))
        strata_rows.extend(descriptive_groups(rows, scores, selected, test_instances, audit, policy, patch_map))
        crop_rows.append(crop_exposure_summary(
            rows, selected, audit["matches"], test_instances, policy
        ))
    paired_rows = []
    for comparator in ("global_top_N520", "bounded_global_min5_max50_N520"):
        paired_rows.extend(paired_delta_summary(tile_rows, comparator))

    # A true 1% scene-frequency cap on 26 scenes permits floor(0.01*26)=0 exposed scenes.
    allowed_crop_scenes = math.floor(0.01 * 26)
    weed_ids = {str(row["instance_id"]) for row in test_instances if row["truth_role"] == "weed"}
    no_crop_oracle, oracle_exact = exact_oracle_indices(
        rows, test_indices, audit["qualified"], weed_ids, K, maximum_crop_fraction=0.0
    )
    no_crop_metrics = queue_selection_metrics(
        rows, no_crop_oracle, test_instances, audit, "pooled_test", "ALL"
    )
    oracle_row = {
        "scene_frequency_alpha": 0.01,
        "test_scenes": 26,
        "integer_rule": "floor(alpha * test_scenes)",
        "allowed_crop_exposed_scenes": allowed_crop_scenes,
        "implementation_for_zero_allowed_scenes": "exclude every candidate with crop_fraction_candidate > 0",
        "optimization_exact_all_images": int(oracle_exact),
        **no_crop_metrics,
    }
    burden = Counter()
    by_file_count = Counter(str(rows[index]["file"]) for index in test_indices)
    values = np.asarray(list(by_file_count.values()), dtype=float)
    burden.update({"images": len(values), "candidates": len(test_indices)})
    burden_row = {
        "images": int(burden["images"]),
        "candidates": int(burden["candidates"]),
        "mean_candidates_per_image": float(values.mean()),
        "median_candidates_per_image": float(np.median(values)),
        "q25_candidates_per_image": float(np.quantile(values, 0.25)),
        "q75_candidates_per_image": float(np.quantile(values, 0.75)),
        "minimum_candidates_per_image": int(values.min()),
        "maximum_candidates_per_image": int(values.max()),
        "images_above_100_candidates": int((values > 100).sum()),
    }
    workload_rows = []
    for policy, selected in policies.items():
        for seconds in (1, 2, 3):
            workload_rows.append({
                "policy": policy,
                "accepted_candidates": len(selected),
                "seconds_per_candidate_scenario": seconds,
                "total_minutes_for_26_tiles": len(selected) * seconds / 60.0,
                "mean_minutes_per_tile": len(selected) * seconds / 60.0 / 26.0,
                "interpretation": "scenario only; no operator timing study was performed",
            })
    prediction_rows = [{
        "candidate_id": rows[index]["candidate_id"],
        "file": rows[index]["file"],
        "date": rows[index]["date"],
        "dataset_role_target": rows[index]["truth"],
        "score": f"{float(scores[index]):.12f}",
    } for index in test_indices]

    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "regularization": output / "validation_queue_regularization_selection.tsv",
        "pooled": output / "official_test_queue_policy_comparison.tsv",
        "tile": output / "official_test_queue_policy_per_tile.tsv",
        "paired": output / "official_test_paired_tile_delta_summary.tsv",
        "strata": output / "official_test_queue_policy_date_patch.tsv",
        "crop": output / "official_test_crop_exposure_severity.tsv",
        "patch": output / "official_test_patch_proxy_mapping.tsv",
        "workload": output / "operator_workload_scenarios.tsv",
        "predictions": output / "official_test_queue_selected_predictions.tsv",
    }
    for key, data in (
        ("regularization", sensitivity_rows),
        ("pooled", pooled_rows),
        ("tile", tile_rows),
        ("paired", paired_rows),
        ("strata", strata_rows),
        ("crop", crop_rows),
        ("patch", patch_rows),
        ("workload", workload_rows),
        ("predictions", prediction_rows),
    ):
        write_tsv(paths[key], data)
    (output / "candidate_burden_distribution.json").write_text(json.dumps(burden_row, indent=2) + "\n", encoding="utf-8")
    (output / "crop_scene_constraint_oracle.json").write_text(json.dumps(oracle_row, indent=2) + "\n", encoding="utf-8")
    governance = {
        "analysis_class": "reviewer_requested_post_test_diagnostic",
        "test_labels_not_used_for": [
            "regularization selection",
            "score-threshold selection",
            "equal-budget policy allocation",
            "candidate-ID generation",
        ],
        "test_scores_and_image_IDs_used_for": [
            "global top-N allocation",
            "bounded min/max quota allocation",
        ],
        "test_labels_used_for": "offline scoring after all policy selections",
        "population_confidence_interval": "not reported because only two test spatial patches from one field",
    }
    (output / "test_governance.json").write_text(json.dumps(governance, indent=2) + "\n", encoding="utf-8")
    summary = {
        "run_id": args.run_id,
        "status": "completed_reviewer_requested_post_test_diagnostic",
        "selected_C_by_validation_queue_metric": selected_c,
        "validation_transferred_score_threshold": threshold,
        "equal_total_budget": EQUAL_BUDGET,
        "pooled_results": {row["policy"]: row for row in pooled_rows},
        "candidate_burden": burden_row,
        "crop_scene_constraint_oracle": oracle_row,
        "governance": governance,
        "operator_task_definition": {
            "current_offline_task": "candidate triage: confirm likely weed and optionally flag the region for mask correction",
            "not_evaluated": ["treatment selection", "automatic spraying", "operator timing", "error correction", "actuation"],
            "workload_rows_are_scenarios_not_measurements": True,
        },
        "inputs": {
            "archive_sha256": sha256(args.archive.resolve()),
            "manifest_sha256": sha256(args.manifest.resolve()),
            "candidates_sha256": sha256(args.candidate_root / "candidate_components.tsv"),
            "masked_dino_embeddings_sha256": sha256(args.masked_dino_embeddings.resolve()),
            "config_sha256": sha256(args.config.resolve()),
        },
        "outputs": {key + "_sha256": sha256(path) for key, path in paths.items()},
        "runtime_seconds": time.perf_counter() - started,
        "script_sha256": sha256(Path(__file__).resolve()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "limitations": [
            "Patch membership is a capture-index proxy because authoritative patch IDs/geocoordinates are absent from the public archive.",
            "The 26 test tiles come from two spatial patches in one field; no population confidence interval is reported.",
            "Workload conversions assume 1/2/3 seconds per candidate and are not operator measurements.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
