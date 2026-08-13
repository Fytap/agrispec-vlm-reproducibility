#!/usr/bin/env python3
"""Post-test model-contract and standard instance-metric diagnostics.

This script replays the validation-selected checkpoints on the unchanged
WeedsGalore official test split.  It does not select a checkpoint, proposal
setting, queue size, or external-domain threshold.  The outputs are
exploratory sensitivity analyses intended to separate three questions:

1. how the three target-mask models behave under a 3 x 3 x 3 matching grid;
2. how queue-conditioned recovery compares with IoU-based AP/AR diagnostics;
3. whether the primary archived K=20 endpoint is reproduced exactly.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


INSTANCE_COVERAGES = (0.25, 0.50, 0.75)
LABELED_COVERAGES = (0.25, 0.50, 0.75)
ROLE_PURITIES = (0.50, 0.75, 0.90)
IOU_THRESHOLDS = tuple(float(value) for value in np.arange(0.50, 0.951, 0.05))
QUEUE_K = 20
STANDARD_MAX_DETECTIONS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, choices=(6, 7), required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def maximum_matching(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Maximum-cardinality, then maximum-IoU matching above a threshold."""
    if iou.size == 0:
        return []
    n_pred, n_truth = iou.shape
    size = max(n_pred, n_truth)
    cost = np.zeros((size, size), dtype=np.float64)
    valid = iou >= threshold
    cost[:n_pred, :n_truth][valid] = -(1000.0 + iou[valid])
    rows, cols = linear_sum_assignment(cost)
    return [
        (int(row), int(col))
        for row, col in zip(rows.tolist(), cols.tolist())
        if row < n_pred and col < n_truth and iou[row, col] >= threshold
    ]


def interpolated_ap(records: list[dict[str, Any]], total_truth: int, threshold: float) -> float:
    """COCO-style 101-point interpolated AP for one weed class."""
    ordered = sorted(records, key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
    matched: dict[str, set[int]] = defaultdict(set)
    tp: list[int] = []
    fp: list[int] = []
    for row in ordered:
        ious = np.asarray(row["ious"], dtype=np.float64)
        best_index = -1
        best_value = -1.0
        for index, value in enumerate(ious.tolist()):
            if index in matched[str(row["file"])] or value < threshold:
                continue
            if value > best_value:
                best_index = index
                best_value = value
        if best_index >= 0:
            matched[str(row["file"])].add(best_index)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    if not ordered or total_truth <= 0:
        return 0.0
    tp_cumulative = np.cumsum(tp)
    fp_cumulative = np.cumsum(fp)
    recall = tp_cumulative / total_truth
    precision = tp_cumulative / np.maximum(1, tp_cumulative + fp_cumulative)
    values = []
    for recall_level in np.linspace(0.0, 1.0, 101):
        eligible = precision[recall >= recall_level]
        values.append(float(eligible.max()) if eligible.size else 0.0)
    return float(np.mean(values))


def semantic_candidates(
    common: Any,
    stem: str,
    probabilities: np.ndarray,
    root: Path,
    threshold: float,
    minimum_area: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    semantic, instance_map, instances = common.truth_instances(stem, root)
    weed_records = [record for record in instances if record.role == "weed"]
    weed_index = {record.value: index for index, record in enumerate(weed_records)}
    plant_probability = 1.0 - probabilities[0]
    labels, count = ndimage.label(plant_probability >= threshold, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    rows: list[dict[str, Any]] = []
    for component in range(1, count + 1):
        area = int(areas[component])
        if area < minimum_area:
            continue
        mask = labels == component
        crop_pixels = int((semantic[mask] == 1).sum())
        weed_pixels = int((semantic[mask] == 2).sum())
        labeled_pixels = crop_pixels + weed_pixels
        plant_mean = float((probabilities[1][mask] + probabilities[2][mask]).mean())
        score = float(probabilities[2][mask].mean()) / max(1e-8, plant_mean)
        overlap = np.zeros(len(weed_records), dtype=np.int64)
        values, counts = np.unique(instance_map[mask], return_counts=True)
        for value, pixels in zip(values.tolist(), counts.tolist()):
            index = weed_index.get(int(value))
            if index is not None:
                overlap[index] = int(pixels)
        truth_areas = np.asarray([record.area for record in weed_records], dtype=np.int64)
        union = area + truth_areas - overlap
        rows.append(
            {
                "candidate_id": (
                    f"{stem}:semantic:T{int(round(threshold * 100)):03d}:"
                    f"A{minimum_area:03d}:C{component:05d}"
                ),
                "file": stem,
                "score": score,
                "predicted_role": "weed" if score >= 0.5 else "crop",
                "area": area,
                "labeled_coverage": labeled_pixels / max(1, area),
                "weed_purity": weed_pixels / max(1, labeled_pixels),
                "crop_fraction": crop_pixels / max(1, area),
                "instance_coverages": overlap / np.maximum(1, truth_areas),
                "ious": overlap / np.maximum(1, union),
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
    return rows, [record.instance_id for record in weed_records]


def maskrcnn_candidates(
    common: Any,
    stem: str,
    predictions: list[dict[str, Any]],
    root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    semantic, instance_map, instances = common.truth_instances(stem, root)
    weed_records = [record for record in instances if record.role == "weed"]
    weed_index = {record.value: index for index, record in enumerate(weed_records)}
    truth_areas = np.asarray([record.area for record in weed_records], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions, 1):
        mask = np.asarray(prediction["mask"], dtype=bool)
        area = int(mask.sum())
        if area == 0:
            continue
        crop_pixels = int((semantic[mask] == 1).sum())
        weed_pixels = int((semantic[mask] == 2).sum())
        labeled_pixels = crop_pixels + weed_pixels
        predicted_label = int(prediction["label"])
        detection_score = float(prediction["score"])
        overlap = np.zeros(len(weed_records), dtype=np.int64)
        values, counts = np.unique(instance_map[mask], return_counts=True)
        for value, pixels in zip(values.tolist(), counts.tolist()):
            truth_index = weed_index.get(int(value))
            if truth_index is not None:
                overlap[truth_index] = int(pixels)
        union = area + truth_areas - overlap
        rows.append(
            {
                "candidate_id": f"{stem}:maskrcnn:C{index:05d}",
                "file": stem,
                "score": detection_score if predicted_label == 2 else -detection_score,
                "predicted_role": "weed" if predicted_label == 2 else "crop",
                "area": area,
                "labeled_coverage": labeled_pixels / max(1, area),
                "weed_purity": weed_pixels / max(1, labeled_pixels),
                "crop_fraction": crop_pixels / max(1, area),
                "instance_coverages": overlap / np.maximum(1, truth_areas),
                "ious": overlap / np.maximum(1, union),
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
    return rows, [record.instance_id for record in weed_records]


def contract_counts(
    accepted: list[dict[str, Any]],
    n_truth: int,
    instance_coverage: float,
    labeled_coverage: float,
    role_purity: float,
) -> dict[str, int]:
    spatial_edges: dict[str, set[int]] = {}
    qualified_edges: dict[str, set[int]] = {}
    for row in accepted:
        candidate_id = str(row["candidate_id"])
        coverage = np.asarray(row["instance_coverages"], dtype=np.float64)
        spatial = set(np.flatnonzero(coverage >= instance_coverage).tolist())
        spatial_edges[candidate_id] = spatial
        qualified_edges[candidate_id] = (
            spatial
            if float(row["labeled_coverage"]) >= labeled_coverage
            and float(row["weed_purity"]) >= role_purity
            else set()
        )
    spatial_set = set().union(*(spatial_edges.values())) if spatial_edges else set()
    qualified_set = set().union(*(qualified_edges.values())) if qualified_edges else set()
    owner: dict[int, str] = {}

    def augment(candidate_id: str, visited: set[int]) -> bool:
        for truth_index in sorted(qualified_edges.get(candidate_id, set())):
            if truth_index in visited:
                continue
            visited.add(truth_index)
            previous = owner.get(truth_index)
            if previous is None or augment(previous, visited):
                owner[truth_index] = candidate_id
                return True
        return False

    for row in sorted(accepted, key=lambda item: str(item["candidate_id"])):
        augment(str(row["candidate_id"]), set())
    eligible = sum(
        float(row["labeled_coverage"]) >= labeled_coverage and float(row["weed_purity"]) >= role_purity
        for row in accepted
    )
    return {
        "weed_instances": n_truth,
        "accepted_candidates": len(accepted),
        "eligible_candidates": int(eligible),
        "spatial_recalled": len(spatial_set),
        "set_coverage_recalled": len(qualified_set),
        "one_to_one_recalled": len(owner),
    }


def discover_runs(root: Path) -> list[dict[str, Any]]:
    result_root = root / "results" / "p2_development"
    specifications = (
        ("unet", "ResNet18--U-Net", "P2_WEEDSGALORE_TARGET_SEMANTIC_H200_SEED_*_20260812_v1"),
        ("deeplabv3plus", "DeepLabV3+--ResNet50", "P2_WEEDSGALORE_DEEPLABV3PLUS_H200_SEED_*_20260812_v*"),
        ("maskrcnn", "Mask R-CNN--ResNet50-FPN-v2", "P2_WEEDSGALORE_MASKRCNN_H200_SEED_*_20260812_v1"),
    )
    rows: list[dict[str, Any]] = []
    for model_key, model_label, pattern in specifications:
        for directory in sorted(result_root.glob(pattern)):
            summary_path = directory / "summary.json"
            checkpoint = directory / "best_validation_queue.pt"
            if not summary_path.is_file() or not checkpoint.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model_key": model_key,
                    "model": model_label,
                    "seed": int(summary["training"]["seed"]),
                    "directory": directory,
                    "checkpoint": checkpoint,
                    "summary": summary,
                }
            )
    expected = {"unet": 5, "deeplabv3plus": 3, "maskrcnn": 3}
    observed = Counter(row["model_key"] for row in rows)
    if dict(observed) != expected:
        raise ValueError(f"Unexpected checkpoint registry: {dict(observed)}")
    return rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    expected_visible = str(args.physical_gpu_id)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RuntimeError("Explicit GPU assignment mismatch")
    root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    scripts = root / "scripts"
    sys.path.insert(0, str(scripts))
    import torch
    import train_evaluate_p2_weedsgalore_target_semantic_seed_stability_h200 as common
    import train_evaluate_p2_weedsgalore_deeplabv3plus_h200_v8 as deep
    import train_evaluate_p2_weedsgalore_maskrcnn_h200_v4 as maskmod

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    test_stems = common.read_split(dataset_root, "test")
    runs = discover_runs(root)
    contract_accumulators: dict[tuple[str, int, float, float, float], Counter] = defaultdict(Counter)
    per_image_rows: list[dict[str, Any]] = []
    standard_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    run_manifest: list[dict[str, Any]] = []

    encoder_weights = root / "data/model-cache/SAT_STRONG_BASELINES_20260812_v1/resnet50-19c8e357.pth"
    mask_weights = root / "data/model-cache/SAT_STRONG_BASELINES_20260812_v1/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
    for run in runs:
        checkpoint_hash = sha256(run["checkpoint"])
        expected_hash = str(run["summary"]["model"]["checkpoint_sha256"])
        if checkpoint_hash != expected_hash:
            raise ValueError(f"Checkpoint hash mismatch: {run['directory']}")
        if run["model_key"] == "unet":
            model = common.build_model().to(device)
        elif run["model_key"] == "deeplabv3plus":
            model = deep.build_model(encoder_weights, "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097").to(device)
        else:
            model = maskmod.build_model(mask_weights, "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e").to(device)
        checkpoint = torch.load(run["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if run["model_key"] == "maskrcnn":
            predictions, inference_seconds, peak_memory = maskmod.predict(model, test_stems, dataset_root, device)
        else:
            predictions, inference_seconds, peak_memory = common.predict(model, test_stems, dataset_root, device)
        selection = run["summary"]["selected_on_validation"]
        threshold = float(selection.get("threshold", 0.0))
        minimum_area = int(selection.get("minimum_area_pixels", 0))
        total_truth = 0
        standard_detections: list[dict[str, Any]] = []
        standard_match_totals = Counter()
        queue_match_totals = Counter()
        queue_pq_tp = 0
        queue_pq_fp = 0
        queue_pq_fn = 0
        queue_pq_iou = 0.0
        for stem in test_stems:
            if run["model_key"] == "maskrcnn":
                candidates, truth_ids = maskrcnn_candidates(common, stem, predictions[stem], dataset_root)
            else:
                candidates, truth_ids = semantic_candidates(
                    common, stem, predictions[stem], dataset_root, threshold, minimum_area
                )
            accepted = candidates[:QUEUE_K]
            weed_predictions = [row for row in candidates if row["predicted_role"] == "weed"][:STANDARD_MAX_DETECTIONS]
            total_truth += len(truth_ids)
            for row in weed_predictions:
                standard_detections.append(row)
            standard_iou = (
                np.stack([np.asarray(row["ious"], dtype=np.float64) for row in weed_predictions])
                if weed_predictions else np.zeros((0, len(truth_ids)), dtype=np.float64)
            )
            queue_weed = [row for row in accepted if row["predicted_role"] == "weed"]
            queue_iou = (
                np.stack([np.asarray(row["ious"], dtype=np.float64) for row in queue_weed])
                if queue_weed else np.zeros((0, len(truth_ids)), dtype=np.float64)
            )
            image_standard = {}
            for value in IOU_THRESHOLDS:
                standard_count = len(maximum_matching(standard_iou, value))
                queue_count = len(maximum_matching(queue_iou, value))
                standard_match_totals[f"{value:.2f}"] += standard_count
                queue_match_totals[f"{value:.2f}"] += queue_count
                image_standard[f"AR100_IoU_{value:.2f}"] = standard_count / max(1, len(truth_ids))
                image_standard[f"AR20_IoU_{value:.2f}"] = queue_count / max(1, len(truth_ids))
            pq_matches = maximum_matching(queue_iou, 0.5000000001)
            queue_pq_tp += len(pq_matches)
            queue_pq_fp += len(queue_weed) - len(pq_matches)
            queue_pq_fn += len(truth_ids) - len(pq_matches)
            queue_pq_iou += sum(float(queue_iou[row, col]) for row, col in pq_matches)
            crop_values = [float(row["crop_fraction"]) for row in accepted]
            primary = contract_counts(accepted, len(truth_ids), 0.50, 0.50, 0.90)
            per_image_rows.append(
                {
                    "model": run["model"],
                    "seed": run["seed"],
                    "file": stem,
                    "date": stem[:10],
                    **primary,
                    "candidate_precision": primary["eligible_candidates"] / max(1, primary["accepted_candidates"]),
                    "spatial_recall": primary["spatial_recalled"] / max(1, primary["weed_instances"]),
                    "set_coverage_recall": primary["set_coverage_recalled"] / max(1, primary["weed_instances"]),
                    "one_to_one_recall": primary["one_to_one_recalled"] / max(1, primary["weed_instances"]),
                    "crop_exposure_at_0.00": int(any(value > 0 for value in crop_values)),
                    "crop_exposure_at_0.01": int(any(value >= 0.01 for value in crop_values)),
                    "crop_exposure_at_0.10": int(any(value >= 0.10 for value in crop_values)),
                    **image_standard,
                }
            )
            for instance_coverage in INSTANCE_COVERAGES:
                for labeled_coverage in LABELED_COVERAGES:
                    for role_purity in ROLE_PURITIES:
                        counts = contract_counts(
                            accepted, len(truth_ids), instance_coverage, labeled_coverage, role_purity
                        )
                        key = (run["model"], run["seed"], instance_coverage, labeled_coverage, role_purity)
                        contract_accumulators[key].update(counts)
                        contract_accumulators[key]["images"] += 1
                        contract_accumulators[key]["crop_exposure_at_0.00"] += int(any(value > 0 for value in crop_values))
                        contract_accumulators[key]["crop_exposure_at_0.01"] += int(any(value >= 0.01 for value in crop_values))
                        contract_accumulators[key]["crop_exposure_at_0.10"] += int(any(value >= 0.10 for value in crop_values))
        aps = {f"{value:.2f}": interpolated_ap(standard_detections, total_truth, value) for value in IOU_THRESHOLDS}
        pq_sq = queue_pq_iou / max(1, queue_pq_tp)
        pq_rq = queue_pq_tp / max(1e-12, queue_pq_tp + 0.5 * queue_pq_fp + 0.5 * queue_pq_fn)
        standard_rows.append(
            {
                "model": run["model"],
                "seed": run["seed"],
                "weed_instances": total_truth,
                "weed_predictions_max100_per_image": len(standard_detections),
                "mAP_IoU_0.50_0.95": float(np.mean(list(aps.values()))),
                "AP_IoU_0.50": aps["0.50"],
                "AP_IoU_0.75": aps["0.75"],
                "AR100_IoU_0.50_0.95": float(np.mean([standard_match_totals[f"{v:.2f}"] / total_truth for v in IOU_THRESHOLDS])),
                "AR100_IoU_0.50": standard_match_totals["0.50"] / total_truth,
                "AR100_IoU_0.75": standard_match_totals["0.75"] / total_truth,
                "queue_AR20_IoU_0.50_0.95": float(np.mean([queue_match_totals[f"{v:.2f}"] / total_truth for v in IOU_THRESHOLDS])),
                "queue_AR20_IoU_0.50": queue_match_totals["0.50"] / total_truth,
                "queue_AR20_IoU_0.75": queue_match_totals["0.75"] / total_truth,
                "queue_PQ_style": pq_sq * pq_rq,
                "queue_SQ_style": pq_sq,
                "queue_RQ_style": pq_rq,
                "queue_PQ_TP": queue_pq_tp,
                "queue_PQ_FP": queue_pq_fp,
                "queue_PQ_FN": queue_pq_fn,
            }
        )
        primary_key = (run["model"], run["seed"], 0.50, 0.50, 0.90)
        observed = contract_accumulators[primary_key]
        archived = run["summary"]["official_test_queue_metrics"]
        comparisons = {
            "accepted_candidates": (float(observed["accepted_candidates"]), float(archived["accepted_candidates"])),
            "candidate_precision": (
                observed["eligible_candidates"] / max(1, observed["accepted_candidates"]),
                float(archived["K20_all_candidate_precision"]),
            ),
            "spatial_recall": (
                observed["spatial_recalled"] / observed["weed_instances"],
                float(archived["K20_many_to_one_spatial_recall"]),
            ),
            "set_coverage_recall": (
                observed["set_coverage_recalled"] / observed["weed_instances"],
                float(archived["K20_many_to_one_role_qualified_recall"]),
            ),
            "one_to_one_recall": (
                observed["one_to_one_recalled"] / observed["weed_instances"],
                float(archived["K20_one_to_one_role_qualified_recall"]),
            ),
            "crop_exposure_at_0.00": (
                observed["crop_exposure_at_0.00"] / observed["images"],
                float(archived["K20_crop_overlap_scene_frequency"]),
            ),
        }
        for metric, (value, reference) in comparisons.items():
            verification_rows.append(
                {
                    "model": run["model"],
                    "seed": run["seed"],
                    "metric": metric,
                    "recomputed": value,
                    "archived": reference,
                    "absolute_difference": abs(value - reference),
                    "exact_within_1e_12": int(abs(value - reference) <= 1e-12),
                }
            )
        run_manifest.append(
            {
                "model": run["model"],
                "seed": run["seed"],
                "checkpoint": str(run["checkpoint"]),
                "checkpoint_sha256": checkpoint_hash,
                "validation_selected_threshold": threshold,
                "validation_selected_minimum_area": minimum_area,
                "test_inference_seconds": inference_seconds,
                "test_peak_gpu_memory_bytes": peak_memory,
            }
        )
        del predictions, checkpoint, model
        gc.collect()
        torch.cuda.empty_cache()

    contract_rows: list[dict[str, Any]] = []
    for key, values in sorted(contract_accumulators.items()):
        model, seed, instance_coverage, labeled_coverage, role_purity = key
        contract_rows.append(
            {
                "model": model,
                "seed": seed,
                "instance_coverage": instance_coverage,
                "candidate_labeled_coverage": labeled_coverage,
                "same_role_purity": role_purity,
                "weed_instances": values["weed_instances"],
                "accepted_candidates": values["accepted_candidates"],
                "candidate_precision": values["eligible_candidates"] / max(1, values["accepted_candidates"]),
                "spatial_recall": values["spatial_recalled"] / max(1, values["weed_instances"]),
                "set_coverage_recall": values["set_coverage_recalled"] / max(1, values["weed_instances"]),
                "one_to_one_recall": values["one_to_one_recalled"] / max(1, values["weed_instances"]),
                "crop_exposure_scene_frequency_at_0.00": values["crop_exposure_at_0.00"] / values["images"],
                "crop_exposure_scene_frequency_at_0.01": values["crop_exposure_at_0.01"] / values["images"],
                "crop_exposure_scene_frequency_at_0.10": values["crop_exposure_at_0.10"] / values["images"],
            }
        )

    metric_names = (
        "candidate_precision", "spatial_recall", "set_coverage_recall", "one_to_one_recall",
        "crop_exposure_scene_frequency_at_0.00", "crop_exposure_scene_frequency_at_0.01",
        "crop_exposure_scene_frequency_at_0.10",
    )
    model_contract_rows: list[dict[str, Any]] = []
    grouped_contracts: dict[tuple[str, float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in contract_rows:
        grouped_contracts[(str(row["model"]), float(row["instance_coverage"]), float(row["candidate_labeled_coverage"]), float(row["same_role_purity"]))].append(row)
    for key, rows in sorted(grouped_contracts.items()):
        model, instance_coverage, labeled_coverage, role_purity = key
        record: dict[str, Any] = {
            "model": model,
            "instance_coverage": instance_coverage,
            "candidate_labeled_coverage": labeled_coverage,
            "same_role_purity": role_purity,
            "seeds": len(rows),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in rows])
            record[f"mean_{metric}"] = float(values.mean())
            record[f"sd_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        model_contract_rows.append(record)

    ranking_rows: list[dict[str, Any]] = []
    for contract in sorted({(r["instance_coverage"], r["candidate_labeled_coverage"], r["same_role_purity"]) for r in model_contract_rows}):
        rows = [r for r in model_contract_rows if (r["instance_coverage"], r["candidate_labeled_coverage"], r["same_role_purity"]) == contract]
        ordered = sorted(rows, key=lambda row: (-float(row["mean_one_to_one_recall"]), str(row["model"])))
        ranking_rows.append(
            {
                "instance_coverage": contract[0],
                "candidate_labeled_coverage": contract[1],
                "same_role_purity": contract[2],
                "ranking_by_mean_one_to_one_recall": " > ".join(str(row["model"]) for row in ordered),
                "best_model": ordered[0]["model"],
                "best_minus_second": float(ordered[0]["mean_one_to_one_recall"]) - float(ordered[1]["mean_one_to_one_recall"]),
            }
        )

    standard_model_rows: list[dict[str, Any]] = []
    for model in sorted({str(row["model"]) for row in standard_rows}):
        rows = [row for row in standard_rows if str(row["model"]) == model]
        record: dict[str, Any] = {"model": model, "seeds": len(rows)}
        for metric in (
            "mAP_IoU_0.50_0.95", "AP_IoU_0.50", "AP_IoU_0.75",
            "AR100_IoU_0.50_0.95", "AR100_IoU_0.50", "AR100_IoU_0.75",
            "queue_AR20_IoU_0.50_0.95", "queue_AR20_IoU_0.50", "queue_AR20_IoU_0.75",
            "queue_PQ_style", "queue_SQ_style", "queue_RQ_style",
        ):
            values = np.asarray([float(row[metric]) for row in rows])
            record[f"mean_{metric}"] = float(values.mean())
            record[f"sd_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        standard_model_rows.append(record)

    write_tsv(output / "checkpoint_manifest.tsv", run_manifest)
    write_tsv(output / "primary_endpoint_replay_audit.tsv", verification_rows)
    write_tsv(output / "per_image_primary_and_standard_metrics.tsv", per_image_rows)
    write_tsv(output / "per_seed_contract_grid.tsv", contract_rows)
    write_tsv(output / "model_contract_grid.tsv", model_contract_rows)
    write_tsv(output / "contract_ranking_stability.tsv", ranking_rows)
    write_tsv(output / "per_seed_standard_instance_metrics.tsv", standard_rows)
    write_tsv(output / "model_standard_instance_metrics.tsv", standard_model_rows)
    result = {
        "run_id": output.name,
        "status": "completed_post_test_model_contract_and_standard_metric_diagnostics",
        "analysis_class": "post_test_exploratory_sensitivity_analysis",
        "official_test_split_changed": False,
        "model_or_threshold_selection_permitted": False,
        "queue_k": QUEUE_K,
        "standard_max_detections_per_image": STANDARD_MAX_DETECTIONS,
        "contract_grid": {
            "instance_coverage": INSTANCE_COVERAGES,
            "candidate_labeled_coverage": LABELED_COVERAGES,
            "same_role_purity": ROLE_PURITIES,
        },
        "iou_thresholds": IOU_THRESHOLDS,
        "checkpoint_runs": len(runs),
        "primary_replay_checks": len(verification_rows),
        "primary_replay_failures": sum(not int(row["exact_within_1e_12"]) for row in verification_rows),
        "hardware": {
            "physical_gpu_id": args.physical_gpu_id,
            "gpu": torch.cuda.get_device_name(device),
        },
        "runtime_seconds": time.perf_counter() - started,
        "script_sha256": sha256(Path(__file__).resolve()),
        "interpretation_scope": (
            "Same-field official-test diagnostics. AP/AR use predicted-weed regions; "
            "PQ-style values use the predicted-weed subset of the fixed K=20 review queue."
        ),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
