#!/usr/bin/env python3
"""Post-lock CWFID diagnostics without changing the locked external endpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_p2_cwfid_external_locked_h200_v1 as locked  # noqa: E402


THRESHOLD_GRID = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
CALIBRATION_EDGES = np.linspace(0.0, 1.0, 21)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--physical-gpu-id", type=int, choices=(6, 7), required=True)
    return p.parse_args()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if y.any() else float("nan")


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise RuntimeError("Physical GPU assignment mismatch")
    import torch

    root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    _, triplets = locked.audit_git_tree(dataset_root)
    if len(triplets) != 60:
        raise ValueError(f"Expected 60 CWFID images, found {len(triplets)}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    image_rows: list[dict[str, Any]] = []
    sweep_tiles: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    calibration: dict[tuple[int, str, int], Counter] = defaultdict(Counter)
    for seed in locked.SEEDS:
        run_dir = checkpoint_root / locked.RUN_TEMPLATE.format(seed=seed)
        summary_path = run_dir / "summary.json"
        checkpoint_path = run_dir / "best_validation_queue.pt"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = str(summary["model"]["checkpoint_sha256"])
        actual = locked.sha256(checkpoint_path)
        if expected != actual:
            raise ValueError(f"Checkpoint hash mismatch for seed {seed}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        selection = checkpoint["selection"]
        minimum_area = int(selection["minimum_area_pixels"])
        model = locked.build_model()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        checkpoint_rows.append({
            "seed": seed,
            "checkpoint_sha256": actual,
            "locked_validation_threshold": float(selection["threshold"]),
            "locked_minimum_area": minimum_area,
        })
        for image_id, image_path, annotation_path, _ in triplets:
            rgb, geometry = locked.load_rgb(image_path)
            normalized = (rgb.astype(np.float32) / 255.0 - locked.IMAGENET_MEAN) / locked.IMAGENET_STD
            tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                probabilities = model(tensor).softmax(dim=1)[0].float().cpu().numpy()
            semantic, _, _ = locked.load_annotation(annotation_path, geometry)
            plant_score = 1.0 - probabilities[0]
            weed_score = probabilities[2]
            conditional_weed = probabilities[2] / np.maximum(1e-12, probabilities[1] + probabilities[2])
            plant_truth = (semantic > 0).astype(np.uint8)
            weed_truth = (semantic == 2).astype(np.uint8)
            plant_pixels = semantic > 0
            row = {
                "seed": seed,
                "image_id": image_id,
                "pixels": int(semantic.size),
                "plant_pixels": int(plant_truth.sum()),
                "weed_pixels": int(weed_truth.sum()),
                "plant_pixel_auc": safe_auc(plant_truth.ravel(), plant_score.ravel()),
                "plant_pixel_ap": safe_ap(plant_truth.ravel(), plant_score.ravel()),
                "plant_pixel_brier": float(np.mean((plant_score - plant_truth) ** 2)),
                "weed_pixel_auc": safe_auc(weed_truth.ravel(), weed_score.ravel()),
                "weed_pixel_ap": safe_ap(weed_truth.ravel(), weed_score.ravel()),
                "weed_pixel_brier": float(np.mean((weed_score - weed_truth) ** 2)),
                "conditional_crop_vs_weed_auc_on_labeled_plant_pixels": safe_auc(weed_truth[plant_pixels], conditional_weed[plant_pixels]),
                "conditional_crop_vs_weed_ap_on_labeled_plant_pixels": safe_ap(weed_truth[plant_pixels], conditional_weed[plant_pixels]),
                "maximum_class_probability_q50": float(np.quantile(probabilities.max(axis=0), 0.50)),
                "maximum_class_probability_q90": float(np.quantile(probabilities.max(axis=0), 0.90)),
                "maximum_class_probability_q99": float(np.quantile(probabilities.max(axis=0), 0.99)),
                "foreground_probability_q50": float(np.quantile(plant_score, 0.50)),
                "foreground_probability_q90": float(np.quantile(plant_score, 0.90)),
                "foreground_probability_q99": float(np.quantile(plant_score, 0.99)),
                "foreground_probability_max": float(plant_score.max()),
                "weed_probability_q99": float(np.quantile(weed_score, 0.99)),
                "weed_probability_max": float(weed_score.max()),
            }
            image_rows.append(row)
            for endpoint, score, truth_values in (
                ("plant", plant_score, plant_truth),
                ("weed", weed_score, weed_truth),
            ):
                bins = np.minimum(np.searchsorted(CALIBRATION_EDGES, score.ravel(), side="right") - 1, len(CALIBRATION_EDGES) - 2)
                bins = np.maximum(bins, 0)
                counts = np.bincount(bins, minlength=len(CALIBRATION_EDGES) - 1)
                score_sums = np.bincount(bins, weights=score.ravel(), minlength=len(CALIBRATION_EDGES) - 1)
                positive_sums = np.bincount(bins, weights=truth_values.ravel(), minlength=len(CALIBRATION_EDGES) - 1)
                for index, count in enumerate(counts):
                    if count == 0:
                        continue
                    counter = calibration[(seed, endpoint, index)]
                    counter["pixels"] += int(count)
                    counter["score_sum"] += float(score_sums[index])
                    counter["positive_sum"] += int(positive_sums[index])
            for threshold in THRESHOLD_GRID:
                labels, candidates = locked.generate_candidates(seed, image_id, probabilities, threshold, minimum_area)
                tile, _, _, _, _ = locked.score_image(seed, image_id, probabilities, labels, candidates, semantic)
                sweep_tiles.append({"diagnostic_threshold": threshold, **tile})
        del model, checkpoint
        torch.cuda.empty_cache()

    calibration_rows: list[dict[str, Any]] = []
    for (seed, endpoint, index), counter in sorted(calibration.items()):
        pixels = int(counter["pixels"])
        calibration_rows.append({
            "seed": seed,
            "endpoint": endpoint,
            "bin_index": index,
            "bin_lower": float(CALIBRATION_EDGES[index]),
            "bin_upper": float(CALIBRATION_EDGES[index + 1]),
            "pixels": pixels,
            "mean_predicted_probability": float(counter["score_sum"] / pixels),
            "observed_positive_fraction": float(counter["positive_sum"] / pixels),
        })

    sweep_rows: list[dict[str, Any]] = []
    for seed in locked.SEEDS:
        for threshold in THRESHOLD_GRID:
            chosen = [row for row in sweep_tiles if int(row["seed"]) == seed and float(row["diagnostic_threshold"]) == threshold]
            totals = Counter()
            for row in chosen:
                for key in (
                    "weed_components", "candidates", "accepted_candidates", "eligible_weed_candidates",
                    "many_to_one_spatial_recalled", "many_to_one_role_qualified_recalled",
                    "one_to_one_spatial_recalled", "one_to_one_role_qualified_recalled", "crop_overlap_image",
                ):
                    totals[key] += int(row[key])
            sweep_rows.append({
                "seed": seed,
                "diagnostic_threshold": threshold,
                "images": len(chosen),
                "weed_components": totals["weed_components"],
                "candidates": totals["candidates"],
                "accepted_candidates": totals["accepted_candidates"],
                "images_with_any_candidate": sum(int(row["candidates"]) > 0 for row in chosen),
                "candidate_precision": totals["eligible_weed_candidates"] / max(1, totals["accepted_candidates"]),
                "spatial_recall": totals["many_to_one_spatial_recalled"] / max(1, totals["weed_components"]),
                "qualified_set_coverage_recall": totals["many_to_one_role_qualified_recalled"] / max(1, totals["weed_components"]),
                "one_to_one_recall": totals["one_to_one_role_qualified_recalled"] / max(1, totals["weed_components"]),
                "crop_exposure_image_frequency": totals["crop_overlap_image"] / max(1, len(chosen)),
            })

    seed_rows: list[dict[str, Any]] = []
    metric_names = [
        "plant_pixel_auc", "plant_pixel_ap", "plant_pixel_brier", "weed_pixel_auc", "weed_pixel_ap",
        "weed_pixel_brier", "conditional_crop_vs_weed_auc_on_labeled_plant_pixels",
        "conditional_crop_vs_weed_ap_on_labeled_plant_pixels", "maximum_class_probability_q50",
        "foreground_probability_q99", "foreground_probability_max", "weed_probability_q99", "weed_probability_max",
    ]
    for seed in locked.SEEDS:
        chosen = [row for row in image_rows if int(row["seed"]) == seed]
        record: dict[str, Any] = {"seed": seed, "images": len(chosen)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in chosen], dtype=float)
            record[f"image_macro_{metric}"] = float(np.nanmean(values))
            record[f"image_sample_sd_{metric}"] = float(np.nanstd(values, ddof=1))
        locked_threshold = next(float(row["locked_validation_threshold"]) for row in checkpoint_rows if int(row["seed"]) == seed)
        locked_sweep = next(row for row in sweep_rows if int(row["seed"]) == seed and float(row["diagnostic_threshold"]) == locked_threshold)
        record["locked_validation_threshold"] = locked_threshold
        record["locked_images_with_any_candidate"] = int(locked_sweep["images_with_any_candidate"])
        record["one_sided_95_binomial_upper_for_image_candidate_probability"] = 1.0 - 0.05 ** (1.0 / len(chosen))
        seed_rows.append(record)

    write_tsv(output / "checkpoint_audit.tsv", checkpoint_rows)
    write_tsv(output / "per_image_continuous_score_diagnostics.tsv", image_rows)
    write_tsv(output / "pixel_calibration_bins.tsv", calibration_rows)
    write_tsv(output / "threshold_sweep.tsv", sweep_rows)
    write_tsv(output / "per_seed_continuous_score_summary.tsv", seed_rows)
    summary = {
        "run_id": args.run_id,
        "status": "completed_post_lock_continuous_score_diagnostic",
        "analysis_class": "post_lock_exploratory_diagnostic",
        "locked_primary_result_changed": False,
        "external_threshold_selection_permitted": False,
        "threshold_grid": THRESHOLD_GRID,
        "calibration_bins": len(CALIBRATION_EDGES) - 1,
        "images": len(triplets),
        "seeds": locked.SEEDS,
        "one_sided_bound_scope": "binomial_reference_for_60_image_candidate_presence_only; not a farm-population or component-recall bound",
        "runtime_seconds": time.perf_counter() - started,
        "config_sha256": hashlib.sha256(args.config.resolve().read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
