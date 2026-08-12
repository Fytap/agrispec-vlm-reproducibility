#!/usr/bin/env python3
"""Validation-selected semantic score and entropy queue diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import train_evaluate_p2_weedsgalore_target_semantic_seed_stability_h200 as common
except ModuleNotFoundError:
    backup_scripts = (
        Path(__file__).resolve().parents[1]
        / "reproducibility_backups"
        / "H200_seed_stability_20260812_v3"
        / "scripts"
    )
    sys.path.insert(0, str(backup_scripts))
    import train_evaluate_p2_weedsgalore_target_semantic_seed_stability_h200 as common


SEEDS = (20260810, 20260811, 20260812, 20260813, 20260814)
POLICIES = ("conditional_weed_probability", "mean_predictive_entropy", "mean_weed_probability")
RUN_TEMPLATE = "P2_WEEDSGALORE_TARGET_SEMANTIC_H200_SEED_{seed}_20260812_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--physical-gpu-id", type=int, choices=(6, 7), required=True)
    return parser.parse_args()


def add_policy_scores(candidates: list[dict[str, Any]], probabilities: np.ndarray, labels: np.ndarray) -> None:
    for candidate in candidates:
        component = int(str(candidate["candidate_id"]).rsplit(":C", 1)[1])
        mask = labels == component
        class_probabilities = probabilities[:, mask]
        candidate["conditional_weed_probability"] = float(candidate["score"])
        candidate["mean_weed_probability"] = float(probabilities[2][mask].mean())
        candidate["mean_predictive_entropy"] = float(
            (-class_probabilities * np.log(np.clip(class_probabilities, 1e-8, 1.0))).sum(axis=0).mean()
        )


def evaluate(
    probabilities: dict[str, np.ndarray],
    dataset_root: Path,
    threshold: float,
    minimum_area: int,
    policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    totals = Counter()
    tile_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for stem in sorted(probabilities):
        probs = probabilities[stem]
        candidates, spatial_edges, qualified_edges, _, _ = common.component_candidates(
            stem, probs, dataset_root, threshold, minimum_area
        )
        plant_probability = 1.0 - probs[0]
        labels, _ = ndimage.label(plant_probability >= threshold, structure=np.ones((3, 3), dtype=np.uint8))
        add_policy_scores(candidates, probs, labels)
        ranked = sorted(candidates, key=lambda row: (-float(row[policy]), str(row["candidate_id"])))
        accepted = ranked[: common.QUEUE_K]
        _, _, instances = common.truth_instances(stem, dataset_root)
        weed_ids = {record.instance_id for record in instances if record.role == "weed"}
        spatial = set().union(*(spatial_edges.get(str(row["candidate_id"]), set()) for row in accepted)) if accepted else set()
        qualified = set().union(*(qualified_edges.get(str(row["candidate_id"]), set()) for row in accepted)) if accepted else set()
        one_to_one = common.one_to_one_count(accepted, qualified_edges)
        eligible_weed = sum(str(row["candidate_truth"]) == "weed" for row in accepted)
        crop_overlap = int(any(float(row["crop_fraction_candidate"]) > 0 for row in accepted))
        tile = {
            "file": stem,
            "date": stem[:10],
            "weed_instances": len(weed_ids),
            "candidates": len(ranked),
            "accepted_candidates": len(accepted),
            "eligible_weed_candidates": eligible_weed,
            "many_to_one_spatial_recalled": len(spatial & weed_ids),
            "many_to_one_role_qualified_recalled": len(qualified & weed_ids),
            "one_to_one_role_qualified_recalled": one_to_one,
            "crop_overlap_scene": crop_overlap,
        }
        tile_rows.append(tile)
        for key, value in tile.items():
            if key not in ("file", "date"):
                totals[key] += int(value)
        totals["images"] += 1
        accepted_ids = {str(row["candidate_id"]) for row in accepted}
        rank_by_id = {str(row["candidate_id"]): rank for rank, row in enumerate(ranked, 1)}
        for row in ranked:
            candidate_rows.append({
                "file": stem,
                "candidate_id": row["candidate_id"],
                "candidate_truth": row["candidate_truth"],
                "policy": policy,
                "policy_score": row[policy],
                "conditional_weed_probability": row["conditional_weed_probability"],
                "mean_weed_probability": row["mean_weed_probability"],
                "mean_predictive_entropy": row["mean_predictive_entropy"],
                "rank_within_image": rank_by_id[str(row["candidate_id"])],
                "accepted_K20": int(str(row["candidate_id"]) in accepted_ids),
            })
    labels = np.asarray([row["candidate_truth"] == "weed" for row in candidate_rows], dtype=np.int64)
    scores = np.asarray([float(row["policy_score"]) for row in candidate_rows], dtype=np.float64)
    summary = {
        "images": totals["images"],
        "weed_instances": totals["weed_instances"],
        "candidates": totals["candidates"],
        "accepted_candidates": totals["accepted_candidates"],
        "all_candidate_precision": totals["eligible_weed_candidates"] / max(1, totals["accepted_candidates"]),
        "many_to_one_spatial_recall": totals["many_to_one_spatial_recalled"] / max(1, totals["weed_instances"]),
        "many_to_one_role_qualified_recall": totals["many_to_one_role_qualified_recalled"] / max(1, totals["weed_instances"]),
        "one_to_one_role_qualified_recall": totals["one_to_one_role_qualified_recalled"] / max(1, totals["weed_instances"]),
        "crop_overlap_scene_frequency": totals["crop_overlap_scene"] / max(1, totals["images"]),
        "candidate_AUC_diagnostic": float(roc_auc_score(labels, scores)),
        "candidate_AP_diagnostic": float(average_precision_score(labels, scores)),
    }
    return summary, tile_rows, candidate_rows


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise RuntimeError("Frozen physical GPU assignment mismatch")
    import torch

    root = args.project_root.resolve()
    archive = args.archive.resolve()
    cache = args.cache_dir.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    config = args.config.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if common.sha256(archive) != args.expected_archive_sha256:
        raise ValueError("Frozen public archive hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    dataset_root = common.extract_public_archive(archive, cache, args.expected_archive_sha256)
    validation_stems = common.read_split(dataset_root, "val")
    test_stems = common.read_split(dataset_root, "test")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    selected_test_tiles: list[dict[str, Any]] = []
    selected_test_candidates: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = checkpoint_root / RUN_TEMPLATE.format(seed=seed)
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        checkpoint_path = run_dir / "best_validation_queue.pt"
        if common.sha256(checkpoint_path) != summary["model"]["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for seed {seed}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        selection = checkpoint["selection"]
        threshold = float(selection["threshold"])
        minimum_area = int(selection["minimum_area_pixels"])
        model = common.build_model()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        validation_probabilities, validation_seconds, validation_peak = common.predict(
            model, validation_stems, dataset_root, device
        )
        test_probabilities, test_seconds, test_peak = common.predict(model, test_stems, dataset_root, device)
        per_policy_validation: list[dict[str, Any]] = []
        per_policy_test: dict[str, tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = {}
        for policy in POLICIES:
            validation_summary, _, _ = evaluate(
                validation_probabilities, dataset_root, threshold, minimum_area, policy
            )
            row = {"seed": seed, "policy": policy, **validation_summary}
            validation_rows.append(row)
            per_policy_validation.append(row)
            test_result = evaluate(test_probabilities, dataset_root, threshold, minimum_area, policy)
            per_policy_test[policy] = test_result
            test_rows.append({"seed": seed, "policy": policy, **test_result[0]})
        chosen = sorted(
            per_policy_validation,
            key=lambda row: (
                -float(row["one_to_one_role_qualified_recall"]),
                -float(row["all_candidate_precision"]),
                str(row["policy"]),
            ),
        )[0]
        chosen_policy = str(chosen["policy"])
        chosen_test_summary, chosen_tiles, chosen_candidates = per_policy_test[chosen_policy]
        for row in chosen_tiles:
            selected_test_tiles.append({"seed": seed, "selected_policy": chosen_policy, **row})
        for row in chosen_candidates:
            selected_test_candidates.append({"seed": seed, "selected_policy": chosen_policy, **row})
        checkpoint_rows.append({
            "seed": seed,
            "checkpoint_sha256": common.sha256(checkpoint_path),
            "validation_selected_epoch": int(selection["epoch"]),
            "validation_selected_component_threshold": threshold,
            "validation_selected_component_minimum_area": minimum_area,
            "validation_selected_ranking_policy": chosen_policy,
            "selected_validation_primary_recall": chosen["one_to_one_role_qualified_recall"],
            "selected_test_primary_recall": chosen_test_summary["one_to_one_role_qualified_recall"],
            "validation_inference_seconds": validation_seconds,
            "test_inference_seconds": test_seconds,
            "peak_gpu_memory_bytes": max(validation_peak, test_peak),
        })
        del model, checkpoint, validation_probabilities, test_probabilities
        torch.cuda.empty_cache()

    common.write_tsv(output / "checkpoint_and_policy_selection.tsv", checkpoint_rows)
    common.write_tsv(output / "validation_policy_metrics.tsv", validation_rows)
    common.write_tsv(output / "official_test_policy_metrics.tsv", test_rows)
    common.write_tsv(output / "selected_policy_test_tile_metrics.tsv", selected_test_tiles)
    common.write_tsv(output / "selected_policy_test_candidates.tsv", selected_test_candidates)
    metric_names = (
        "all_candidate_precision",
        "many_to_one_spatial_recall",
        "many_to_one_role_qualified_recall",
        "one_to_one_role_qualified_recall",
        "crop_overlap_scene_frequency",
        "candidate_AUC_diagnostic",
        "candidate_AP_diagnostic",
    )
    selected_test_by_seed = {
        seed: next(
            row for row in test_rows
            if int(row["seed"]) == seed
            and row["policy"] == next(item["validation_selected_ranking_policy"] for item in checkpoint_rows if int(item["seed"]) == seed)
        )
        for seed in SEEDS
    }
    aggregate = {
        metric: {
            "mean": float(np.mean([selected_test_by_seed[seed][metric] for seed in SEEDS])),
            "sample_sd": float(np.std([selected_test_by_seed[seed][metric] for seed in SEEDS], ddof=1)),
        }
        for metric in metric_names
    }
    summary = {
        "run_id": args.run_id,
        "status": "completed_validation_selected_semantic_ranking_diagnostics",
        "selection_rule": "validation primary recall, then precision, then lexicographic policy name",
        "selected_policies": {str(row["seed"]): row["validation_selected_ranking_policy"] for row in checkpoint_rows},
        "official_test_selected_policy_mean_sample_sd": aggregate,
        "inputs": {
            "archive_sha256": common.sha256(archive),
            "config_sha256": common.sha256(config),
            "script_sha256": common.sha256(Path(__file__).resolve()),
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "physical_gpu_id": args.physical_gpu_id,
            "gpu": torch.cuda.get_device_name(0),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "governance": {
            "analysis_class": "post_test_strong_baseline_extension",
            "ranking_policy_selected_on": "official_validation_only",
            "test_labels_for_selection": False,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output / "SHA256SUMS.tsv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("path\tsha256\n")
        for path in sorted(output.iterdir()):
            if path.name != "SHA256SUMS.tsv":
                stream.write(f"{path.name}\t{common.sha256(path)}\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
