#!/usr/bin/env python3
"""Queued-recall sensitivity to matching, IoU, and multiplicity contracts."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_commissioning_contract_audit import audit_setting, load_truth  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_proposal_audit import official_split_map  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_queue import matching_count  # noqa: E402
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import read_tsv, sha256  # noqa: E402

MODELS = ("all_candidate_role_plus_local_logistic", "all_candidate_frozen_dinov2_linear", "all_candidate_component_masked_dinov2_linear")
KS = (1, 5, 10, 20, 50)
GRID = (0.25, 0.50, 0.75)
PURITY = (0.50, 0.75, 0.90)
IOU_GRID = (0.25, 0.50, 0.75)
MULTIPLICITY_GRID = (0.10, 0.25, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("project-root", "archive", "manifest", "candidate-root", "predictions", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    for name in ("archive", "manifest", "candidates", "predictions"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def select_ids(predictions: list[dict[str, str]], model: str, files: set[str], k: int) -> list[str]:
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        if row["model"] == model and row["file"] in files: by_file[row["file"]].append(row)
    selected = []
    for file_name in sorted(by_file):
        ranked = sorted(by_file[file_name], key=lambda row: (-float(row["score"]), row["candidate_id"]))
        selected.extend(row["candidate_id"] for row in ranked[:k])
    return selected


def evaluate(selected: list[str], edges: dict[str, set[str]], weed_ids: set[str]) -> dict[str, object]:
    recalled = {instance for candidate in selected for instance in edges.get(candidate, set())}
    one_to_one = matching_count(selected, edges, weed_ids)
    return {"selected_candidates": len(selected), "weed_truth_instances": len(weed_ids), "many_to_one_recalled_instances": len(recalled & weed_ids), "many_to_one_queued_recall": len(recalled & weed_ids) / max(1, len(weed_ids)), "one_to_one_matches": one_to_one, "one_to_one_queued_recall": one_to_one / max(1, len(weed_ids))}


def main() -> None:
    args = parse_args(); started = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""): raise RuntimeError("CPU-only analysis")
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    if output.exists(): raise FileExistsError(output)
    for path, expected in ((args.archive, args.expected_archive_sha256), (args.manifest, args.expected_manifest_sha256), (args.candidate_root / "candidate_components.tsv", args.expected_candidates_sha256), (args.predictions, args.expected_predictions_sha256)):
        if sha256(path.resolve()) != expected: raise ValueError(f"Input hash mismatch: {path}")
    manifest = read_tsv(args.manifest.resolve()); split_map = official_split_map(args.archive.resolve(), manifest); truth = load_truth(args.archive.resolve(), manifest); audit = audit_setting(args.candidate_root.resolve(), manifest, truth)
    predictions = read_tsv(args.predictions.resolve())
    test_instances = [row for row in audit["instances"] if split_map[str(row["sample_id"])] == "test" and row["truth_role"] == "weed"]
    test_matches = [row for row in audit["matches"] if split_map[str(row["sample_id"])] == "test" and row["truth_role"] == "weed"]
    files_by_stratum = {"ALL": {str(row["sample_id"]) for row in test_instances}}
    for date in ("2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15"):
        files_by_stratum[date] = {str(row["sample_id"]) for row in test_instances if row["date"] == date}
    sensitivity_rows, iou_rows = [], []
    for model in MODELS:
        for k in KS:
            for stratum, files in files_by_stratum.items():
                weed_ids = {str(row["instance_id"]) for row in test_instances if str(row["sample_id"]) in files}
                selected = select_ids(predictions, model, files, k)
                for instance_coverage in GRID:
                    for labeled_coverage in GRID:
                        for purity in PURITY:
                            edges: dict[str, set[str]] = defaultdict(set)
                            for row in test_matches:
                                if row["sample_id"] in files and float(row["instance_coverage"]) >= instance_coverage and float(row["candidate_labeled_coverage"]) >= labeled_coverage and float(row["same_role_purity"]) >= purity:
                                    edges[str(row["candidate_id"])].add(str(row["instance_id"]))
                            sensitivity_rows.append({"model": model, "k": k, "stratum": stratum, "minimum_instance_coverage": instance_coverage, "minimum_candidate_labeled_coverage": labeled_coverage, "minimum_same_role_purity": purity, **evaluate(selected, edges, weed_ids)})
                for iou_threshold in IOU_GRID:
                    edges = defaultdict(set)
                    for row in test_matches:
                        if row["sample_id"] in files and float(row["iou"]) >= iou_threshold:
                            edges[str(row["candidate_id"])].add(str(row["instance_id"]))
                    iou_rows.append({"model": model, "k": k, "stratum": stratum, "minimum_iou": iou_threshold, **evaluate(selected, edges, weed_ids)})

    # Candidate-level multiplicity for the primary masked DINOv2 K=20 queue.
    primary_ids = set(select_ids(predictions, "all_candidate_component_masked_dinov2_linear", files_by_stratum["ALL"], 20))
    matches_by_candidate: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in test_matches:
        if row["candidate_id"] in primary_ids: matches_by_candidate[str(row["candidate_id"])].append(row)
    multiplicity_rows = []
    for threshold in MULTIPLICITY_GRID:
        counts = Counter()
        candidate_rows = []
        for candidate_id in sorted(primary_ids):
            covered = {str(row["instance_id"]) for row in matches_by_candidate.get(candidate_id, []) if float(row["instance_coverage"]) >= threshold}
            counts[len(covered)] += 1
            candidate_rows.append({"candidate_id": candidate_id, "minimum_instance_coverage": threshold, "covered_weed_instances": len(covered)})
        for covered_count in sorted(counts):
            multiplicity_rows.append({"model": "all_candidate_component_masked_dinov2_linear", "k": 20, "minimum_instance_coverage": threshold, "covered_weed_instances_per_candidate": covered_count, "queued_candidates": counts[covered_count], "queued_candidate_fraction": counts[covered_count] / max(1, len(primary_ids))})

    output.mkdir(parents=True)
    files = {"contract": output / "queued_matching_contract_sensitivity.tsv", "iou": output / "queued_iou_sensitivity.tsv", "multiplicity": output / "queued_candidate_weed_instance_multiplicity.tsv"}
    for key, rows in (("contract", sensitivity_rows), ("iou", iou_rows), ("multiplicity", multiplicity_rows)): write_tsv(files[key], rows)
    primary = [row for row in sensitivity_rows if row["model"] == "all_candidate_component_masked_dinov2_linear" and int(row["k"]) == 20 and row["stratum"] == "ALL"]
    summary = {"run_id": args.run_id, "status": "completed_official_queue_contract_sensitivity", "scope": {"public_data_only": True, "sealed_test_read": False, "offline_test_scoring_only": True, "prediction_scores_refit": False}, "primary_masked_dino_k20_range": {"many_to_one_queued_recall_min": min(float(row["many_to_one_queued_recall"]) for row in primary), "many_to_one_queued_recall_max": max(float(row["many_to_one_queued_recall"]) for row in primary), "one_to_one_queued_recall_min": min(float(row["one_to_one_queued_recall"]) for row in primary), "one_to_one_queued_recall_max": max(float(row["one_to_one_queued_recall"]) for row in primary)}, "definitions": {"many_to_one": "one queued candidate may recover multiple truth weed instances", "one_to_one": "maximum bipartite matching permits each candidate and truth instance at most one match", "iou": "candidate-component versus truth-instance pixel IoU"}, "inputs": {"archive_sha256": sha256(args.archive.resolve()), "manifest_sha256": sha256(args.manifest.resolve()), "candidates_sha256": sha256(args.candidate_root / "candidate_components.tsv"), "predictions_sha256": sha256(args.predictions.resolve())}, "outputs": {key + "_sha256": sha256(path) for key, path in files.items()}, "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version}, "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(args.config.resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
