#!/usr/bin/env python3
"""Shared-ranker-weight original-versus-commissioned proposal queue comparison."""
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

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_commissioning_contract_audit import audit_setting, load_truth  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_proposal_audit import official_split_map  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_queue import C_GRID, ROLE_LOCAL_FEATURES, fit_model, metric_record, predict, selected_by_score  # noqa: E402
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import read_tsv, sha256  # noqa: E402

KS = (1, 5, 10, 20, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("project-root", "archive", "manifest", "original-root", "commissioned-root", "proposal-summary", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    for name in ("archive", "manifest", "original-candidates", "commissioned-candidates", "proposal-summary"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def matrix(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in ROLE_LOCAL_FEATURES] for row in rows], dtype=np.float64)


def burden(variant: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for split in ("train", "val", "test"):
        current = [row for row in rows if row["official_split"] == split]
        by_file: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in current: by_file[str(row["file"])].append(row)
        counts = np.asarray([len(value) for value in by_file.values()])
        roles = Counter(str(row["truth"]) for row in current)
        output.append({
            "variant": variant, "official_split": split, "images": len(by_file), "candidates": len(current),
            "candidates_per_image_mean": float(counts.mean()), "candidates_per_image_median": float(np.median(counts)),
            "candidates_per_image_q25": float(np.quantile(counts, 0.25)), "candidates_per_image_q75": float(np.quantile(counts, 0.75)),
            "candidates_per_image_max": int(counts.max()),
            "images_with_fewer_than_20_candidates": int((counts < 20).sum()), "fraction_images_with_fewer_than_20_candidates": float((counts < 20).mean()),
            "eligible_weed_candidates": roles["weed"], "eligible_crop_candidates": roles["crop"],
            "ambiguous_or_background_candidates": roles["ambiguous_or_background"],
            "eligible_weed_fraction": roles["weed"] / max(1, len(current)), "eligible_crop_fraction": roles["crop"] / max(1, len(current)),
            "ambiguous_or_background_fraction": roles["ambiguous_or_background"] / max(1, len(current)),
        })
    return output


def main() -> None:
    args = parse_args(); started = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""): raise RuntimeError("CPU-only experiment")
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    if output.exists(): raise FileExistsError(output)
    checks = (
        (args.archive, args.expected_archive_sha256), (args.manifest, args.expected_manifest_sha256),
        (args.original_root / "candidate_components.tsv", args.expected_original_candidates_sha256),
        (args.commissioned_root / "candidate_components.tsv", args.expected_commissioned_candidates_sha256),
        (args.proposal_summary, args.expected_proposal_summary_sha256),
    )
    for path, expected in checks:
        if sha256(path.resolve()) != expected: raise ValueError(f"Input hash mismatch: {path}")
    proposal_summary = json.loads(args.proposal_summary.read_text(encoding="utf-8"))
    if proposal_summary["setting_selection"]["selected_setting"] != "T070_A032": raise ValueError("Commissioned setting mismatch")
    manifest = read_tsv(args.manifest.resolve()); split_map = official_split_map(args.archive.resolve(), manifest); truth = load_truth(args.archive.resolve(), manifest)
    audits = {
        "original_T095_A032": audit_setting(args.original_root.resolve(), manifest, truth),
        "commissioned_T070_A032": audit_setting(args.commissioned_root.resolve(), manifest, truth),
    }
    for audit in audits.values():
        for row in audit["rows"]: row["official_split"] = split_map[str(row["file"])]

    # Fit one weight vector on commissioned training data, tune C on commissioned validation,
    # refit on commissioned train+validation, and apply it unchanged to both candidate pools.
    fit_rows = audits["commissioned_T070_A032"]["rows"]; x = matrix(fit_rows)
    y = np.asarray([row["truth"] == "weed" for row in fit_rows], dtype=np.int64)
    train = [index for index, row in enumerate(fit_rows) if row["official_split"] == "train"]
    validation = [index for index, row in enumerate(fit_rows) if row["official_split"] == "val"]
    sensitivity = []
    for c_value in C_GRID:
        candidate_model = fit_model(x[train], y[train], c_value); values = predict(candidate_model, x[validation])
        sensitivity.append({"c": c_value, "validation_all_candidate_roc_auc": float(roc_auc_score(y[validation], values)), "validation_all_candidate_average_precision": float(average_precision_score(y[validation], values))})
    selected = max(sensitivity, key=lambda row: (row["validation_all_candidate_roc_auc"], row["validation_all_candidate_average_precision"], -abs(np.log10(row["c"]))))
    fitted = fit_model(x[train + validation], y[train + validation], float(selected["c"]))

    burden_rows, ranking_rows, queue_rows, prediction_rows = [], [], [], []
    for variant, audit in audits.items():
        rows = audit["rows"]; scores = predict(fitted, matrix(rows)); burden_rows.extend(burden(variant, rows))
        test_indices = [index for index, row in enumerate(rows) if row["official_split"] == "test"]
        test_y = np.asarray([rows[index]["truth"] == "weed" for index in test_indices], dtype=np.int64)
        ranking_rows.append({"variant": variant, "ranker_weights": "identical commissioned-train-plus-validation fitted weights", "test_candidates": len(test_indices), "test_weed_candidates": int(test_y.sum()), "test_nonweed_candidates": int((1 - test_y).sum()), "test_all_candidate_roc_auc": float(roc_auc_score(test_y, scores[test_indices])), "test_all_candidate_average_precision": float(average_precision_score(test_y, scores[test_indices]))})
        test_instances = [row for row in audit["instances"] if split_map[str(row["sample_id"])] == "test"]
        for k in KS:
            chosen = selected_by_score(rows, scores, test_indices, k)
            queue_rows.append({"variant": variant, "ranker_weights": "identical", "k": k, **metric_record(rows, chosen, test_instances, audit["spatial"], audit["qualified"], "pooled_test", "ALL")})
            for date in ("2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15"):
                date_indices = [index for index in test_indices if rows[index]["date"] == date]
                date_instances = [row for row in test_instances if row["date"] == date]
                date_chosen = selected_by_score(rows, scores, date_indices, k)
                queue_rows.append({"variant": variant, "ranker_weights": "identical", "k": k, **metric_record(rows, date_chosen, date_instances, audit["spatial"], audit["qualified"], "acquisition_date", date)})
        for index in test_indices:
            prediction_rows.append({"variant": variant, "candidate_id": rows[index]["candidate_id"], "file": rows[index]["file"], "acquisition_date": rows[index]["date"], "dataset_role_target": rows[index]["truth"], "score": f"{float(scores[index]):.12f}"})

    output.mkdir(parents=True)
    files = {"burden": output / "proposal_burden_and_composition.tsv", "ranking": output / "matched_weight_ranking.tsv", "queue": output / "matched_weight_queue_curves.tsv", "predictions": output / "matched_weight_test_predictions.tsv"}
    for key, values in (("burden", burden_rows), ("ranking", ranking_rows), ("queue", queue_rows), ("predictions", prediction_rows)): write_tsv(files[key], values)
    summary = {
        "run_id": args.run_id, "status": "completed_official_spatial_matched_weight_proposal_queue_comparison",
        "scope": {"public_data_only": True, "sealed_test_read": False, "official_train_fit": True, "official_validation_regularization_selection": True, "official_test_offline_scoring_only": True, "identical_ranker_weights_across_proposal_pools": True},
        "proposal_definitions": {"original_T095_A032": {"foreground_threshold": 0.95, "minimum_area_pixels": 32, "selection_source": "fixed in the target-truth-free external-inference configuration; not optimized on WeedsGalore truth"}, "commissioned_T070_A032": {"foreground_threshold": 0.70, "minimum_area_pixels": 32, "selection_source": "official training split only; maximize role-qualified weed proposal recall under <=100 mean candidates/image"}},
        "ranker": {"features": list(ROLE_LOCAL_FEATURES), "target": "all-candidate weed versus rest", "class_weight": "balanced", "selected_c": selected["c"], "validation_sensitivity": sensitivity, "fit_pool": "commissioned official train+validation", "comparison": "same scaler and coefficient vector applied without refitting"},
        "primary_k20": {variant: next(row for row in queue_rows if row["variant"] == variant and int(row["k"]) == 20 and row["aggregation"] == "pooled_test") for variant in audits},
        "inputs": {"archive_sha256": sha256(args.archive.resolve()), "manifest_sha256": sha256(args.manifest.resolve()), "original_candidates_sha256": sha256(args.original_root / "candidate_components.tsv"), "commissioned_candidates_sha256": sha256(args.commissioned_root / "candidate_components.tsv"), "proposal_summary_sha256": sha256(args.proposal_summary.resolve())},
        "outputs": {key + "_sha256": sha256(path) for key, path in files.items()},
        "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version, "numpy": np.__version__, "scikit_learn": __import__("sklearn").__version__},
        "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(args.config.resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
