#!/usr/bin/env python3
"""Nested proposal-setting selection plus fixed-budget reranking evaluation."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import (  # noqa: E402
    L2_GRID, average_precision, fit_logistic, predict, read_tsv, roc_auc, sha256,
)


DATES = ("2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15")
FEATURES = ("role_logit", "soil_probability_mean", "log1p_area_pixels", "mean_foreground_probability", "max_foreground_probability", "log_bbox_area", "log_bbox_aspect", "bbox_fill_fraction")
PREFIXES = (1, 5, 10, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("project-root", "archive", "manifest", "grid-root", "grid-summary", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-grid-summary-sha256", required=True)
    parser.add_argument("--maximum-mean-candidates-per-image", type=float, default=100.0)
    parser.add_argument("--minimum-instance-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-candidate-labeled-coverage", type=float, default=0.50)
    parser.add_argument("--minimum-role-purity", type=float, default=0.90)
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_truth(archive_path: Path, manifest: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    output = {}
    with zipfile.ZipFile(archive_path) as archive:
        for row in manifest:
            prefix = "weedsgalore-dataset/"
            semantic = np.asarray(Image.open(io.BytesIO(archive.read(prefix + row["semantic_relpath"]))))
            instances = np.asarray(Image.open(io.BytesIO(archive.read(prefix + row["instance_relpath"]))))
            records = {}
            for truth_id in (int(value) for value in np.unique(instances) if int(value) > 0):
                mask = instances == truth_id
                values = semantic[mask]
                crop, weed = int((values == 1).sum()), int((values > 1).sum())
                if crop + weed:
                    records[truth_id] = {"instance_id": f"{row['sample_id']}:instance:{truth_id}", "role": "crop" if crop >= weed else "weed", "area": int(mask.sum())}
            output[row["sample_id"]] = {"date": row["provisional_source_session_id"], "semantic": semantic, "instances": instances, "records": records}
    return output


def audit_setting(setting_path: Path, manifest: list[dict[str, str]], truth: dict[str, dict[str, object]], detailed: bool) -> dict[str, object]:
    candidates = read_tsv(setting_path / "candidate_components.tsv")
    semantic_scores = {row["candidate_id"]: row for row in read_tsv(setting_path / "candidate_semantic_scores.tsv")}
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates: by_file[row["file"]].append(row)
    component_archive = np.load(setting_path / "candidate_component_label_maps.npz")
    maps, sample_ids = component_archive["label_maps"], [str(value) for value in component_archive["sample_ids"]]
    if sample_ids != [row["sample_id"] for row in manifest]: raise ValueError(f"Map lineage mismatch: {setting_path.name}")
    totals = {date: {role: 0 for role in ("crop", "weed")} for date in DATES}
    recalled = {date: {role: 0 for role in ("crop", "weed")} for date in DATES}
    candidate_counts = Counter()
    rows: list[dict[str, object]] = []
    qualified: dict[str, set[str]] = defaultdict(set)
    instance_roles: dict[str, str] = {}
    for frame_index, sample_id in enumerate(sample_ids):
        current = truth[sample_id]; date = str(current["date"]); semantic = current["semantic"]; instances = current["instances"]
        labels = maps[frame_index]; local = {int(row["component_index"]): row for row in by_file[sample_id]}; candidate_counts[date] += len(local)
        max_label = int(labels.max())
        area = np.bincount(labels.ravel(), minlength=max_label + 1)
        crop_pixels = np.bincount(labels[semantic == 1].ravel(), minlength=max_label + 1)
        weed_pixels = np.bincount(labels[semantic > 1].ravel(), minlength=max_label + 1)
        candidate_contract = {}
        for label, candidate in local.items():
            labeled = int(crop_pixels[label] + weed_pixels[label]); coverage = labeled / max(1, int(area[label]))
            crop_purity, weed_purity = int(crop_pixels[label]) / max(1, labeled), int(weed_pixels[label]) / max(1, labeled)
            eligible = coverage >= 0.50 and max(crop_purity, weed_purity) >= 0.90
            target = ("crop" if crop_purity >= weed_purity else "weed") if eligible else "ambiguous_or_background"
            candidate_contract[label] = {"target": target, "coverage": coverage, "crop_purity": crop_purity, "weed_purity": weed_purity}
            if detailed:
                score = semantic_scores[candidate["candidate_id"]]; width = float(candidate["x1_exclusive"]) - float(candidate["x0"]); height = float(candidate["y1_exclusive"]) - float(candidate["y0"]); pixels = float(candidate["area_pixels"]); eps = 1e-8
                rows.append({"candidate_id": candidate["candidate_id"], "date": date, "file": sample_id, "truth": target, "eligible": eligible, "role_logit": math.log((float(score["weed_probability_mean"]) + eps) / (float(score["crop_probability_mean"]) + eps)), "soil_probability_mean": float(score["soil_probability_mean"]), "log1p_area_pixels": math.log1p(pixels), "mean_foreground_probability": float(candidate["mean_foreground_probability"]), "max_foreground_probability": float(candidate["max_foreground_probability"]), "log_bbox_area": math.log(max(1.0, width * height)), "log_bbox_aspect": math.log(max(width, 1.0) / max(height, 1.0)), "bbox_fill_fraction": pixels / max(1.0, width * height)})
        foreground = (instances > 0) & (labels > 0)
        overlaps = {}
        if foreground.any():
            pairs, counts = np.unique(np.column_stack([instances[foreground], labels[foreground]]), axis=0, return_counts=True)
            overlaps = {(int(pair[0]), int(pair[1])): int(count) for pair, count in zip(pairs, counts, strict=True)}
        for truth_id, instance in current["records"].items():
            role, iid = instance["role"], instance["instance_id"]; totals[date][role] += 1; instance_roles[iid] = role
            found = False
            for label, candidate in local.items():
                overlap = overlaps.get((truth_id, label), 0)
                if not overlap: continue
                contract = candidate_contract[label]
                purity = contract["crop_purity"] if role == "crop" else contract["weed_purity"]
                is_qualified = overlap / instance["area"] >= 0.50 and contract["coverage"] >= 0.50 and purity >= 0.90
                if is_qualified:
                    found = True
                    if detailed: qualified[candidate["candidate_id"]].add(iid)
            if found: recalled[date][role] += 1
    metrics = {date: {role: {"truth_instances": totals[date][role], "recalled_instances": recalled[date][role], "proposal_recall": recalled[date][role] / max(1, totals[date][role])} for role in ("crop", "weed")} | {"images": sum(row["provisional_source_session_id"] == date for row in manifest), "candidates": candidate_counts[date], "candidates_per_image": candidate_counts[date] / max(1, sum(row["provisional_source_session_id"] == date for row in manifest))} for date in DATES}
    return {"metrics": metrics, "rows": rows if detailed else None, "qualified": qualified if detailed else None, "instance_roles": instance_roles if detailed else None, "candidates": len(candidates)}


def matrix(rows: list[dict[str, object]], dates: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    selected = [row for row in rows if row["eligible"] and row["date"] in dates]
    return np.asarray([[float(row[name]) for name in FEATURES] for row in selected]), np.asarray([row["truth"] == "weed" for row in selected], dtype=np.int64), selected


def nested_fit(rows: list[dict[str, object]], training_dates: tuple[str, ...], held_out: str) -> tuple[dict[str, float], dict[str, object]]:
    sensitivity = []
    for l2 in L2_GRID:
        values = {}
        for validation in training_dates:
            train_dates = tuple(date for date in training_dates if date != validation)
            train_x, train_y, _ = matrix(rows, train_dates); val_x, val_y, _ = matrix(rows, (validation,))
            values[validation] = float(roc_auc(predict(fit_logistic(train_x, train_y, float(l2)), val_x), val_y))
        sensitivity.append({"l2_penalty": float(l2), "mean_inner_date_auroc": float(np.mean(list(values.values()))), "per_date": values})
    chosen = max(sensitivity, key=lambda item: (item["mean_inner_date_auroc"], -abs(math.log10(item["l2_penalty"]))))
    train_x, train_y, _ = matrix(rows, training_dates); model = fit_logistic(train_x, train_y, float(chosen["l2_penalty"]))
    test = [row for row in rows if row["date"] == held_out]; test_x = np.asarray([[float(row[name]) for name in FEATURES] for row in test])
    scores = {row["candidate_id"]: float(score) for row, score in zip(test, predict(model, test_x), strict=True)}
    eligible = [row for row in test if row["eligible"]]; labels = np.asarray([row["truth"] == "weed" for row in eligible], dtype=np.int64); vals = np.asarray([scores[row["candidate_id"]] for row in eligible])
    return scores, {"selected_l2_penalty": chosen["l2_penalty"], "l2_sensitivity": sensitivity, "auroc": float(roc_auc(vals, labels)), "average_precision": float(average_precision(vals, labels, [row["candidate_id"] for row in eligible])), "eligible_candidates": len(eligible)}


def queue_metrics(rows: list[dict[str, object]], scores: dict[str, float], qualified: dict[str, set[str]], instance_roles: dict[str, str], held_out: str, k: int) -> dict[str, object]:
    by_file: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["date"] == held_out: by_file[str(row["file"])].append(row)
    accepted = []
    for file_rows in by_file.values(): accepted.extend(sorted(file_rows, key=lambda row: (-scores[row["candidate_id"]], row["candidate_id"]))[:k])
    counts = Counter(row["truth"] for row in accepted); recalled = {iid for row in accepted for iid in qualified.get(row["candidate_id"], set())}
    weed_total = sum(role == "weed" and iid.startswith(tuple(by_file)) for iid, role in instance_roles.items())
    # Prefix matching by file is explicit because instance identifiers start with the sample ID.
    weed_ids = {iid for iid, role in instance_roles.items() if role == "weed" and any(iid.startswith(file + ":instance:") for file in by_file)}
    crop_scenes = {row["file"] for row in accepted for iid in qualified.get(row["candidate_id"], set()) if instance_roles[iid] == "crop"}
    return {"k": k, "images": len(by_file), "accepted_candidates": len(accepted), "precision_all_candidates": counts["weed"] / max(1, len(accepted)), "accepted_weed_candidates": counts["weed"], "accepted_crop_candidates": counts["crop"], "accepted_ambiguous_background_candidates": counts["ambiguous_or_background"], "weed_instance_recall": len(recalled & weed_ids) / max(1, len(weed_ids)), "recalled_weed_instances": len(recalled & weed_ids), "truth_weed_instances": len(weed_ids), "crop_false_scene_risk": len(crop_scenes) / max(1, len(by_file))}


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""): raise RuntimeError("This experiment is CPU-only")
    root, output_dir = args.project_root.resolve(), args.output_dir.resolve()
    if output_dir.exists(): raise FileExistsError(output_dir)
    if sha256(args.archive.resolve()) != args.expected_archive_sha256 or sha256(args.manifest.resolve()) != args.expected_manifest_sha256 or sha256(args.grid_summary.resolve()) != args.expected_grid_summary_sha256: raise ValueError("Input hash mismatch")
    started = time.perf_counter(); manifest = read_tsv(args.manifest.resolve()); truth = load_truth(args.archive.resolve(), manifest); grid = json.loads(args.grid_summary.read_text(encoding="utf-8"))
    setting_records = {record["setting"]: record for record in grid["settings"] if record["return_code"] == 0}
    audits = {}; setting_rows = []
    for setting, record in setting_records.items():
        audit = audit_setting(args.grid_root.resolve() / setting, manifest, truth, False); audits[setting] = audit
        for date in DATES:
            setting_rows.append({"setting": setting, "threshold": record["threshold"], "minimum_area": record["minimum_area"], "date": date, "images": audit["metrics"][date]["images"], "candidates": audit["metrics"][date]["candidates"], "candidates_per_image": audit["metrics"][date]["candidates_per_image"], "crop_instances": audit["metrics"][date]["crop"]["truth_instances"], "crop_proposal_recall": audit["metrics"][date]["crop"]["proposal_recall"], "weed_instances": audit["metrics"][date]["weed"]["truth_instances"], "weed_proposal_recall": audit["metrics"][date]["weed"]["proposal_recall"]})
    selected = {}
    for held_out in DATES:
        training = tuple(date for date in DATES if date != held_out); candidates = []
        for setting, record in setting_records.items():
            recall = float(np.mean([audits[setting]["metrics"][date]["weed"]["proposal_recall"] for date in training])); burden = sum(audits[setting]["metrics"][date]["candidates"] for date in training) / sum(audits[setting]["metrics"][date]["images"] for date in training)
            candidates.append({"setting": setting, "training_macro_weed_proposal_recall": recall, "training_candidates_per_image": burden, "threshold": record["threshold"], "minimum_area": record["minimum_area"]})
        feasible = [item for item in candidates if item["training_candidates_per_image"] <= args.maximum_mean_candidates_per_image]
        selected[held_out] = max(feasible, key=lambda item: (item["training_macro_weed_proposal_recall"], -item["training_candidates_per_image"], item["threshold"], item["minimum_area"]))
        selected[held_out]["training_dates"] = list(training)
    detailed = {setting: audit_setting(args.grid_root.resolve() / setting, manifest, truth, True) for setting in sorted({item["setting"] for item in selected.values()})}
    outer_rows = []; prediction_rows = []
    for held_out in DATES:
        choice = selected[held_out]; setting = choice["setting"]; data = detailed[setting]; training = tuple(date for date in DATES if date != held_out)
        scores, ranking = nested_fit(data["rows"], training, held_out)
        held_metrics = data["metrics"][held_out]
        base = {"held_out_date": held_out, "selected_setting": setting, "threshold": choice["threshold"], "minimum_area": choice["minimum_area"], "training_macro_weed_proposal_recall": choice["training_macro_weed_proposal_recall"], "training_candidates_per_image": choice["training_candidates_per_image"], "held_out_weed_proposal_recall": held_metrics["weed"]["proposal_recall"], "held_out_crop_proposal_recall": held_metrics["crop"]["proposal_recall"], "held_out_candidates_per_image": held_metrics["candidates_per_image"], "role_auroc": ranking["auroc"], "role_average_precision": ranking["average_precision"], "selected_l2_penalty": ranking["selected_l2_penalty"]}
        for k in PREFIXES: outer_rows.append({**base, **{f"queue_{key}": value for key, value in queue_metrics(data["rows"], scores, data["qualified"], data["instance_roles"], held_out, k).items()}})
        for row in data["rows"]:
            if row["date"] == held_out: prediction_rows.append({"candidate_id": row["candidate_id"], "held_out_date": held_out, "selected_setting": setting, "file": row["file"], "dataset_role_target": row["truth"], "policy_scoring_eligible": int(row["eligible"]), "role_plus_local_probability": f"{scores[row['candidate_id']]:.10f}"})
        choice["held_out"] = base; choice["ranking_artifact"] = ranking
    output_dir.mkdir(parents=True); setting_path, outer_path, prediction_path = output_dir / "proposal_setting_metrics.tsv", output_dir / "outer_commissioning_results.tsv", output_dir / "outer_predictions.tsv"
    write_tsv(setting_path, setting_rows); write_tsv(outer_path, outer_rows); write_tsv(prediction_path, prediction_rows)
    macro_by_k = {str(k): {metric: float(np.mean([row[metric] for row in outer_rows if int(row["queue_k"]) == k])) for metric in ("held_out_weed_proposal_recall", "held_out_candidates_per_image", "role_auroc", "role_average_precision", "queue_precision_all_candidates", "queue_weed_instance_recall", "queue_crop_false_scene_risk")} for k in PREFIXES}
    summary = {"run_id": args.run_id, "status": "completed_posthoc_nested_proposal_commissioning", "scope": {"public_data_only": True, "sealed_test_read": False, "proposal_generation_truth_free": True, "outer_date_not_used_for_setting_model_or_regularization_selection": True, "formal_risk_certificate": False}, "inputs": {"archive_sha256": sha256(args.archive.resolve()), "manifest_sha256": sha256(args.manifest.resolve()), "grid_summary_sha256": sha256(args.grid_summary.resolve())}, "selection_contract": {"objective": "maximum training-date macro weed-instance proposal recall", "maximum_mean_training_candidates_per_image": args.maximum_mean_candidates_per_image, "tie_break": "fewer candidates, then higher threshold, then larger minimum area"}, "matching_contract": {"minimum_instance_coverage": args.minimum_instance_coverage, "minimum_candidate_labeled_coverage": args.minimum_candidate_labeled_coverage, "minimum_same_role_purity": args.minimum_role_purity}, "per_outer_date": selected, "macro_by_fixed_review_budget": macro_by_k, "outputs": {"setting_metrics_sha256": sha256(setting_path), "outer_results_sha256": sha256(outer_path), "outer_predictions_sha256": sha256(prediction_path)}, "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version, "numpy": np.__version__}, "limitations": ["The threshold grid and selection analysis are post-hoc Major Revision development work on four dates from one public source.", "Candidate-budget constraints and fixed review prefixes do not confer risk guarantees.", "Instance truth is used only for training-date setting selection and held-out offline scoring, not proposal generation or held-out fitting.", "Expert visual support and operational review time remain unmeasured."], "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(args.config.resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
