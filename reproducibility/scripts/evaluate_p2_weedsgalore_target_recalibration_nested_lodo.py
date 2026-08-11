#!/usr/bin/env python3
"""Nested leave-one-date-out target-domain recalibration on WeedsGalore."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SESSIONS = ("2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15")
L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
FEATURE_GROUPS = {
    "orientation_recalibration": ("role_logit",),
    "local_specialist_recalibration": (
        "role_logit", "soil_probability_mean", "log1p_area_pixels",
        "mean_foreground_probability", "max_foreground_probability",
        "log_bbox_area", "log_bbox_aspect", "bbox_fill_fraction",
    ),
    "typed_context_recalibration": (
        "role_logit", "soil_probability_mean", "log1p_area_pixels",
        "mean_foreground_probability", "max_foreground_probability",
        "log_bbox_area", "log_bbox_aspect", "bbox_fill_fraction",
        "centroid_x_normalized", "centroid_y_normalized",
        "session_centered_role_logit", "session_robust_role_logit",
        "session_empirical_cdf",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def empirical_cdf(values: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    output: dict[str, float] = {}
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = ((cursor + end - 1) / 2) / max(1, len(ordered) - 1)
        for index in range(cursor, end):
            output[ordered[index][0]] = rank
        cursor = end
    return output


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    order = np.argsort(scores, kind="mergesort")
    values = scores[order]
    truth = labels[order]
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if not positives or not negatives:
        return None
    rank_sum = 0.0
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[end] == values[cursor]:
            end += 1
        rank_sum += ((cursor + 1 + end) / 2.0) * float(truth[cursor:end].sum())
        cursor = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(scores: np.ndarray, labels: np.ndarray, ids: list[str]) -> float | None:
    positives = int(labels.sum())
    if not positives:
        return None
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), ids[index]))
    found = 0
    total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            found += 1
            total += found / rank
    return total / positives


def fit_logistic(matrix: np.ndarray, labels: np.ndarray, l2: float) -> dict[str, object]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack([np.ones(len(matrix)), (matrix - mean) / scale])
    positives = max(1, int(labels.sum()))
    negatives = max(1, len(labels) - positives)
    sample_weight = np.where(labels == 1, len(labels) / (2 * positives), len(labels) / (2 * negatives))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.diag([0.0, *([l2] * matrix.shape[1])])
    iterations = 0
    for iterations in range(1, 101):
        probability = sigmoid(design @ coefficients)
        curvature = sample_weight * np.maximum(probability * (1.0 - probability), 1e-8)
        gradient = design.T @ (sample_weight * (probability - labels)) + penalty @ coefficients
        hessian = design.T @ (curvature[:, None] * design) + penalty + np.eye(design.shape[1]) * 1e-10
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return {"mean": mean, "scale": scale, "coefficients": coefficients, "iterations": iterations, "l2": l2, "training_rows": len(labels), "training_weed": int(labels.sum()), "training_crop": int(len(labels) - labels.sum())}


def predict(model: dict[str, object], matrix: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(matrix)), (matrix - model["mean"]) / model["scale"]])
    return sigmoid(design @ model["coefficients"])


def arrays(rows: list[dict[str, object]], sessions: tuple[str, ...], features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = [row for row in rows if row["eligible"] and row["session_id"] in sessions]
    return np.asarray([[row[name] for name in features] for row in selected], dtype=np.float64), np.asarray([row["truth"] == "weed" for row in selected], dtype=np.int64), [str(row["candidate_id"]) for row in selected]


def evaluate(rows: list[dict[str, object]], scores: dict[str, float]) -> dict[str, object]:
    eligible = [row for row in rows if row["eligible"]]
    values = np.asarray([scores[str(row["candidate_id"])] for row in eligible], dtype=np.float64)
    labels = np.asarray([row["truth"] == "weed" for row in eligible], dtype=np.int64)
    ids = [str(row["candidate_id"]) for row in eligible]
    predicted = values >= 0.5
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    per_session = {}
    for session in SESSIONS:
        indices = [index for index, row in enumerate(eligible) if row["session_id"] == session]
        session_scores = values[indices]
        session_labels = labels[indices]
        per_session[session] = {"rows": len(indices), "weed": int(session_labels.sum()), "crop": int(len(indices) - session_labels.sum()), "roc_auc": roc_auc(session_scores, session_labels), "average_precision": average_precision(session_scores, session_labels, [ids[index] for index in indices])}
    return {
        "eligible_rows": len(eligible), "weed_rows": int(labels.sum()), "crop_rows": int(len(labels) - labels.sum()),
        "pooled_roc_auc": roc_auc(values, labels), "pooled_average_precision": average_precision(values, labels, ids),
        "mean_provisional_session_roc_auc": sum(value["roc_auc"] for value in per_session.values()) / len(per_session),
        "mean_provisional_session_average_precision": sum(value["average_precision"] for value in per_session.values()) / len(per_session),
        "threshold_0p5": {"true_weed": tp, "false_crop": fp, "true_crop": tn, "missed_weed": fn, "precision": tp / max(1, tp + fp), "weed_recall": tp / max(1, tp + fn), "balanced_accuracy": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))},
        "per_provisional_session": per_session,
    }


def serial_model(model: dict[str, object], features: tuple[str, ...]) -> dict[str, object]:
    return {"feature_names": list(features), "standardization_mean": model["mean"].tolist(), "standardization_scale": model["scale"].tolist(), "coefficients_intercept_then_standardized_features": model["coefficients"].tolist(), "irls_iterations": model["iterations"], "l2_penalty_nonintercept": model["l2"], "training_rows": model["training_rows"], "training_weed": model["training_weed"], "training_crop": model["training_crop"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("project-root", "candidates", "semantic-scores", "targets", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    for name in ("candidates", "semantic-scores", "targets"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("This nested recalibration diagnostic is CPU-only")
    root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    names = ("candidates", "semantic_scores", "targets")
    paths = {name: getattr(args, name).resolve() for name in names}
    expected = {name: getattr(args, f"expected_{name}_sha256") for name in names}
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"Input hash mismatch: {name}")
    candidates = {row["candidate_id"]: row for row in read_tsv(paths["candidates"])}
    semantic = {row["candidate_id"]: row for row in read_tsv(paths["semantic_scores"])}
    targets = {row["candidate_id"]: row for row in read_tsv(paths["targets"])}
    if len(targets) != 5713 or set(candidates) != set(targets) or set(semantic) != set(targets):
        raise ValueError("Candidate coverage mismatch")
    rows: list[dict[str, object]] = []
    by_session: dict[str, list[tuple[str, float]]] = defaultdict(list)
    epsilon = 1e-8
    for candidate_id, candidate in candidates.items():
        score = semantic[candidate_id]
        target = targets[candidate_id]
        if candidate["session_id"] != score["session_id"] or candidate["session_id"] != target["session_id"] or candidate["file"] != target["file"]:
            raise ValueError(f"Lineage mismatch: {candidate_id}")
        width = float(candidate["x1_exclusive"]) - float(candidate["x0"])
        height = float(candidate["y1_exclusive"]) - float(candidate["y0"])
        area = float(candidate["area_pixels"])
        role_logit = math.log((float(score["weed_probability_mean"]) + epsilon) / (float(score["crop_probability_mean"]) + epsilon))
        row: dict[str, object] = {
            "candidate_id": candidate_id, "session_id": candidate["session_id"], "file": candidate["file"],
            "truth": target["dataset_role_target"], "eligible": target["policy_scoring_eligible"] == "1",
            "role_logit": role_logit, "soil_probability_mean": float(score["soil_probability_mean"]),
            "log1p_area_pixels": math.log1p(area), "mean_foreground_probability": float(candidate["mean_foreground_probability"]), "max_foreground_probability": float(candidate["max_foreground_probability"]),
            "log_bbox_area": math.log(max(1.0, width * height)), "log_bbox_aspect": math.log(max(width, 1.0) / max(height, 1.0)), "bbox_fill_fraction": area / max(1.0, width * height),
            "centroid_x_normalized": float(candidate["centroid_x"]) / 600.0, "centroid_y_normalized": float(candidate["centroid_y"]) / 600.0,
        }
        rows.append(row)
        by_session[candidate["session_id"]].append((candidate_id, role_logit))
    if set(by_session) != set(SESSIONS):
        raise ValueError("Provisional session coverage mismatch")
    indexed = {str(row["candidate_id"]): row for row in rows}
    for values in by_session.values():
        numeric = np.asarray([value for _, value in values], dtype=np.float64)
        median = float(np.median(numeric))
        first, third = np.quantile(numeric, [0.25, 0.75])
        scale = max(float(third - first), 1e-6)
        ranks = empirical_cdf(values)
        for candidate_id, value in values:
            indexed[candidate_id]["session_centered_role_logit"] = value - median
            indexed[candidate_id]["session_robust_role_logit"] = (value - median) / scale
            indexed[candidate_id]["session_empirical_cdf"] = ranks[candidate_id]

    crossfit: dict[str, dict[str, float]] = {group: {} for group in FEATURE_GROUPS}
    artifacts: dict[str, object] = {}
    for outer_session in SESSIONS:
        training_sessions = tuple(session for session in SESSIONS if session != outer_session)
        artifacts[outer_session] = {}
        outer_rows = [row for row in rows if row["session_id"] == outer_session]
        for group, features in FEATURE_GROUPS.items():
            sensitivity = []
            for l2 in L2_GRID:
                inner_auc = []
                for validation_session in training_sessions:
                    inner_training = tuple(session for session in training_sessions if session != validation_session)
                    train_x, train_y, _ = arrays(rows, inner_training, features)
                    validation_x, validation_y, _ = arrays(rows, (validation_session,), features)
                    model = fit_logistic(train_x, train_y, l2)
                    inner_auc.append(float(roc_auc(predict(model, validation_x), validation_y)))
                sensitivity.append({"l2": l2, "mean_inner_session_roc_auc": sum(inner_auc) / len(inner_auc), "per_inner_session_roc_auc": dict(zip(training_sessions, inner_auc, strict=True))})
            selected = max(sensitivity, key=lambda record: (record["mean_inner_session_roc_auc"], -abs(math.log10(record["l2"]))))
            train_x, train_y, _ = arrays(rows, training_sessions, features)
            model = fit_logistic(train_x, train_y, float(selected["l2"]))
            test_x = np.asarray([[row[name] for name in features] for row in outer_rows], dtype=np.float64)
            for row, probability in zip(outer_rows, predict(model, test_x), strict=True):
                crossfit[group][str(row["candidate_id"])] = float(probability)
            artifacts[outer_session][group] = {"held_out_provisional_session": outer_session, "training_sessions": list(training_sessions), "nested_l2_selection": sensitivity, "selected_l2": selected["l2"], "model": serial_model(model, features)}

    raw_scores = {str(row["candidate_id"]): 1.0 / (1.0 + math.exp(-float(row["role_logit"]))) for row in rows}
    evaluations = {"raw_role_baseline": evaluate(rows, raw_scores)}
    for group, scores in crossfit.items():
        evaluations[group] = evaluate(rows, scores)
    raw_per_session = evaluations["raw_role_baseline"]["per_provisional_session"]
    for group in FEATURE_GROUPS:
        group_per_session = evaluations[group]["per_provisional_session"]
        deltas = {session: group_per_session[session]["roc_auc"] - raw_per_session[session]["roc_auc"] for session in SESSIONS}
        evaluations[group]["roc_auc_delta_vs_raw_role"] = {"per_provisional_session": deltas, "mean": sum(deltas.values()) / len(deltas), "pooled": evaluations[group]["pooled_roc_auc"] - evaluations["raw_role_baseline"]["pooled_roc_auc"]}

    output_dir.mkdir(parents=True)
    prediction_path = output_dir / "nested_lodo_recalibrated_predictions.tsv"
    fields = ["candidate_id", "session_id", "file", "dataset_role_target", "policy_scoring_eligible", "raw_role_score", *[f"{group}_probability" for group in FEATURE_GROUPS]]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            candidate_id = str(row["candidate_id"])
            writer.writerow({"candidate_id": candidate_id, "session_id": row["session_id"], "file": row["file"], "dataset_role_target": row["truth"], "policy_scoring_eligible": int(row["eligible"]), "raw_role_score": f"{raw_scores[candidate_id]:.10f}", **{f"{group}_probability": f"{crossfit[group][candidate_id]:.10f}" for group in FEATURE_GROUPS}})
    artifact_path = output_dir / "nested_lodo_model_artifacts.json"
    artifact_path.write_text(json.dumps(artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "run_id": args.run_id,
        "status": "completed_posthoc_weedsgalore_nested_lodo_target_recalibration",
        "scope": {"public_data_only": True, "sealed_test_read": False, "posthoc_after_fixed_transfer_failure": True, "outer_role": "provisional_acquisition_date", "inner_model_selection": "leave_one_training_date_out", "truth_used_for_training_and_offline_scoring_only": True, "test_features_label_free": True, "formal_risk_certificate": False},
        "inputs": {name + "_sha256": sha256(path) for name, path in paths.items()},
        "l2_grid": list(L2_GRID), "feature_groups": {name: list(features) for name, features in FEATURE_GROUPS.items()},
        "evaluation": evaluations,
        "outputs": {"predictions_sha256": sha256(prediction_path), "model_artifacts_sha256": sha256(artifact_path)},
        "limitations": ["The four dates are provisional groups and do not establish independent farm-level generalization.", "This diagnostic was registered after observing fixed-policy transfer failure and is not a confirmatory test.", "Dataset role targets are not expert visual-support labels."],
        "runtime": {"python": sys.version, "numpy": np.__version__},
        "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(args.config.resolve()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
