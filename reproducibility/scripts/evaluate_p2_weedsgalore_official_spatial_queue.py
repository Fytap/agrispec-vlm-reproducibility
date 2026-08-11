#!/usr/bin/env python3
"""Official-spatial queue evaluation for the SAT major revision.

The script fits rankers on the official training split, selects regularisation
on the official validation split, refits on train+validation, and scores the
official test split once.  It reports candidate ranking, full queue curves,
random and exact-oracle controls, one-to-one matching, crop-overlap exposure,
prospective label-selection simulations, and repeated-location cluster
bootstrap intervals.  Target masks are used only for fitting/validation and
offline test scoring, never for proposal construction or test-time features.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.csgraph import maximum_bipartite_matching
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_commissioning_contract_audit import (  # noqa: E402
    DATES,
    audit_setting,
    load_truth,
)
from evaluate_p2_weedsgalore_official_spatial_proposal_audit import official_split_map  # noqa: E402
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import read_tsv, sha256  # noqa: E402


KS = (1, 5, 10, 20, 50)
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CROP_THRESHOLDS = (0.0, 0.01, 0.10, 0.50)
LOCAL_FEATURES = (
    "soil_probability_mean", "log1p_area_pixels", "mean_foreground_probability",
    "max_foreground_probability", "log_bbox_area", "log_bbox_aspect", "bbox_fill_fraction",
)
ROLE_LOCAL_FEATURES = ("role_logit", *LOCAL_FEATURES)
GEOMETRY_FEATURES = ("log1p_area_pixels", "log_bbox_area", "log_bbox_aspect", "bbox_fill_fraction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("project-root", "archive", "manifest", "candidate-root", "dino-embeddings", "masked-dino-embeddings", "scene-embeddings", "proposal-summary", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    for name in ("archive", "manifest", "candidates", "dino-embeddings", "masked-dino-embeddings", "scene-embeddings", "proposal-summary"):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--random-ranking-repeats", type=int, default=1000)
    parser.add_argument("--label-selection-repeats", type=int, default=10)
    parser.add_argument("--label-budget", type=int, default=300)
    parser.add_argument("--base-seed", type=int, default=2026081000)
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hash_ids(ids: list[str] | set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def load_npz(path: Path, id_key: str) -> tuple[list[str], np.ndarray]:
    archive = np.load(path)
    ids = [str(value) for value in archive[id_key]]
    matrix = np.asarray(archive["embeddings"], dtype=np.float64)
    if len(ids) != matrix.shape[0] or len(ids) != len(set(ids)):
        raise ValueError(f"Embedding coverage mismatch: {path}")
    return ids, matrix


def align_embeddings(rows: list[dict[str, object]], path: Path, id_key: str = "candidate_ids") -> np.ndarray:
    ids, matrix = load_npz(path, id_key)
    lookup = {candidate_id: index for index, candidate_id in enumerate(ids)}
    expected = [str(row["candidate_id"]) for row in rows]
    if set(expected) != set(ids):
        raise ValueError(f"Embedding candidate IDs do not match: {path}")
    return matrix[[lookup[candidate_id] for candidate_id in expected]]


def fit_model(x: np.ndarray, y: np.ndarray, c_value: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c_value, class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=20260810)
    model.fit(scaler.transform(x), y)
    return scaler, model


def predict(model: tuple[StandardScaler, LogisticRegression], x: np.ndarray) -> np.ndarray:
    scaler, classifier = model
    return classifier.predict_proba(scaler.transform(x))[:, 1]


def tune_and_score(
    rows: list[dict[str, object]], matrix: np.ndarray, target_contract: str,
    sampled_train_ids: set[str] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if target_contract not in {"eligible_crop_vs_weed", "all_candidate_weed_vs_rest"}:
        raise ValueError(target_contract)
    indices = {split: [] for split in ("train", "val", "test")}
    for index, row in enumerate(rows):
        split = str(row["official_split"])
        if target_contract == "eligible_crop_vs_weed" and not int(row["eligible"]):
            continue
        if split == "train" and sampled_train_ids is not None and str(row["candidate_id"]) not in sampled_train_ids:
            continue
        indices[split].append(index)
    labels = np.asarray([row["truth"] == "weed" for row in rows], dtype=np.int64)
    if len(set(labels[indices["train"]])) != 2 or len(set(labels[indices["val"]])) != 2:
        raise ValueError("Both classes are required for train and validation")
    sensitivity = []
    for c_value in C_GRID:
        model = fit_model(matrix[indices["train"]], labels[indices["train"]], c_value)
        values = predict(model, matrix[indices["val"]])
        sensitivity.append({"c": c_value, "validation_roc_auc": float(roc_auc_score(labels[indices["val"]], values)), "validation_average_precision": float(average_precision_score(labels[indices["val"]], values))})
    chosen = max(sensitivity, key=lambda item: (item["validation_roc_auc"], item["validation_average_precision"], -abs(math.log10(item["c"]))))
    fit_indices = indices["train"] + indices["val"]
    model = fit_model(matrix[fit_indices], labels[fit_indices], float(chosen["c"]))
    scores = predict(model, matrix)
    artifact = {
        "target_contract": target_contract,
        "selected_c": chosen["c"],
        "validation_sensitivity": sensitivity,
        "training_candidates": len(indices["train"]),
        "validation_candidates": len(indices["val"]),
        "refit_candidates": len(fit_indices),
        "sampled_training_ids_sha256": hash_ids(sampled_train_ids) if sampled_train_ids is not None else None,
    }
    return scores, artifact


def reconstruct_locations(
    manifest: list[dict[str, str]], split_map: dict[str, str], scene_path: Path,
) -> tuple[dict[str, str], list[dict[str, object]]]:
    sample_ids, matrix = load_npz(scene_path, "sample_ids")
    embedding = {sample_id: matrix[index] for index, sample_id in enumerate(sample_ids)}
    if set(embedding) != {row["sample_id"] for row in manifest}:
        raise ValueError("Scene embedding sample IDs do not match manifest")
    row_by_id = {row["sample_id"]: row for row in manifest}
    normalized = {sample_id: value / max(np.linalg.norm(value), 1e-12) for sample_id, value in embedding.items()}
    location: dict[str, str] = {}
    match_rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        anchors = sorted(sample_id for sample_id in embedding if split_map[sample_id] == split and row_by_id[sample_id]["provisional_source_session_id"] == DATES[0])
        expected = {"train": 32, "val": 8, "test": 8}[split]
        if len(anchors) != expected:
            raise ValueError(f"Unexpected May-25 anchor count for {split}: {len(anchors)}")
        for index, sample_id in enumerate(anchors):
            location[sample_id] = f"{split}_embedding_linked_location_{index:02d}"
            match_rows.append({"official_split": split, "location_id": location[sample_id], "acquisition_date": DATES[0], "sample_id": sample_id, "anchor_sample_id": sample_id, "cosine_similarity": 1.0, "row_best_nonassigned_similarity": "", "assignment_margin": "", "link_method": "anchor"})
        anchor_matrix = np.stack([normalized[sample_id] for sample_id in anchors])
        for date in DATES[1:]:
            targets = sorted(sample_id for sample_id in embedding if split_map[sample_id] == split and row_by_id[sample_id]["provisional_source_session_id"] == date)
            target_matrix = np.stack([normalized[sample_id] for sample_id in targets])
            similarities = target_matrix @ anchor_matrix.T
            target_indices, anchor_indices = linear_sum_assignment(-similarities)
            if len(target_indices) != len(targets):
                raise ValueError("Every target scene must receive one repeated-location link")
            for target_index, anchor_index in zip(target_indices, anchor_indices, strict=True):
                sample_id, anchor_id = targets[target_index], anchors[anchor_index]
                values = similarities[target_index]
                alternatives = np.delete(values, anchor_index)
                alternative = float(alternatives.max()) if len(alternatives) else float("nan")
                location[sample_id] = location[anchor_id]
                match_rows.append({"official_split": split, "location_id": location[sample_id], "acquisition_date": date, "sample_id": sample_id, "anchor_sample_id": anchor_id, "cosine_similarity": float(values[anchor_index]), "row_best_nonassigned_similarity": alternative, "assignment_margin": float(values[anchor_index] - alternative) if alternatives.size else "", "link_method": "within_split_global_Hungarian_DINOv2_cosine"})
    if len(location) != 156 or len(set(location.values())) != 48:
        raise ValueError({"samples": len(location), "locations": len(set(location.values()))})
    for location_id in set(location.values()):
        members = [sample_id for sample_id, value in location.items() if value == location_id]
        dates = [row_by_id[sample_id]["provisional_source_session_id"] for sample_id in members]
        if len(dates) not in (3, 4) or len(dates) != len(set(dates)):
            raise ValueError(f"Invalid repeated-location cluster: {location_id}")
    return location, match_rows


def reconstruct_test_patch_proxy(manifest: list[dict[str, str]], split_map: dict[str, str]) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Recover the two official test-patch groups from the disjoint capture-index ranges.

    The archive exposes the split but not a patch identifier.  Test tile stems
    form two non-overlapping capture-index ranges of 13 observations each.  We
    split at the largest numeric gap and retain this as a conservative patch
    proxy, not as an exact geocoordinate or location identifier.
    """
    test_rows = [row for row in manifest if split_map[row["sample_id"]] == "test"]
    indexed = sorted((int(Path(row["semantic_relpath"]).stem.rsplit("_", 1)[1]), row["sample_id"]) for row in test_rows)
    gaps = [indexed[index + 1][0] - indexed[index][0] for index in range(len(indexed) - 1)]
    cut = int(np.argmax(gaps)) + 1
    groups = (indexed[:cut], indexed[cut:])
    if sorted(len(group) for group in groups) != [13, 13]:
        raise ValueError(f"Test patch proxy did not produce two 13-image groups: {[len(group) for group in groups]}")
    mapping: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for group_index, group in enumerate(groups):
        patch_id = f"test_capture_index_patch_proxy_{group_index}"
        for capture_index, sample_id in group:
            mapping[sample_id] = patch_id
            records.append({"patch_proxy_id": patch_id, "sample_id": sample_id, "capture_index": capture_index, "group_min_capture_index": min(value for value, _ in group), "group_max_capture_index": max(value for value, _ in group), "group_size": len(group), "method": "largest capture-index gap within the official test split"})
    return mapping, records


def matching_count(candidate_ids: list[str], matched: dict[str, set[str]], allowed_instances: set[str]) -> int:
    candidates = [candidate_id for candidate_id in candidate_ids if matched.get(candidate_id)]
    instances = sorted({instance for candidate_id in candidates for instance in matched[candidate_id] if instance in allowed_instances})
    if not candidates or not instances:
        return 0
    instance_index = {instance: index for index, instance in enumerate(instances)}
    row_indices, column_indices = [], []
    for row_index, candidate_id in enumerate(candidates):
        for instance in matched[candidate_id]:
            if instance in instance_index:
                row_indices.append(row_index)
                column_indices.append(instance_index[instance])
    graph = csr_matrix((np.ones(len(row_indices)), (row_indices, column_indices)), shape=(len(candidates), len(instances)))
    result = maximum_bipartite_matching(graph, perm_type="column")
    return int((result >= 0).sum())


def selected_by_score(rows: list[dict[str, object]], scores: np.ndarray, indices: list[int], k: int) -> list[int]:
    by_file: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_file[str(rows[index]["file"])].append(index)
    selected = []
    for file_name in sorted(by_file):
        ranked = sorted(by_file[file_name], key=lambda index: (-float(scores[index]), str(rows[index]["candidate_id"])))
        selected.extend(ranked[:k])
    return selected


def metric_record(
    rows: list[dict[str, object]], selected: list[int], test_instances: list[dict[str, object]],
    spatial: dict[str, set[str]], qualified: dict[str, set[str]], group_name: str, group_value: str,
) -> dict[str, object]:
    candidate_ids = [str(rows[index]["candidate_id"]) for index in selected]
    weed_ids = {str(row["instance_id"]) for row in test_instances if row["truth_role"] == "weed"}
    spatial_hit = {instance for candidate_id in candidate_ids for instance in spatial.get(candidate_id, set())}
    qualified_hit = {instance for candidate_id in candidate_ids for instance in qualified.get(candidate_id, set())}
    weed_candidates = sum(rows[index]["truth"] == "weed" for index in selected)
    files = {str(row["sample_id"]) for row in test_instances}
    record: dict[str, object] = {
        "aggregation": group_name, "stratum": group_value,
        "images": len(files), "accepted_candidates": len(selected),
        "weed_candidates": weed_candidates,
        "all_candidate_queue_precision": weed_candidates / max(1, len(selected)),
        "weed_truth_instances": len(weed_ids),
        "spatial_recalled_weed_instances": len(spatial_hit & weed_ids),
        "queued_spatial_weed_instance_recall": len(spatial_hit & weed_ids) / max(1, len(weed_ids)),
        "role_qualified_recalled_weed_instances": len(qualified_hit & weed_ids),
        "queued_role_qualified_weed_instance_recall": len(qualified_hit & weed_ids) / max(1, len(weed_ids)),
        "one_to_one_spatial_matches": matching_count(candidate_ids, spatial, weed_ids),
        "one_to_one_spatial_recall": matching_count(candidate_ids, spatial, weed_ids) / max(1, len(weed_ids)),
        "one_to_one_role_qualified_matches": matching_count(candidate_ids, qualified, weed_ids),
        "one_to_one_role_qualified_recall": matching_count(candidate_ids, qualified, weed_ids) / max(1, len(weed_ids)),
    }
    for threshold in CROP_THRESHOLDS:
        exposed = {
            str(rows[index]["file"]) for index in selected
            if (float(rows[index]["crop_fraction_candidate"]) > 0 if threshold == 0 else float(rows[index]["crop_fraction_candidate"]) >= threshold)
        }
        record[f"crop_overlap_scene_numerator_{threshold:.2f}"] = len(exposed)
        record[f"crop_overlap_scene_frequency_{threshold:.2f}"] = len(exposed) / max(1, len(files))
    return record


def exact_oracle_indices(
    rows: list[dict[str, object]], indices: list[int], matched: dict[str, set[str]],
    allowed_instances: set[str], k: int, maximum_crop_fraction: float | None,
) -> tuple[list[int], bool]:
    by_file: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        if maximum_crop_fraction is None or float(rows[index]["crop_fraction_candidate"]) <= maximum_crop_fraction:
            by_file[str(rows[index]["file"])].append(index)
    selected: list[int] = []
    exact = True
    for file_name, candidates in sorted(by_file.items()):
        local_instances = sorted({instance for index in candidates for instance in matched.get(str(rows[index]["candidate_id"]), set()) if instance in allowed_instances})
        if not local_instances or not candidates:
            continue
        instance_index = {instance: pos for pos, instance in enumerate(local_instances)}
        candidate_index = {index: pos for pos, index in enumerate(candidates)}
        row_indices, col_indices = [], []
        for index in candidates:
            for instance in matched.get(str(rows[index]["candidate_id"]), set()):
                if instance in instance_index:
                    row_indices.append(instance_index[instance])
                    col_indices.append(candidate_index[index])
        incidence = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), shape=(len(local_instances), len(candidates)))
        # y_i - sum_c A_ic x_c <= 0; sum_c x_c <= K.
        constraints = hstack([-incidence, csr_matrix(np.eye(len(local_instances)))], format="csr")
        budget = csr_matrix(np.concatenate([np.ones(len(candidates)), np.zeros(len(local_instances))])[None, :])
        matrix = csr_matrix(np.vstack([constraints.toarray(), budget.toarray()]))
        lower = np.full(matrix.shape[0], -np.inf)
        upper = np.concatenate([np.zeros(len(local_instances)), [min(k, len(candidates))]])
        objective = np.concatenate([np.zeros(len(candidates)), -np.ones(len(local_instances))])
        result = milp(c=objective, integrality=np.ones(len(objective)), bounds=Bounds(np.zeros(len(objective)), np.ones(len(objective))), constraints=LinearConstraint(matrix, lower, upper), options={"time_limit": 60.0})
        if result.x is None:
            exact = False
            # Deterministic greedy fallback is preserved and flagged.
            covered: set[str] = set()
            remaining = list(candidates)
            chosen: list[int] = []
            for _ in range(min(k, len(remaining))):
                best = max(remaining, key=lambda index: (len((matched.get(str(rows[index]["candidate_id"]), set()) & allowed_instances) - covered), -index))
                chosen.append(best)
                covered |= matched.get(str(rows[best]["candidate_id"]), set()) & allowed_instances
                remaining.remove(best)
            selected.extend(chosen)
        else:
            selected.extend([index for index, value in zip(candidates, result.x[:len(candidates)], strict=True) if value > 0.5])
            exact = exact and bool(result.success)
    return selected, exact


def coreset_ids(matrix: np.ndarray, ids: list[str], budget: int, seed: int) -> set[str]:
    model = MiniBatchKMeans(n_clusters=budget, batch_size=2048, max_iter=100, n_init=1, random_state=seed).fit(matrix)
    distances = model.transform(matrix)
    chosen: set[int] = set()
    for center in range(budget):
        for index in np.argsort(distances[:, center], kind="mergesort"):
            if int(index) not in chosen:
                chosen.add(int(index))
                break
    if len(chosen) != budget:
        raise ValueError("Coreset selection produced duplicate candidates")
    return {ids[index] for index in chosen}


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("This evaluation is CPU-only")
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    checks = (
        (args.archive, args.expected_archive_sha256), (args.manifest, args.expected_manifest_sha256),
        (args.candidate_root / "candidate_components.tsv", args.expected_candidates_sha256),
        (args.dino_embeddings, args.expected_dino_embeddings_sha256),
        (args.masked_dino_embeddings, args.expected_masked_dino_embeddings_sha256),
        (args.scene_embeddings, args.expected_scene_embeddings_sha256),
        (args.proposal_summary, args.expected_proposal_summary_sha256),
    )
    for path, expected in checks:
        if sha256(path.resolve()) != expected:
            raise ValueError(f"Input hash mismatch: {path}")
    proposal_summary = json.loads(args.proposal_summary.read_text(encoding="utf-8"))
    if proposal_summary["setting_selection"]["selected_setting"] != "T070_A032":
        raise ValueError("Candidate root does not match training-selected proposal setting")

    started = time.perf_counter()
    manifest = read_tsv(args.manifest.resolve())
    split_map = official_split_map(args.archive.resolve(), manifest)
    truth = load_truth(args.archive.resolve(), manifest)
    audit = audit_setting(args.candidate_root.resolve(), manifest, truth)
    rows = audit["rows"]
    for row in rows:
        row["official_split"] = split_map[str(row["file"])]
    dino = align_embeddings(rows, args.dino_embeddings.resolve())
    masked_dino = align_embeddings(rows, args.masked_dino_embeddings.resolve())
    role_local = np.asarray([[float(row[name]) for name in ROLE_LOCAL_FEATURES] for row in rows], dtype=np.float64)
    geometry = np.asarray([[float(row[name]) for name in GEOMETRY_FEATURES] for row in rows], dtype=np.float64)
    source = 1.0 / (1.0 + np.exp(-np.asarray([float(row["role_logit"]) for row in rows])))

    model_scores: dict[str, np.ndarray] = {"source_resnet18_role_score": source}
    model_artifacts: dict[str, object] = {"source_resnet18_role_score": {"fit": "none", "provenance": "SugarBeets source-trained custom ResNet-18 U-Net semantic specialist candidate mean probabilities"}}
    specifications = (
        ("eligible_role_plus_local_logistic", role_local, "eligible_crop_vs_weed"),
        ("all_candidate_role_plus_local_logistic", role_local, "all_candidate_weed_vs_rest"),
        ("all_candidate_geometry_only_logistic", geometry, "all_candidate_weed_vs_rest"),
        ("eligible_frozen_dinov2_linear", dino, "eligible_crop_vs_weed"),
        ("all_candidate_frozen_dinov2_linear", dino, "all_candidate_weed_vs_rest"),
        ("all_candidate_component_masked_dinov2_linear", masked_dino, "all_candidate_weed_vs_rest"),
    )
    for name, matrix, target in specifications:
        model_scores[name], model_artifacts[name] = tune_and_score(rows, matrix, target)

    test_indices = [index for index, row in enumerate(rows) if row["official_split"] == "test"]
    test_instances = [row for row in audit["instances"] if split_map[str(row["sample_id"])] == "test"]
    labels = np.asarray([row["truth"] == "weed" for row in rows], dtype=np.int64)
    ranking_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for model_name, scores in model_scores.items():
        for subset, indices in (
            ("all_candidates", test_indices),
            ("eligible_crop_or_weed", [index for index in test_indices if int(rows[index]["eligible"])]),
        ):
            y = labels[indices]
            ranking_rows.append({"model": model_name, "test_subset": subset, "candidates": len(indices), "weed_candidates": int(y.sum()), "nonweed_candidates": int((1 - y).sum()), "roc_auc": float(roc_auc_score(y, scores[indices])), "average_precision": float(average_precision_score(y, scores[indices]))})
        for index in test_indices:
            prediction_rows.append({"model": model_name, "candidate_id": rows[index]["candidate_id"], "file": rows[index]["file"], "acquisition_date": rows[index]["date"], "dataset_role_target": rows[index]["truth"], "score": f"{float(scores[index]):.12f}"})

    location_map, location_rows = reconstruct_locations(manifest, split_map, args.scene_embeddings.resolve())
    patch_map, patch_mapping_rows = reconstruct_test_patch_proxy(manifest, split_map)
    for row in rows:
        row["location_id"] = location_map[str(row["file"])]
    queue_rows: list[dict[str, object]] = []
    per_location_rows: list[dict[str, object]] = []
    per_patch_rows: list[dict[str, object]] = []
    for model_name, scores in model_scores.items():
        for k in KS:
            selected = selected_by_score(rows, scores, test_indices, k)
            pooled = metric_record(rows, selected, test_instances, audit["spatial"], audit["qualified"], "pooled_test", "ALL")
            queue_rows.append({"model": model_name, "k": k, **pooled})
            for date in DATES:
                date_indices = [index for index in test_indices if rows[index]["date"] == date]
                date_selected = selected_by_score(rows, scores, date_indices, k)
                date_instances = [row for row in test_instances if row["date"] == date]
                queue_rows.append({"model": model_name, "k": k, **metric_record(rows, date_selected, date_instances, audit["spatial"], audit["qualified"], "acquisition_date", date)})
            for location_id in sorted({location_map[str(row["sample_id"])] for row in test_instances}):
                local_indices = [index for index in test_indices if rows[index]["location_id"] == location_id]
                local_selected = selected_by_score(rows, scores, local_indices, k)
                local_instances = [row for row in test_instances if location_map[str(row["sample_id"])] == location_id]
                per_location_rows.append({"model": model_name, "k": k, **metric_record(rows, local_selected, local_instances, audit["spatial"], audit["qualified"], "embedding_linked_location", location_id)})
            for patch_id in sorted(set(patch_map.values())):
                patch_indices = [index for index in test_indices if patch_map[str(rows[index]["file"])] == patch_id]
                patch_selected = selected_by_score(rows, scores, patch_indices, k)
                patch_instances = [row for row in test_instances if patch_map[str(row["sample_id"])] == patch_id]
                per_patch_rows.append({"model": model_name, "k": k, **metric_record(rows, patch_selected, patch_instances, audit["spatial"], audit["qualified"], "test_patch_proxy", patch_id)})

    weed_test_ids = {str(row["instance_id"]) for row in test_instances if row["truth_role"] == "weed"}
    oracle_rows: list[dict[str, object]] = []
    oracle_selected_lookup: dict[tuple[str, int], list[int]] = {}
    for contract, matched in (("spatial", audit["spatial"]), ("role_qualified", audit["qualified"])):
        for constraint_name, maximum_crop in (("unconstrained", None), ("crop_fraction_at_most_0.01", 0.01)):
            for k in KS:
                chosen, exact = exact_oracle_indices(rows, test_indices, matched, weed_test_ids, k, maximum_crop)
                oracle_selected_lookup[(f"oracle_{contract}_{constraint_name}", k)] = chosen
                metrics = metric_record(rows, chosen, test_instances, audit["spatial"], audit["qualified"], "pooled_test", "ALL")
                oracle_rows.append({"oracle_contract": contract, "crop_constraint": constraint_name, "k": k, "optimization_exact_all_images": int(exact), **metrics})

    random_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.base_seed)
    by_file = defaultdict(list)
    for index in test_indices:
        by_file[str(rows[index]["file"])].append(index)
    for repeat in range(args.random_ranking_repeats):
        shuffled_scores = np.zeros(len(rows), dtype=np.float64)
        for indices in by_file.values():
            shuffled_scores[indices] = rng.random(len(indices))
        for k in KS:
            selected = selected_by_score(rows, shuffled_scores, test_indices, k)
            metrics = metric_record(rows, selected, test_instances, audit["spatial"], audit["qualified"], "pooled_test", "ALL")
            random_rows.append({"repeat": repeat, "seed": args.base_seed, "k": k, **metrics})

    # Conservative descriptive cluster bootstrap over the two official-test patch proxies.
    primary_model = "all_candidate_frozen_dinov2_linear"
    cluster_rows = [row for row in per_patch_rows if row["model"] == primary_model and int(row["k"]) == 20]
    bootstrap_rows: list[dict[str, object]] = []
    cluster_rng = np.random.default_rng(args.base_seed + 1)
    for repeat in range(args.bootstrap_repeats):
        sample = cluster_rng.integers(0, len(cluster_rows), size=len(cluster_rows))
        selected_clusters = [cluster_rows[index] for index in sample]
        weed_candidates = sum(int(row["weed_candidates"]) for row in selected_clusters)
        accepted = sum(int(row["accepted_candidates"]) for row in selected_clusters)
        recalled = sum(int(row["role_qualified_recalled_weed_instances"]) for row in selected_clusters)
        truth_count = sum(int(row["weed_truth_instances"]) for row in selected_clusters)
        bootstrap_rows.append({"repeat": repeat, "clusters_sampled": len(selected_clusters), "all_candidate_queue_precision": weed_candidates / max(1, accepted), "queued_role_qualified_weed_instance_recall": recalled / max(1, truth_count)})

    # Prospective selection uses no role labels; the retrospective comparator is explicit.
    train_indices = [index for index, row in enumerate(rows) if row["official_split"] == "train"]
    train_ids = [str(rows[index]["candidate_id"]) for index in train_indices]
    truth_by_id = {str(row["candidate_id"]): str(row["truth"]) for row in rows}
    label_rows: list[dict[str, object]] = []
    label_id_rows: list[dict[str, object]] = []
    for repeat in range(args.label_selection_repeats):
        repeat_rng = np.random.default_rng(args.base_seed + 100 + repeat)
        uniform_local = repeat_rng.choice(len(train_indices), size=args.label_budget, replace=False)
        uniform_ids = {train_ids[int(index)] for index in uniform_local}
        core_ids = coreset_ids(dino[train_indices], train_ids, args.label_budget, args.base_seed + 200 + repeat)
        weed_train = [candidate_id for candidate_id in train_ids if truth_by_id[candidate_id] == "weed"]
        nonweed_train = [candidate_id for candidate_id in train_ids if truth_by_id[candidate_id] != "weed"]
        stratified_ids = set(repeat_rng.choice(weed_train, size=args.label_budget // 2, replace=False)) | set(repeat_rng.choice(nonweed_train, size=args.label_budget - args.label_budget // 2, replace=False))
        for strategy, selected_ids, uses_labels in (("prospective_uniform", uniform_ids, 0), ("prospective_dinov2_coreset", core_ids, 0), ("retrospective_class_stratified", stratified_ids, 1)):
            scores, artifact = tune_and_score(rows, dino, "all_candidate_weed_vs_rest", selected_ids)
            selected = selected_by_score(rows, scores, test_indices, 20)
            metrics = metric_record(rows, selected, test_instances, audit["spatial"], audit["qualified"], "pooled_test", "ALL")
            y = labels[test_indices]
            label_rows.append({"strategy": strategy, "uses_labels_for_selection": uses_labels, "repeat": repeat, "seed": args.base_seed + repeat, "screened_training_pool": len(train_indices), "labels_revealed": len(selected_ids), "selected_ids_sha256": hash_ids(selected_ids), "selected_c": artifact["selected_c"], "test_all_candidate_roc_auc": float(roc_auc_score(y, scores[test_indices])), "test_all_candidate_average_precision": float(average_precision_score(y, scores[test_indices])), **metrics})
            label_id_rows.extend({"strategy": strategy, "repeat": repeat, "candidate_id": candidate_id} for candidate_id in sorted(selected_ids))

    # Ceiling decomposition for the strongest fully frozen ranker at K=20.
    proposal_test = next(row for row in read_tsv(args.proposal_summary.parent / "official_split_proposal_layers.tsv") if row["variant"] == "training_selected_commissioned" and row["official_split"] == "test" and row["acquisition_date"] == "ALL" and row["truth_role"] == "weed")
    observed = next(row for row in queue_rows if row["model"] == primary_model and int(row["k"]) == 20 and row["aggregation"] == "pooled_test")
    oracle = next(row for row in oracle_rows if row["oracle_contract"] == "role_qualified" and row["crop_constraint"] == "unconstrained" and int(row["k"]) == 20)
    ceiling_rows = [{
        "model": primary_model, "k": 20,
        "truth_weed_instances": proposal_test["truth_instances"],
        "role_qualified_proposal_ceiling": proposal_test["role_qualified_proposal_recall"],
        "exact_oracle_budget_ceiling": oracle["queued_role_qualified_weed_instance_recall"],
        "observed_ranked_queue_recall": observed["queued_role_qualified_weed_instance_recall"],
        "proposal_generation_loss_from_perfect_recall": 1.0 - float(proposal_test["role_qualified_proposal_recall"]),
        "fixed_budget_loss_within_proposals": float(proposal_test["role_qualified_proposal_recall"]) - float(oracle["queued_role_qualified_weed_instance_recall"]),
        "observed_ranking_gap_to_oracle": float(oracle["queued_role_qualified_weed_instance_recall"]) - float(observed["queued_role_qualified_weed_instance_recall"]),
    }]

    output.mkdir(parents=True)
    files = {
        "ranking": output / "official_test_candidate_ranking.tsv",
        "predictions": output / "official_test_predictions.tsv",
        "queue": output / "official_test_queue_curves.tsv",
        "per_location": output / "official_test_queue_by_embedding_linked_location.tsv",
        "location_mapping": output / "embedding_linked_location_mapping.tsv",
        "per_patch": output / "official_test_queue_by_patch_proxy.tsv",
        "patch_mapping": output / "official_test_patch_proxy_mapping.tsv",
        "oracle": output / "official_test_exact_oracles.tsv",
        "random": output / "official_test_random_ranking_draws.tsv",
        "bootstrap": output / "official_test_location_cluster_bootstrap_draws.tsv",
        "label_budget": output / "prospective_label_selection_results.tsv",
        "label_ids": output / "prospective_label_selection_ids.tsv",
        "ceiling": output / "proposal_budget_ranking_ceiling_decomposition.tsv",
    }
    for key, values in (("ranking", ranking_rows), ("predictions", prediction_rows), ("queue", queue_rows), ("per_location", per_location_rows), ("location_mapping", location_rows), ("per_patch", per_patch_rows), ("patch_mapping", patch_mapping_rows), ("oracle", oracle_rows), ("random", random_rows), ("bootstrap", bootstrap_rows), ("label_budget", label_rows), ("label_ids", label_id_rows), ("ceiling", ceiling_rows)):
        write_tsv(files[key], values)

    primary_queue = {str(k): next(row for row in queue_rows if row["model"] == primary_model and int(row["k"]) == k and row["aggregation"] == "pooled_test") for k in KS}
    bootstrap_precision = np.asarray([float(row["all_candidate_queue_precision"]) for row in bootstrap_rows])
    bootstrap_recall = np.asarray([float(row["queued_role_qualified_weed_instance_recall"]) for row in bootstrap_rows])
    summary = {
        "run_id": args.run_id,
        "status": "completed_official_spatial_queue_major_revision_evaluation",
        "scope": {"public_data_only": True, "sealed_test_read": False, "official_training_fit": True, "official_validation_hyperparameter_selection": True, "official_test_offline_scoring_only": True, "test_truth_used_as_feature": False, "formal_risk_certificate": False},
        "proposal_setting": "T070_A032 selected using official training split only",
        "models": model_artifacts,
        "primary_frozen_model": primary_model,
        "primary_queue_by_k": primary_queue,
        "patch_cluster_bootstrap_k20": {"clusters": len(cluster_rows), "repeats": args.bootstrap_repeats, "interpretation": "descriptive percentile interval; only two capture-index patch proxies, so no formal uncertainty claim", "precision_percentiles_2.5_50_97.5": np.percentile(bootstrap_precision, [2.5, 50, 97.5]).tolist(), "role_qualified_recall_percentiles_2.5_50_97.5": np.percentile(bootstrap_recall, [2.5, 50, 97.5]).tolist()},
        "cluster_reconstruction": {"primary_inference_cluster": "two 13-image official-test capture-index patch proxies split at the largest numeric gap", "patch_proxy_count": len(set(patch_map.values())), "exact_geocoordinates_available": False, "embedding_linked_location_diagnostic": {"locations_total": len(set(location_map.values())), "test_locations": len({value for value in location_map.values() if value.startswith("test_")}), "method": "within-official-split Hungarian assignment of frozen DINOv2 full-tile cosine similarity", "used_for_primary_interval": False, "reason": "assignment margins were insufficient for authoritative location identity"}},
        "queue_contract": {"ranking_tie_break": "descending score then candidate_id lexical ascending", "fewer_than_k": "all available candidates are returned and actual accepted count is the precision denominator", "crop_overlap_thresholds": list(CROP_THRESHOLDS), "one_to_one": "maximum bipartite matching under the same spatial or role-qualified edge contract"},
        "label_selection": {"budget": args.label_budget, "repeats": args.label_selection_repeats, "prospective_strategies": ["uniform", "DINOv2 coreset"], "retrospective_reference": "class-stratified sampling uses labels for selection and is not a realizable prospective workflow"},
        "inputs": {"archive_sha256": sha256(args.archive.resolve()), "manifest_sha256": sha256(args.manifest.resolve()), "candidates_sha256": sha256(args.candidate_root / "candidate_components.tsv"), "dino_embeddings_sha256": sha256(args.dino_embeddings.resolve()), "masked_dino_embeddings_sha256": sha256(args.masked_dino_embeddings.resolve()), "scene_embeddings_sha256": sha256(args.scene_embeddings.resolve()), "proposal_summary_sha256": sha256(args.proposal_summary.resolve())},
        "outputs": {key + "_sha256": sha256(path) for key, path in files.items()},
        "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version, "numpy": np.__version__, "scikit_learn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__},
        "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(args.config.resolve()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
