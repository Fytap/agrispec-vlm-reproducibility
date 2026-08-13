#!/usr/bin/env python3
"""Post-test sensitivity of commissioning to ranker source and match contract."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_p2_weedsgalore_commissioning_contract_audit as base  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_proposal_audit import official_split_map  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_queue import (  # noqa: E402
    C_GRID,
    ROLE_LOCAL_FEATURES,
    fit_model,
    matching_count,
    predict,
)


INSTANCE_THRESHOLDS = (0.25, 0.50, 0.75)
LABELED_THRESHOLDS = (0.25, 0.50, 0.75)
PURITY_THRESHOLDS = (0.50, 0.75, 0.90)
CROP_THRESHOLDS = (0.00, 0.01, 0.10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--original-root", type=Path, required=True)
    p.add_argument("--grid-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tune_and_fit(rows: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    x = np.asarray([[float(row[name]) for name in ROLE_LOCAL_FEATURES] for row in rows], dtype=np.float64)
    y = np.asarray([row["truth"] == "weed" for row in rows], dtype=np.int64)
    train = np.asarray([i for i, row in enumerate(rows) if row["official_split"] == "train"], dtype=int)
    validation = np.asarray([i for i, row in enumerate(rows) if row["official_split"] == "val"], dtype=int)
    if len(np.unique(y[train])) != 2 or len(np.unique(y[validation])) != 2:
        raise ValueError("Both classes are required in training and validation")
    sensitivity = []
    for value in C_GRID:
        model = fit_model(x[train], y[train], value)
        score = predict(model, x[validation])
        sensitivity.append({
            "C": value,
            "validation_auc": float(roc_auc_score(y[validation], score)),
            "validation_ap": float(average_precision_score(y[validation], score)),
        })
    selected = max(sensitivity, key=lambda row: (row["validation_auc"], row["validation_ap"], -abs(np.log10(row["C"]))))
    model = fit_model(x[np.concatenate([train, validation])], y[np.concatenate([train, validation])], selected["C"])
    return model, {"selected_C": selected["C"], "sensitivity": sensitivity, "rows": len(rows)}


def dynamic_edges(
    audit: dict[str, Any], instance_threshold: float, labeled_threshold: float, purity_threshold: float
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    spatial: dict[str, set[str]] = defaultdict(set)
    qualified: dict[str, set[str]] = defaultdict(set)
    for row in audit["matches"]:
        if row["truth_role"] != "weed":
            continue
        candidate_id = str(row["candidate_id"])
        instance_id = str(row["instance_id"])
        if float(row["instance_coverage"]) >= instance_threshold:
            spatial[candidate_id].add(instance_id)
            if (
                float(row["candidate_labeled_coverage"]) >= labeled_threshold
                and float(row["same_role_purity"]) >= purity_threshold
            ):
                qualified[candidate_id].add(instance_id)
    return spatial, qualified


def evaluate_selected(
    audit: dict[str, Any], selected: list[dict[str, Any]], instance_threshold: float,
    labeled_threshold: float, purity_threshold: float,
) -> dict[str, float]:
    spatial, qualified = dynamic_edges(audit, instance_threshold, labeled_threshold, purity_threshold)
    selected_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_file[str(row["file"])].append(row)
    weed_by_file: dict[str, set[str]] = defaultdict(set)
    for row in audit["instances"]:
        if row["truth_role"] == "weed" and row["sample_id"] in selected_by_file:
            weed_by_file[str(row["sample_id"])].add(str(row["instance_id"]))
    weed_total = sum(len(values) for values in weed_by_file.values())
    spatial_union: set[str] = set()
    qualified_union: set[str] = set()
    one_to_one = 0
    eligible = 0
    crop_scenes = {threshold: 0 for threshold in CROP_THRESHOLDS}
    for file, rows in selected_by_file.items():
        candidate_ids = [str(row["candidate_id"]) for row in rows]
        spatial_union.update(instance for candidate in candidate_ids for instance in spatial.get(candidate, set()))
        qualified_union.update(instance for candidate in candidate_ids for instance in qualified.get(candidate, set()))
        one_to_one += matching_count(candidate_ids, qualified, weed_by_file[file])
        for row in rows:
            coverage = float(row["candidate_labeled_coverage"])
            weed_fraction = float(row["weed_fraction_candidate"])
            weed_purity = weed_fraction / max(1e-12, coverage)
            eligible += int(coverage >= labeled_threshold and weed_purity >= purity_threshold)
        for threshold in CROP_THRESHOLDS:
            crop_scenes[threshold] += int(any(float(row["crop_fraction_candidate"]) >= threshold and float(row["crop_fraction_candidate"]) > 0 for row in rows))
    output = {
        "accepted_candidates": float(len(selected)),
        "candidate_precision": eligible / max(1, len(selected)),
        "spatial_recall": len(spatial_union) / max(1, weed_total),
        "set_coverage_recall": len(qualified_union) / max(1, weed_total),
        "one_to_one_recall": one_to_one / max(1, weed_total),
    }
    for threshold in CROP_THRESHOLDS:
        output[f"crop_exposure_scene_frequency_at_{threshold:.2f}"] = crop_scenes[threshold] / max(1, len(selected_by_file))
    return output


def proposal_ceiling(
    audit: dict[str, Any], test_rows: list[dict[str, Any]], instance_threshold: float,
    labeled_threshold: float, purity_threshold: float,
) -> dict[str, float]:
    _, qualified = dynamic_edges(audit, instance_threshold, labeled_threshold, purity_threshold)
    by_file: dict[str, list[str]] = defaultdict(list)
    for row in test_rows:
        by_file[str(row["file"])].append(str(row["candidate_id"]))
    weed_by_file: dict[str, set[str]] = defaultdict(set)
    for row in audit["instances"]:
        if row["truth_role"] == "weed" and row["sample_id"] in by_file:
            weed_by_file[str(row["sample_id"])].add(str(row["instance_id"]))
    weed_total = sum(len(values) for values in weed_by_file.values())
    set_union: set[str] = set()
    oracle = 0
    for file, candidate_ids in by_file.items():
        set_union.update(instance for candidate in candidate_ids for instance in qualified.get(candidate, set()))
        oracle += min(20, matching_count(candidate_ids, qualified, weed_by_file[file]))
    return {
        "proposal_set_coverage_ceiling": len(set_union) / max(1, weed_total),
        "per_tile_K20_oracle_one_to_one_recall": oracle / max(1, weed_total),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = base.read_tsv(args.manifest.resolve())
    truth = base.load_truth(args.archive.resolve(), manifest)
    split_map = official_split_map(args.archive.resolve(), manifest)
    audits = {
        "original": base.audit_setting(args.original_root.resolve(), manifest, truth),
        "commissioned": base.audit_setting((args.grid_root.resolve() / "T070_A032"), manifest, truth),
    }
    for audit in audits.values():
        for row in audit["rows"]:
            row["official_split"] = split_map[str(row["file"])]

    trained = {}
    ranker_audit = {}
    trained["original_only"], ranker_audit["original_only"] = tune_and_fit(audits["original"]["rows"])
    trained["commissioned_only"], ranker_audit["commissioned_only"] = tune_and_fit(audits["commissioned"]["rows"])
    union_rows = [dict(row) for variant in ("original", "commissioned") for row in audits[variant]["rows"]]
    trained["union"], ranker_audit["union"] = tune_and_fit(union_rows)

    schemes = {
        "commissioned_only_shared": {"original": trained["commissioned_only"], "commissioned": trained["commissioned_only"]},
        "original_only_shared": {"original": trained["original_only"], "commissioned": trained["original_only"]},
        "union_shared": {"original": trained["union"], "commissioned": trained["union"]},
        "reciprocal_cross_pool": {"original": trained["commissioned_only"], "commissioned": trained["original_only"]},
    }
    scheme_scores: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for scheme, assignment in schemes.items():
        for variant, model in assignment.items():
            rows = [row for row in audits[variant]["rows"] if row["official_split"] == "test"]
            matrix = np.asarray([[float(row[name]) for name in ROLE_LOCAL_FEATURES] for row in rows], dtype=np.float64)
            values = predict(model, matrix)
            scheme_scores[scheme][variant] = [{**row, "score": float(values[i])} for i, row in enumerate(rows)]

    metric_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    ceiling_rows: list[dict[str, Any]] = []
    for scheme, variants in scheme_scores.items():
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for variant, rows in variants.items():
            by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_file[str(row["file"])].append(row)
            grouped[variant] = {
                file: sorted(values, key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
                for file, values in by_file.items()
            }
        files = sorted(set(grouped["original"]) & set(grouped["commissioned"]))
        common_k = {file: min(20, len(grouped["original"][file]), len(grouped["commissioned"][file])) for file in files}
        selected = {
            variant: [row for file in files for row in grouped[variant][file][: common_k[file]]]
            for variant in ("original", "commissioned")
        }
        for instance_threshold in INSTANCE_THRESHOLDS:
            for labeled_threshold in LABELED_THRESHOLDS:
                for purity_threshold in PURITY_THRESHOLDS:
                    current = {}
                    for variant in ("original", "commissioned"):
                        current[variant] = evaluate_selected(
                            audits[variant], selected[variant], instance_threshold, labeled_threshold, purity_threshold
                        )
                        ceiling = proposal_ceiling(
                            audits[variant], variants[variant], instance_threshold, labeled_threshold, purity_threshold
                        )
                        metric_rows.append({
                            "ranker_scheme": scheme,
                            "variant": variant,
                            "instance_coverage": instance_threshold,
                            "candidate_labeled_coverage": labeled_threshold,
                            "same_role_purity": purity_threshold,
                            **current[variant],
                            **ceiling,
                        })
                        ceiling_rows.append({
                            "ranker_scheme": scheme,
                            "variant": variant,
                            "instance_coverage": instance_threshold,
                            "candidate_labeled_coverage": labeled_threshold,
                            "same_role_purity": purity_threshold,
                            **ceiling,
                        })
                    effect_rows.append({
                        "ranker_scheme": scheme,
                        "instance_coverage": instance_threshold,
                        "candidate_labeled_coverage": labeled_threshold,
                        "same_role_purity": purity_threshold,
                        **{
                            f"commissioned_minus_original_{key}": current["commissioned"][key] - current["original"][key]
                            for key in current["original"]
                        },
                    })

    stability_rows: list[dict[str, Any]] = []
    for scheme in schemes:
        values = [row for row in effect_rows if row["ranker_scheme"] == scheme]
        for metric in ("one_to_one_recall", "set_coverage_recall", "spatial_recall", "candidate_precision"):
            column = f"commissioned_minus_original_{metric}"
            observed = np.asarray([float(row[column]) for row in values])
            stability_rows.append({
                "ranker_scheme": scheme,
                "metric": metric,
                "contracts": len(observed),
                "positive_contracts": int((observed > 0).sum()),
                "zero_contracts": int((observed == 0).sum()),
                "negative_contracts": int((observed < 0).sum()),
                "minimum_effect": float(observed.min()),
                "median_effect": float(np.median(observed)),
                "maximum_effect": float(observed.max()),
            })

    write_tsv(output / "contract_grid_metrics.tsv", metric_rows)
    write_tsv(output / "contract_grid_effects.tsv", effect_rows)
    write_tsv(output / "effect_sign_stability.tsv", stability_rows)
    write_tsv(output / "proposal_ceiling_and_oracle.tsv", ceiling_rows)
    primary_rows = [
        row for row in metric_rows
        if float(row["instance_coverage"]) == 0.50
        and float(row["candidate_labeled_coverage"]) == 0.50
        and float(row["same_role_purity"]) == 0.90
    ]
    write_tsv(output / "primary_contract_ranker_schemes.tsv", primary_rows)
    summary = {
        "run_id": args.run_id,
        "status": "completed_post_test_ranker_and_contract_sensitivity",
        "analysis_class": "post_test_exploratory_sensitivity_analysis",
        "ranker_schemes": list(schemes),
        "contract_grid": {
            "instance_coverage": INSTANCE_THRESHOLDS,
            "candidate_labeled_coverage": LABELED_THRESHOLDS,
            "same_role_purity": PURITY_THRESHOLDS,
        },
        "ranker_training_audit": ranker_audit,
        "truth_use": "training_and_validation_labels_for_ranker_fit; official_test_labels_for_offline_scoring_only",
        "input_hashes": {
            "archive": sha256(args.archive.resolve()),
            "manifest": sha256(args.manifest.resolve()),
            "original_candidates": sha256(args.original_root.resolve() / "candidate_components.tsv"),
            "commissioned_candidates": sha256(args.grid_root.resolve() / "T070_A032" / "candidate_components.tsv"),
            "script": sha256(Path(__file__).resolve()),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
