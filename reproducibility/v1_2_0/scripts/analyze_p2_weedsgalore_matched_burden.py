#!/usr/bin/env python3
"""Matched-burden commissioning audit with deterministic one-to-one matching."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_p2_weedsgalore_commissioning_contract_audit as base  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_proposal_audit import official_split_map  # noqa: E402
from evaluate_p2_weedsgalore_official_spatial_queue import (  # noqa: E402
    C_GRID,
    ROLE_LOCAL_FEATURES,
    fit_model,
    matching_count,
    metric_record,
    predict,
)
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--original-root", type=Path, required=True)
    p.add_argument("--grid-root", type=Path, required=True)
    p.add_argument("--proposal-summary", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def metric(records: list[dict[str, Any]], accepted_ids: set[str], audit: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accepted = [r for r in records if str(r["candidate_id"]) in accepted_ids]
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        all_by_file[str(row["file"])].append(row)
    for row in accepted:
        by_file[str(row["file"])].append(row)
    totals = Counter()
    tile_rows: list[dict[str, Any]] = []
    for file, all_rows in sorted(all_by_file.items()):
        selected = by_file.get(file, [])
        weed_ids = {
            str(r["instance_id"])
            for r in audit["instances"]
            if str(r["sample_id"]) == file and str(r["truth_role"]) == "weed"
        }
        qids = {
            iid
            for row in selected
            for iid in audit["qualified"].get(str(row["candidate_id"]), set())
        }
        sids = {
            iid
            for row in selected
            for iid in audit["spatial"].get(str(row["candidate_id"]), set())
        }
        one = matching_count([str(row["candidate_id"]) for row in selected], audit["qualified"], weed_ids)
        eligible = sum(str(row["truth"]) == "weed" for row in selected)
        crop_overlap = any(float(row["crop_fraction_candidate"]) > 0 for row in selected)
        tile = {
            "file": file,
            "date": str(all_rows[0]["date"]),
            "weed_instances": len(weed_ids),
            "accepted_candidates": len(selected),
            "eligible_weed_candidates": eligible,
            "spatial_recalled": len(sids & weed_ids),
            "qualified_recalled": len(qids & weed_ids),
            "one_to_one_recalled": one,
            "crop_overlap_scene": int(crop_overlap),
        }
        tile_rows.append(tile)
        for key, value in tile.items():
            if key not in ("file", "date"):
                totals[key] += int(value)
        totals["images"] += 1
    summary = {
        "images": totals["images"],
        "weed_instances": totals["weed_instances"],
        "accepted_candidates": totals["accepted_candidates"],
        "mean_candidates_per_image": totals["accepted_candidates"] / max(1, totals["images"]),
        "all_candidate_precision": totals["eligible_weed_candidates"] / max(1, totals["accepted_candidates"]),
        "many_to_one_spatial_recall": totals["spatial_recalled"] / max(1, totals["weed_instances"]),
        "many_to_one_role_qualified_recall": totals["qualified_recalled"] / max(1, totals["weed_instances"]),
        "one_to_one_role_qualified_recall": totals["one_to_one_recalled"] / max(1, totals["weed_instances"]),
        "crop_overlap_scene_frequency": totals["crop_overlap_scene"] / max(1, totals["images"]),
    }
    return summary, tile_rows


def main() -> None:
    a = args()
    out = a.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = base.read_tsv(a.manifest.resolve())
    truth = base.load_truth(a.archive.resolve(), manifest)
    split_map = official_split_map(a.archive.resolve(), manifest)
    original = base.audit_setting(a.original_root.resolve(), manifest, truth)
    proposal_summary = json.loads(a.proposal_summary.read_text(encoding="utf-8"))
    selected_setting = str(proposal_summary["setting_selection"]["selected_setting"])
    if selected_setting != "T070_A032":
        raise ValueError("Expected the globally training-selected commissioned setting T070_A032")
    commissioned_global = base.audit_setting((a.grid_root / selected_setting).resolve(), manifest, truth)
    # Use the manuscript's shared-weight comparison: one commissioned-pool
    # train/validation ranker is applied unchanged to both test candidate pools.
    audits = {"original_T095_A032": original, "commissioned_T070_A032": commissioned_global}
    for audit in audits.values():
        for row in audit["rows"]:
            row["official_split"] = split_map[str(row["file"])]
    fit_rows = commissioned_global["rows"]
    feature_matrix = np.asarray([[float(row[name]) for name in ROLE_LOCAL_FEATURES] for row in fit_rows], dtype=np.float64)
    targets = np.asarray([row["truth"] == "weed" for row in fit_rows], dtype=np.int64)
    train_indices = [i for i, row in enumerate(fit_rows) if row["official_split"] == "train"]
    validation_indices = [i for i, row in enumerate(fit_rows) if row["official_split"] == "val"]
    sensitivity = []
    for c_value in C_GRID:
        candidate_model = fit_model(feature_matrix[train_indices], targets[train_indices], c_value)
        values = predict(candidate_model, feature_matrix[validation_indices])
        sensitivity.append({
            "c": c_value,
            "validation_auc": float(roc_auc_score(targets[validation_indices], values)),
            "validation_ap": float(average_precision_score(targets[validation_indices], values)),
        })
    selected_ranker = max(sensitivity, key=lambda r: (r["validation_auc"], r["validation_ap"], -abs(np.log10(r["c"]))))
    fitted = fit_model(feature_matrix[train_indices + validation_indices], targets[train_indices + validation_indices], float(selected_ranker["c"]))
    variants: dict[str, list[dict[str, Any]]] = {}
    for name, audit in audits.items():
        rows = audit["rows"]
        matrix = np.asarray([[float(row[feature]) for feature in ROLE_LOCAL_FEATURES] for row in rows], dtype=np.float64)
        scores = predict(fitted, matrix)
        variants[name] = [
            {**row, "score": float(scores[index])}
            for index, row in enumerate(rows)
            if row["official_split"] == "test"
        ]
    sorted_by_file: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, rows in variants.items():
        group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group[str(row["file"])].append(row)
        sorted_by_file[name] = {
            file: sorted(values, key=lambda r: (-float(r["score"]), str(r["candidate_id"])))
            for file, values in group.items()
        }
    files = sorted(set(sorted_by_file["original_T095_A032"]) & set(sorted_by_file["commissioned_T070_A032"]))
    if len(files) != len(sorted_by_file["original_T095_A032"]) or len(files) != len(sorted_by_file["commissioned_T070_A032"]):
        raise ValueError("Variant file sets differ")

    curve_rows: list[dict[str, Any]] = []
    tile_k20_rows: list[dict[str, Any]] = []
    curve_summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for requested_k in range(0, 21):
        common_k = {
            file: min(requested_k, len(sorted_by_file["original_T095_A032"][file]), len(sorted_by_file["commissioned_T070_A032"][file]))
            for file in files
        }
        for name in variants:
            accepted = {
                str(row["candidate_id"])
                for file in files
                for row in sorted_by_file[name][file][: common_k[file]]
            }
            summary, tiles = metric(variants[name], accepted, audits[name])
            record = {"variant": name, "requested_k": requested_k, **summary}
            curve_rows.append(record)
            curve_summaries[name].append(record)
            if requested_k == 20:
                for tile in tiles:
                    tile_k20_rows.append({"variant": name, "common_k": common_k[str(tile["file"])], **tile})

    fixed_ids: dict[str, set[str]] = {}
    fixed_totals: dict[str, int] = {}
    for name in variants:
        ids = {
            str(row["candidate_id"])
            for file in files
            for row in sorted_by_file[name][file][:20]
        }
        fixed_ids[name] = ids
        fixed_totals[name] = len(ids)
    common_n = min(fixed_totals.values())
    global_rows: list[dict[str, Any]] = []
    for n in range(common_n + 1):
        for name, rows in variants.items():
            eligible = [row for row in rows if str(row["candidate_id"]) in fixed_ids[name]]
            global_selected = sorted(eligible, key=lambda r: (-float(r["score"]), str(r["candidate_id"])))[:n]
            summary, _ = metric(rows, {str(row["candidate_id"]) for row in global_selected}, audits[name])
            global_rows.append({"variant": name, "global_N": n, **summary})

    def aurbc(records: list[dict[str, Any]], burden_key: str) -> float:
        x = np.asarray([float(r[burden_key]) for r in records])
        y = np.asarray([float(r["one_to_one_role_qualified_recall"]) for r in records])
        order = np.argsort(x)
        return float(np.trapezoid(y[order], x[order]) / max(1e-12, x[order][-1] - x[order][0]))

    summary = {
        "run_id": a.run_id,
        "status": "completed_post_test_matched_burden_audit",
        "common_files": len(files),
        "fixed_K20_available_totals": fixed_totals,
        "global_equal_N": common_n,
        "paired_common_K20": {
            name: next(r for r in curve_rows if r["variant"] == name and r["requested_k"] == 20)
            for name in variants
        },
        "global_equal_N_metrics": {
            name: next(r for r in global_rows if r["variant"] == name and r["global_N"] == common_n)
            for name in variants
        },
        "paired_common_K_AURBC": {
            name: aurbc(curve_summaries[name], "mean_candidates_per_image") for name in variants
        },
        "global_equal_N_AURBC": {
            name: aurbc([r for r in global_rows if r["variant"] == name], "mean_candidates_per_image")
            for name in variants
        },
        "selected_commissioned_setting": selected_setting,
        "shared_ranker": {"selected_C": selected_ranker["c"], "validation_sensitivity": sensitivity},
        "inputs": {
            "archive_sha256": sha(a.archive),
            "manifest_sha256": sha(a.manifest),
            "original_candidates_sha256": sha(a.original_root / "candidate_components.tsv"),
            "proposal_summary_sha256": sha(a.proposal_summary),
            "config_sha256": sha(a.config),
            "script_sha256": sha(Path(__file__).resolve()),
        },
        "governance": {
            "analysis_class": "post_test_matched_burden_audit",
            "test_labels_for_fitting_or_selection": False,
            "negative_results_preserved": True,
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    write(out / "paired_common_k_curve.tsv", curve_rows)
    write(out / "paired_tile_common_k20.tsv", tile_k20_rows)
    write(out / "global_equal_n_curve.tsv", global_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / "SHA256SUMS.tsv").open("w", encoding="utf-8", newline="") as f:
        f.write("path\tsha256\n")
        for path in sorted(out.iterdir()):
            if path.name != "SHA256SUMS.tsv":
                f.write(f"{path.name}\t{sha(path)}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
