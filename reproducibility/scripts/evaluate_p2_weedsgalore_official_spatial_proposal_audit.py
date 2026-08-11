#!/usr/bin/env python3
"""Official spatial-split proposal commissioning and contract audit.

This CPU-only analysis replaces acquisition-date LODO as the primary split.
The proposal setting is selected from the official training split only.  The
official validation and test splits are never used for setting selection.
Target masks are used for training-split setting selection and offline scoring;
they never enter proposal generation or candidate features.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_commissioning_contract_audit import (  # noqa: E402
    CONTRACT_GRID,
    DATES,
    IOU_GRID,
    SM_GRID,
    audit_setting,
    load_truth,
)
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import (  # noqa: E402
    read_tsv,
    sha256,
)


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("project-root", "archive", "manifest", "grid-root", "grid-summary", "original-root", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-grid-summary-sha256", required=True)
    parser.add_argument("--expected-original-candidates-sha256", required=True)
    parser.add_argument("--maximum-training-candidates-per-image", type=float, default=100.0)
    return parser.parse_args()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def official_split_map(archive_path: Path, manifest: list[dict[str, str]]) -> dict[str, str]:
    by_stem: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for split in SPLITS:
            values = archive.read(f"weedsgalore-dataset/splits/{split}.txt").decode("utf-8").splitlines()
            expected = {"train": 104, "val": 26, "test": 26}[split]
            if len(values) != expected or len(values) != len(set(values)):
                raise ValueError(f"Unexpected official {split} split structure")
            for stem in values:
                if stem in by_stem:
                    raise ValueError(f"Official split overlap: {stem}")
                by_stem[stem] = split
    sample_map: dict[str, str] = {}
    for row in manifest:
        stem = Path(row["semantic_relpath"]).stem
        if stem not in by_stem:
            raise ValueError(f"Manifest sample absent from official split: {stem}")
        sample_map[row["sample_id"]] = by_stem[stem]
    if len(sample_map) != 156 or Counter(sample_map.values()) != Counter({"train": 104, "val": 26, "test": 26}):
        raise ValueError("Official split coverage mismatch")
    return sample_map


def aggregate_proposals(variant: str, setting: str, audit: dict[str, object], split_map: dict[str, str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_key: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for split in SPLITS:
        for date in (*DATES, "ALL"):
            for role in ("crop", "weed"):
                by_key[(split, date, role)] = []
    for row in audit["instances"]:
        split = split_map[str(row["sample_id"])]
        role = str(row["truth_role"])
        by_key[(split, str(row["date"]), role)].append(row)
        by_key[(split, "ALL", role)].append(row)

    proposal_rows: list[dict[str, object]] = []
    for (split, date, role), rows in by_key.items():
        if not rows:
            continue
        record: dict[str, object] = {
            "variant": variant,
            "setting": setting,
            "official_split": split,
            "acquisition_date": date,
            "truth_role": role,
            "truth_instances": len(rows),
            "spatial_recalled_instances": sum(int(row["spatial_recalled"]) for row in rows),
            "spatial_proposal_recall": float(np.mean([int(row["spatial_recalled"]) for row in rows])),
            "role_qualified_recalled_instances": sum(int(row["role_qualified_recalled"]) for row in rows),
            "role_qualified_proposal_recall": float(np.mean([int(row["role_qualified_recalled"]) for row in rows])),
        }
        for threshold in IOU_GRID:
            record[f"proposal_ar_iou_{threshold:.2f}"] = float(np.mean([float(row["best_iou"]) >= threshold for row in rows]))
        proposal_rows.append(record)

    candidate_counts: Counter[tuple[str, str]] = Counter()
    role_counts: Counter[tuple[str, str, str]] = Counter()
    image_ids: dict[tuple[str, str], set[str]] = {}
    for row in audit["rows"]:
        split = split_map[str(row["file"])]
        date = str(row["date"])
        for group in (date, "ALL"):
            candidate_counts[(split, group)] += 1
            role_counts[(split, group, str(row["truth"]))] += 1
            image_ids.setdefault((split, group), set()).add(str(row["file"]))
    burden_rows = []
    for (split, date), count in sorted(candidate_counts.items()):
        images = len(image_ids[(split, date)])
        burden_rows.append({
            "variant": variant,
            "setting": setting,
            "official_split": split,
            "acquisition_date": date,
            "images": images,
            "candidates": count,
            "candidates_per_image": count / images,
            "eligible_weed_candidates": role_counts[(split, date, "weed")],
            "eligible_crop_candidates": role_counts[(split, date, "crop")],
            "ambiguous_or_background_candidates": role_counts[(split, date, "ambiguous_or_background")],
        })
    return proposal_rows, burden_rows


def sensitivity_rows(variant: str, setting: str, audit: dict[str, object], split_map: dict[str, str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    contract: list[dict[str, object]] = []
    split_merge: list[dict[str, object]] = []
    for split in SPLITS:
        instances = [row for row in audit["instances"] if split_map[str(row["sample_id"])] == split]
        matches = [row for row in audit["matches"] if split_map[str(row["sample_id"])] == split]
        ids_by_role = {role: {str(row["instance_id"]) for row in instances if row["truth_role"] == role} for role in ("crop", "weed")}
        for instance_coverage in CONTRACT_GRID:
            for labeled_coverage in CONTRACT_GRID:
                for purity in (0.50, 0.75, 0.90):
                    hit = {str(row["instance_id"]) for row in matches if float(row["instance_coverage"]) >= instance_coverage and float(row["candidate_labeled_coverage"]) >= labeled_coverage and float(row["same_role_purity"]) >= purity}
                    for role in ("crop", "weed"):
                        denominator = ids_by_role[role]
                        contract.append({
                            "variant": variant, "setting": setting, "official_split": split,
                            "truth_role": role, "minimum_instance_coverage": instance_coverage,
                            "minimum_candidate_labeled_coverage": labeled_coverage,
                            "minimum_same_role_purity": purity, "truth_instances": len(denominator),
                            "recalled_instances": len(hit & denominator),
                            "role_qualified_proposal_recall": len(hit & denominator) / max(1, len(denominator)),
                        })
        for threshold in SM_GRID:
            per_instance: Counter[str] = Counter()
            per_candidate: dict[str, set[str]] = {}
            for row in matches:
                if float(row["instance_coverage"]) >= threshold:
                    iid, cid = str(row["instance_id"]), str(row["candidate_id"])
                    per_instance[iid] += 1
                    per_candidate.setdefault(cid, set()).add(iid)
            split_merge.append({
                "variant": variant, "setting": setting, "official_split": split,
                "instance_coverage_threshold": threshold,
                "truth_instances": len(instances),
                "split_instances": sum(value >= 2 for value in per_instance.values()),
                "candidates_with_two_or_more_instances": sum(len(value) >= 2 for value in per_candidate.values()),
                "maximum_candidates_per_instance": max(per_instance.values(), default=0),
                "maximum_instances_per_candidate": max((len(value) for value in per_candidate.values()), default=0),
            })
    return contract, split_merge


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError("This audit is CPU-only")
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    checks = (
        (args.archive, args.expected_archive_sha256),
        (args.manifest, args.expected_manifest_sha256),
        (args.grid_summary, args.expected_grid_summary_sha256),
        (args.original_root / "candidate_components.tsv", args.expected_original_candidates_sha256),
    )
    for path, expected in checks:
        if sha256(path.resolve()) != expected:
            raise ValueError(f"Input hash mismatch: {path}")

    started = time.perf_counter()
    manifest = read_tsv(args.manifest.resolve())
    split_map = official_split_map(args.archive.resolve(), manifest)
    truth = load_truth(args.archive.resolve(), manifest)
    grid = json.loads(args.grid_summary.read_text(encoding="utf-8"))
    records = {row["setting"]: row for row in grid["settings"] if row["return_code"] == 0}
    audits: dict[str, dict[str, object]] = {}
    selection_rows: list[dict[str, object]] = []
    for setting, record in sorted(records.items()):
        audit = audit_setting((args.grid_root / setting).resolve(), manifest, truth)
        audits[setting] = audit
        proposal, burden = aggregate_proposals("candidate_setting", setting, audit, split_map)
        weed = next(row for row in proposal if row["official_split"] == "train" and row["acquisition_date"] == "ALL" and row["truth_role"] == "weed")
        load = next(row for row in burden if row["official_split"] == "train" and row["acquisition_date"] == "ALL")
        selection_rows.append({
            "setting": setting,
            "threshold": record["threshold"],
            "minimum_area_pixels": record["minimum_area"],
            "training_images": load["images"],
            "training_candidates": load["candidates"],
            "training_candidates_per_image": load["candidates_per_image"],
            "training_weed_instances": weed["truth_instances"],
            "training_spatial_weed_proposal_recall": weed["spatial_proposal_recall"],
            "training_role_qualified_weed_proposal_recall": weed["role_qualified_proposal_recall"],
            "feasible_candidate_burden": int(float(load["candidates_per_image"]) <= args.maximum_training_candidates_per_image),
        })
    feasible = [row for row in selection_rows if int(row["feasible_candidate_burden"])]
    if not feasible:
        raise ValueError("No proposal setting satisfies the training candidate burden")
    selected = max(feasible, key=lambda row: (
        float(row["training_role_qualified_weed_proposal_recall"]),
        -float(row["training_candidates_per_image"]),
        float(row["threshold"]),
        int(row["minimum_area_pixels"]),
    ))

    original = audit_setting(args.original_root.resolve(), manifest, truth)
    commissioned = audits[str(selected["setting"])]
    proposal_rows: list[dict[str, object]] = []
    burden_rows: list[dict[str, object]] = []
    contract_rows: list[dict[str, object]] = []
    split_merge_rows: list[dict[str, object]] = []
    for variant, setting, audit in (
        ("original_T095_A032", "T095_A032", original),
        ("training_selected_commissioned", str(selected["setting"]), commissioned),
    ):
        proposals, burdens = aggregate_proposals(variant, setting, audit, split_map)
        contracts, split_merges = sensitivity_rows(variant, setting, audit, split_map)
        proposal_rows.extend(proposals)
        burden_rows.extend(burdens)
        contract_rows.extend(contracts)
        split_merge_rows.extend(split_merges)

    excluded_rows = []
    for sample_id, item in truth.items():
        instance_mask = item["instances"]
        semantic = item["semantic"]
        for instance_id in (int(value) for value in np.unique(instance_mask) if int(value) > 0):
            pixels = semantic[instance_mask == instance_id]
            crop = int((pixels == 1).sum())
            weed = int((pixels > 1).sum())
            if crop + weed == 0:
                excluded_rows.append({
                    "sample_id": sample_id,
                    "official_split": split_map[sample_id],
                    "instance_label": instance_id,
                    "instance_id": f"{sample_id}:instance:{instance_id}",
                    "instance_pixels": int(len(pixels)),
                    "crop_semantic_pixels": crop,
                    "weed_semantic_pixels": weed,
                    "exclusion_reason": "instance mask has zero crop/weed semantic pixels",
                })

    output.mkdir(parents=True)
    spatial_manifest_rows = [{
        "sample_id": row["sample_id"],
        "tile_stem": Path(row["semantic_relpath"]).stem,
        "acquisition_date": row["provisional_source_session_id"],
        "official_split": split_map[row["sample_id"]],
    } for row in manifest]
    paths = {
        "spatial_manifest": output / "official_spatial_manifest.tsv",
        "setting_selection": output / "training_only_setting_selection.tsv",
        "proposal_layers": output / "official_split_proposal_layers.tsv",
        "candidate_burden": output / "official_split_candidate_burden.tsv",
        "matching_sensitivity": output / "official_split_matching_contract_sensitivity.tsv",
        "split_merge": output / "official_split_split_merge_sensitivity.tsv",
        "excluded_instances": output / "semantic_instance_inconsistencies.tsv",
    }
    for key, rows in (
        ("spatial_manifest", spatial_manifest_rows),
        ("setting_selection", selection_rows),
        ("proposal_layers", proposal_rows),
        ("candidate_burden", burden_rows),
        ("matching_sensitivity", contract_rows),
        ("split_merge", split_merge_rows),
    ):
        write_tsv(paths[key], rows)
    if excluded_rows:
        write_tsv(paths["excluded_instances"], excluded_rows)
    else:
        paths.pop("excluded_instances")

    summary = {
        "run_id": args.run_id,
        "status": "completed_official_spatial_proposal_audit",
        "scope": {
            "public_data_only": True,
            "sealed_test_read": False,
            "proposal_generation_reads_target_truth": False,
            "training_split_truth_used_for_setting_selection": True,
            "validation_and_test_used_for_setting_selection": False,
            "official_spatial_split_primary": True,
            "formal_risk_certificate": False,
        },
        "official_split_counts": dict(Counter(split_map.values())),
        "setting_selection": {
            "selected_setting": selected["setting"],
            "objective": "maximum official-training role-qualified weed-instance proposal recall",
            "maximum_training_candidates_per_image": args.maximum_training_candidates_per_image,
            "tie_break": "lower candidate burden, then higher foreground threshold, then larger minimum area",
            "selected_record": selected,
        },
        "matching_contract": {
            "spatial_proposal_recall": "candidate covers at least 50% of a truth instance, ignoring candidate role",
            "role_qualified_proposal_recall": "spatial rule plus at least 50% candidate labeled coverage and at least 90% same-role purity",
            "split_merge_sensitivity_thresholds": list(SM_GRID),
        },
        "dataset_accounting": {
            "reported_crop_instances": 2169,
            "reported_weed_instances": 10031,
            "evaluated_crop_instances": sum(row["truth_role"] == "crop" for item in (commissioned["instances"],) for row in item),
            "evaluated_weed_instances": sum(row["truth_role"] == "weed" for item in (commissioned["instances"],) for row in item),
            "excluded_semantic_instance_inconsistencies": len(excluded_rows),
        },
        "inputs": {"archive_sha256": sha256(args.archive.resolve()), "manifest_sha256": sha256(args.manifest.resolve()), "grid_summary_sha256": sha256(args.grid_summary.resolve())},
        "outputs": {key + "_sha256": sha256(path) for key, path in paths.items()},
        "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version, "numpy": np.__version__},
        "script_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
