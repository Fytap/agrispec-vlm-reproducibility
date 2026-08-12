#!/usr/bin/env python3
"""Replay the compact evidence tables for release v1.1.0.

This entry point intentionally uses only the Python standard library.  It reads
the immutable derived tables distributed with the release, checks the key row
counts and numerical invariants reported in the manuscript, and writes a small
set of publication-facing tables plus a SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


RESULTS = Path("results/p2_development")
PROPOSAL_RUN = "P2_WEEDSGALORE_OFFICIAL_SPATIAL_PROPOSAL_AUDIT_20260810_v1"
QUOTA_RUN = "P2_WEEDSGALORE_TARGET_SEMANTIC_QUOTA_ROBUSTNESS_EXACT_20260812_v2"
PATCH_RUN = "P2_WEEDSGALORE_TARGET_SEMANTIC_PATCH_EFFECTS_20260812_v1"
MASK_RUN = "P2_WEEDSGALORE_TARGET_SEMANTIC_EXACT_MASK_AUDIT_20260812_v1"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return abs(actual - expected) <= tolerance


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Reproducibility directory containing results/ (default: inferred).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = root / RESULTS

    proposal = read_tsv(results / PROPOSAL_RUN / "training_only_setting_selection.tsv")
    quota = read_tsv(results / QUOTA_RUN / "test_quota_grid_aggregate.tsv")
    selected = read_tsv(
        results / QUOTA_RUN / "validation_selected_robust_policy_per_seed.tsv"
    )
    patch = read_tsv(results / PATCH_RUN / "patch_effect_aggregate.tsv")
    with (results / MASK_RUN / "summary.json").open("r", encoding="utf-8") as handle:
        mask = json.load(handle)

    require(len(proposal) == 12, f"Expected 12 proposal settings, found {len(proposal)}")
    policies = {row["policy"] for row in quota}
    expected_quota = {
        f"min{minimum}_max{maximum}"
        for minimum in (0, 5, 10, 20)
        for maximum in (20, 30, 50, "unrestricted")
    }
    require(expected_quota.issubset(policies), "The complete 4x4 quota grid is absent")
    require("date_balanced_global" in policies, "Date-balanced comparator is absent")
    require(len(selected) == 5, f"Expected five H200 seeds, found {len(selected)}")

    expected_means = {
        ("min0_max20", "candidate_level_role_qualified_precision"): 0.7168327,
        ("min0_max20", "one_to_one_role_qualified_recall"): 0.2026869,
        ("min0_max20", "worst_date_one_to_one_recall"): 0.1100186,
        ("date_balanced_global", "candidate_level_role_qualified_precision"): 0.7710074,
        ("date_balanced_global", "one_to_one_role_qualified_recall"): 0.2191589,
        ("date_balanced_global", "worst_date_one_to_one_recall"): 0.1162801,
        ("min5_max50", "candidate_level_role_qualified_precision"): 0.8938717,
        ("min5_max50", "one_to_one_role_qualified_recall"): 0.2578271,
        ("min5_max50", "worst_date_one_to_one_recall"): 0.0473098,
    }
    indexed = {(row["policy"], row["metric"]): float(row["mean"]) for row in quota}
    for key, expected in expected_means.items():
        require(key in indexed, f"Missing quota summary row: {key}")
        require(close(indexed[key], expected, 5e-8), f"Unexpected mean for {key}: {indexed[key]}")

    selected_counts: dict[str, int] = {}
    for row in selected:
        name = row["validation_selected_policy"]
        selected_counts[name] = selected_counts.get(name, 0) + 1
    require(selected_counts == {"min0_max20": 4, "min0_max30": 1},
            f"Unexpected validation selections: {selected_counts}")

    patch_index = {
        (row["policy"], row["patch_proxy"], row["metric"]): row for row in patch
    }
    p0 = float(patch_index[("date_balanced_global", "test_capture_index_patch_proxy_0",
                            "one_to_one_recall_difference_vs_fixed")]["mean"])
    p1 = float(patch_index[("date_balanced_global", "test_capture_index_patch_proxy_1",
                            "one_to_one_recall_difference_vs_fixed")]["mean"])
    require(p0 < 0 < p1, "Expected opposite patch-proxy recall effects")

    exact = mask["exact_replay"]
    require(exact["candidate_id_sets_identical"], "Candidate IDs differ in exact replay")
    require(exact["fixed_scene_metrics_identical"], "Fixed-scene metrics differ in exact replay")
    require(exact["exact_edge_pairs"] == 453, "Unexpected exact edge-pair count")
    require(exact["box_only_edge_pairs"] == 4, "Unexpected box-only edge count")

    write_tsv(output / "proposal_grid.tsv", proposal)
    chosen_metrics = {
        "candidate_level_role_qualified_precision",
        "one_to_one_role_qualified_recall",
        "worst_date_one_to_one_recall",
        "crop_overlap_scene_frequency",
    }
    chosen_policies = {"min0_max20", "date_balanced_global", "min5_max50"}
    quota_summary = [
        row for row in quota
        if row["policy"] in chosen_policies and row["metric"] in chosen_metrics
    ]
    write_tsv(output / "quota_summary.tsv", quota_summary)
    write_tsv(output / "validation_selection.tsv", selected)
    write_tsv(output / "patch_summary.tsv", patch)

    evidence = {
        "release": "v1.1.0",
        "status": "verified",
        "proposal_settings": len(proposal),
        "quota_grid_policies": 16,
        "comparison_policies": sorted(policies - expected_quota),
        "h200_seeds": len(selected),
        "validation_selected_policy_counts": selected_counts,
        "date_balanced_patch_recall_difference": {"patch_proxy_0": p0, "patch_proxy_1": p1},
        "exact_mask_replay": exact,
    }
    evidence_path = output / "evidence_summary.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_rows = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "SHA256SUMS.tsv":
            manifest_rows.append({"sha256": sha256(path), "file": path.name})
    write_tsv(output / "SHA256SUMS.tsv", manifest_rows)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
