#!/usr/bin/env python3
"""Decompose one-to-one matching, set coverage, merge, and duplicate effects."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weed-instances", type=int, default=1712)
    return p.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def edges(row: dict[str, str]) -> set[str]:
    return {value for value in row["exact_qualified_instance_ids"].split(";") if value}


def matching_count(rows: list[dict[str, str]]) -> int:
    owner: dict[str, str] = {}
    edge_map = {row["candidate_id"]: sorted(edges(row)) for row in rows}

    def augment(candidate_id: str, visited: set[str]) -> bool:
        for instance_id in edge_map[candidate_id]:
            if instance_id in visited:
                continue
            visited.add(instance_id)
            previous = owner.get(instance_id)
            if previous is None or augment(previous, visited):
                owner[instance_id] = candidate_id
                return True
        return False

    for candidate_id in sorted(edge_map):
        augment(candidate_id, set())
    return len(owner)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = read_tsv(args.items)
    if len(rows) != 518 or any(int(row["accepted_K20"]) != 1 for row in rows):
        raise ValueError("Expected the frozen 518-region K=20 queue")

    groups = {
        "all_selected": rows,
        "operator_consensus_reviewable": [row for row in rows if row["operator_consensus"] == "reviewable"],
        "mask_role_qualified": [row for row in rows if int(row["mask_role_qualified"]) == 1],
        "operator_and_mask": [row for row in rows if int(row["offline_dual_gate_accept"]) == 1],
    }
    summary_rows: list[dict[str, Any]] = []
    for name, group in groups.items():
        edge_sets = [edges(row) for row in group]
        union = set().union(*edge_sets) if edge_sets else set()
        memberships = sum(len(value) for value in edge_sets)
        matched = matching_count(group)
        summary_rows.append({
            "group": name,
            "reviewed_candidates": len(group),
            "candidates_with_no_qualified_instance": sum(len(value) == 0 for value in edge_sets),
            "single_instance_candidates": sum(len(value) == 1 for value in edge_sets),
            "multi_instance_candidates": sum(len(value) > 1 for value in edge_sets),
            "maximum_instances_in_one_candidate": max((len(value) for value in edge_sets), default=0),
            "qualified_edge_memberships": memberships,
            "distinct_set_covered_instances": len(union),
            "one_to_one_matched_instances": matched,
            "merge_capacity_gap_instances": len(union) - matched,
            "duplicate_memberships_beyond_set_union": memberships - len(union),
            "set_coverage_recall": len(union) / args.weed_instances,
            "one_to_one_recall": matched / args.weed_instances,
            "qualified_instances_per_review_upper_bound": memberships / max(1, len(group)),
        })

    multiplicity = Counter()
    human_by_multiplicity: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        count = len(edges(row))
        band = "0" if count == 0 else "1" if count == 1 else "2" if count == 2 else "3_plus"
        multiplicity[band] += 1
        human_by_multiplicity[band][row["operator_consensus"]] += 1
    multiplicity_rows: list[dict[str, Any]] = []
    for band in ("0", "1", "2", "3_plus"):
        total = multiplicity[band]
        reviewable = human_by_multiplicity[band]["reviewable"]
        multiplicity_rows.append({
            "qualified_instance_multiplicity": band,
            "candidates": total,
            "operator_consensus_reviewable": reviewable,
            "operator_consensus_reviewable_fraction": reviewable / max(1, total),
            "operator_consensus_not_reviewable": human_by_multiplicity[band]["not_reviewable"],
            "operator_unresolved": total - reviewable - human_by_multiplicity[band]["not_reviewable"],
        })

    write_tsv(output / "merge_aware_queue_summary.tsv", summary_rows)
    write_tsv(output / "reviewability_by_geometric_multiplicity.tsv", multiplicity_rows)
    summary = {
        "status": "completed_merge_aware_geometric_sensitivity",
        "analysis_class": "post_test_exploratory_operator_queue_audit",
        "truth_instances": args.weed_instances,
        "set_coverage_definition": "union_of_role_qualified_instance_edges_over_selected_candidates",
        "one_to_one_definition": "maximum_cardinality_candidate_instance_matching",
        "upper_bound_warning": "set coverage and edge memberships are geometric bounds, not observed per-plant operator output",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
