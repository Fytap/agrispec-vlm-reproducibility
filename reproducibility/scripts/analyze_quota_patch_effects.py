#!/usr/bin/env python3
"""Summarize allocation effects within the two official-test patch proxies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-metrics", type=Path, required=True)
    parser.add_argument("--patch-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    scene_rows = read_tsv(args.scene_metrics.resolve())
    mapping_rows = read_tsv(args.patch_mapping.resolve())
    patch_by_file = {row["sample_id"].split(":")[-1]: row["patch_proxy_id"] for row in mapping_rows}
    if len(patch_by_file) != 26:
        raise ValueError(f"Expected 26 mapped scenes, found {len(patch_by_file)}")
    policies = ("min0_max20", "date_balanced_global", "min5_max50")
    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in scene_rows:
        if row["policy"] not in policies:
            continue
        grouped[(int(row["seed"]), row["policy"], patch_by_file[row["file"]])].append(row)
    patch_rows: list[dict[str, Any]] = []
    for (seed, policy, patch), rows in sorted(grouped.items()):
        accepted = sum(int(row["accepted"]) for row in rows)
        qualified = sum(int(row["qualified"]) for row in rows)
        weed = sum(int(row["weed_instances"]) for row in rows)
        one = sum(int(row["one"]) for row in rows)
        patch_rows.append(
            {
                "seed": seed,
                "policy": policy,
                "patch_proxy": patch,
                "scenes": len(rows),
                "accepted_candidates": accepted,
                "candidate_level_role_qualified_precision": qualified / accepted,
                "one_to_one_role_qualified_recall": one / weed,
                "crop_overlap_scene_frequency": sum(int(row["crop_scene"]) for row in rows) / len(rows),
            }
        )
    effects: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in patch_rows}):
        for patch in sorted({row["patch_proxy"] for row in patch_rows}):
            index = {
                row["policy"]: row
                for row in patch_rows
                if int(row["seed"]) == seed and row["patch_proxy"] == patch
            }
            fixed = index["min0_max20"]
            for policy in ("date_balanced_global", "min5_max50"):
                current = index[policy]
                effects.append(
                    {
                        "seed": seed,
                        "patch_proxy": patch,
                        "policy": policy,
                        "precision_difference_vs_fixed": float(current["candidate_level_role_qualified_precision"]) - float(fixed["candidate_level_role_qualified_precision"]),
                        "one_to_one_recall_difference_vs_fixed": float(current["one_to_one_role_qualified_recall"]) - float(fixed["one_to_one_role_qualified_recall"]),
                        "crop_scene_frequency_difference_vs_fixed": float(current["crop_overlap_scene_frequency"]) - float(fixed["crop_overlap_scene_frequency"]),
                    }
                )
    aggregate: list[dict[str, Any]] = []
    for policy in ("date_balanced_global", "min5_max50"):
        for patch in sorted({row["patch_proxy"] for row in effects}):
            group = [row for row in effects if row["policy"] == policy and row["patch_proxy"] == patch]
            for metric in (
                "precision_difference_vs_fixed",
                "one_to_one_recall_difference_vs_fixed",
                "crop_scene_frequency_difference_vs_fixed",
            ):
                values = np.asarray([float(row[metric]) for row in group])
                aggregate.append(
                    {
                        "policy": policy,
                        "patch_proxy": patch,
                        "metric": metric,
                        "n_training_seeds": len(values),
                        "mean": float(values.mean()),
                        "sample_sd": float(values.std(ddof=1)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                        "positive_seeds": int((values > 0).sum()),
                        "zero_seeds": int((values == 0).sum()),
                        "negative_seeds": int((values < 0).sum()),
                    }
                )
    write_tsv(output / "per_seed_patch_metrics.tsv", patch_rows)
    write_tsv(output / "per_seed_patch_effects.tsv", effects)
    write_tsv(output / "patch_effect_aggregate.tsv", aggregate)
    summary = {
        "analysis": "two-patch-proxy allocation effect audit",
        "status": "completed",
        "analysis_class": "descriptive post-test spatial-stratum audit",
        "patches": sorted(set(patch_by_file.values())),
        "boundary": "the two capture-index groups are official-test patch proxies, not independent fields or farms",
        "inputs": {
            "scene_metrics_sha256": sha256(args.scene_metrics.resolve()),
            "patch_mapping_sha256": sha256(args.patch_mapping.resolve()),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    checksums = []
    for path in sorted(output.iterdir()):
        if path.name != "SHA256SUMS.tsv":
            checksums.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_tsv(output / "SHA256SUMS.tsv", checksums)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
