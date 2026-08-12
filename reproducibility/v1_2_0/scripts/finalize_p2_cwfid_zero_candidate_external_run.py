#!/usr/bin/env python3
"""Finalize a locked external run that produced no candidates.

This utility reads only already-written tabular outputs from the failed run. It
does not open dataset images, annotations, checkpoints, or model predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate an already-opened CWFID run without reopening data."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_run_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    required = [
        "paired_input_audit.tsv",
        "data_lineage.tsv",
        "annotation_semantics_audit.tsv",
        "checkpoint_audit.tsv",
        "per_image_seed_metrics.tsv",
    ]
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    if not args.source_log.is_file():
        raise FileNotFoundError(args.source_log)

    rows = read_tsv(source / "per_image_seed_metrics.tsv")
    seeds = sorted({int(row["seed"]) for row in rows})
    images = sorted({row["image_id"] for row in rows})
    if len(rows) != len(seeds) * len(images):
        raise ValueError("Incomplete seed-by-image result grid")
    if any(int(row["candidates"]) != 0 for row in rows):
        raise ValueError("This consolidator is restricted to zero-candidate runs")
    if any(int(row["accepted_candidates"]) != 0 for row in rows):
        raise ValueError("Accepted candidates must be zero")

    by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)

    seed_rows: list[dict[str, object]] = []
    for seed in seeds:
        group = by_seed[seed]
        weed_components = sum(int(row["weed_components"]) for row in group)
        cm = np.zeros((3, 3), dtype=np.int64)
        for row in group:
            for i in range(3):
                for j in range(3):
                    cm[i, j] += int(row[f"confusion_{i}{j}"])
        ious = []
        for cls in range(3):
            tp = int(cm[cls, cls])
            den = int(cm[cls, :].sum() + cm[:, cls].sum() - tp)
            ious.append(tp / den if den else 0.0)
        seed_rows.append(
            {
                "seed": seed,
                "images": len(group),
                "weed_components": weed_components,
                "candidates": 0,
                "accepted_candidates": 0,
                "all_candidate_precision": 0.0,
                "many_to_one_spatial_recall": 0.0,
                "many_to_one_role_qualified_recall": 0.0,
                "one_to_one_spatial_recall": 0.0,
                "one_to_one_role_qualified_recall": 0.0,
                "crop_overlap_image_frequency": 0.0,
                "background_iou": ious[0],
                "crop_iou": ious[1],
                "weed_iou": ious[2],
                "mean_iou": float(np.mean(ious)),
            }
        )

    # The primary endpoint is identically zero in every image and seed.
    rng = np.random.default_rng(args.bootstrap_seed)
    image_values = np.zeros(len(images), dtype=float)
    boot = np.empty(args.bootstrap_replicates, dtype=float)
    for idx in range(args.bootstrap_replicates):
        draw = rng.integers(0, len(images), size=len(images))
        boot[idx] = float(image_values[draw].mean())
    bootstrap_rows = [
        {
            "metric": "one_to_one_role_qualified_recall",
            "estimate": 0.0,
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "cluster_unit": "image",
            "clusters": len(images),
            "bootstrap_replicates": args.bootstrap_replicates,
        }
    ]

    output.mkdir(parents=True)
    write_tsv(output / "per_seed_metrics.tsv", seed_rows)
    write_tsv(output / "image_cluster_bootstrap.tsv", bootstrap_rows)
    source_audit = [
        {"path": name, "sha256": sha256(source / name), "bytes": (source / name).stat().st_size}
        for name in required
    ]
    source_audit.append(
        {
            "path": str(args.source_log),
            "sha256": sha256(args.source_log),
            "bytes": args.source_log.stat().st_size,
        }
    )
    write_tsv(output / "source_failure_audit.tsv", source_audit)
    (output / "candidate_diagnostics_schema.json").write_text(
        json.dumps(
            {
                "rows": 0,
                "reason": "No candidate was generated by any frozen checkpoint.",
                "columns": [
                    "seed",
                    "image_id",
                    "candidate_id",
                    "score",
                    "area_pixels",
                    "eligible_weed_candidate",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "run_id": args.run_id,
        "status": "completed_post_failure_consolidation_without_data_reopen",
        "source_failed_run": source.name,
        "external_data_or_annotations_reopened": False,
        "source_failure": "zero candidates caused the original TSV writer to reject an empty table",
        "images": len(images),
        "training_seeds": seeds,
        "seed_image_rows": len(rows),
        "unique_weed_components": int(seed_rows[0]["weed_components"]),
        "candidate_count_all_seeds": 0,
        "accepted_candidate_count_all_seeds": 0,
        "primary_endpoint_mean": 0.0,
        "primary_endpoint_image_cluster_ci95": [0.0, 0.0],
        "per_seed_metrics_identical": all(row == {**seed_rows[0], "seed": row["seed"]} for row in seed_rows),
        "semantic_iou": {
            "background": seed_rows[0]["background_iou"],
            "crop": seed_rows[0]["crop_iou"],
            "weed": seed_rows[0]["weed_iou"],
            "mean": seed_rows[0]["mean_iou"],
        },
        "governance": {
            "external_open_count_increment": 0,
            "policy_or_threshold_changed": False,
            "model_inference_repeated": False,
            "inputs_restricted_to_existing_tabular_outputs": True,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    manifests = []
    for path in sorted(output.iterdir()):
        if path.name != "SHA256SUMS.tsv" and path.is_file():
            manifests.append({"path": path.name, "sha256": sha256(path)})
    write_tsv(output / "SHA256SUMS.tsv", manifests)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
