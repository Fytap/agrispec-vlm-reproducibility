#!/usr/bin/env python3
"""Audit queue-quota sensitivity with validation selection and exact masks."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MINIMUMS = (0, 5, 10, 20)
MAXIMUMS: tuple[int | None, ...] = (20, 30, 50, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("target_semantic_trainer", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixed_k(rows: list[dict[str, Any]], k: int = 20) -> list[dict[str, Any]]:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[str(row["file"])].append(row)
    selected: list[dict[str, Any]] = []
    for stem in sorted(by_file):
        selected.extend(sorted(by_file[stem], key=lambda row: (-float(row["score"]), str(row["candidate_id"])))[:k])
    return selected


def quota_allocate(
    rows: list[dict[str, Any]], budget: int, minimum: int, maximum: int | None
) -> list[dict[str, Any]]:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_file[str(row["file"])].append(row)
    for group in by_file.values():
        group.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
    selected = [row for stem in sorted(by_file) for row in by_file[stem][:minimum]]
    if len(selected) > budget:
        raise ValueError(f"Minimum quota {minimum} exceeds budget {budget}")
    selected_ids = {str(row["candidate_id"]) for row in selected}
    remainder: list[dict[str, Any]] = []
    for stem in sorted(by_file):
        limit = len(by_file[stem]) if maximum is None else maximum
        remainder.extend(
            row for row in by_file[stem][minimum:limit] if str(row["candidate_id"]) not in selected_ids
        )
    remainder.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
    needed = budget - len(selected)
    if needed > len(remainder):
        raise ValueError(f"Quota [{minimum}, {maximum}] cannot realize budget {budget}")
    return selected + remainder[:needed]


def date_balanced_allocate(rows: list[dict[str, Any]], reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    budgets: dict[str, int] = defaultdict(int)
    for row in reference:
        budgets[str(row["file"])[:10]] += 1
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[str(row["file"])[:10]].append(row)
    selected: list[dict[str, Any]] = []
    for date in sorted(by_date):
        group = sorted(by_date[date], key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
        selected.extend(group[: budgets[date]])
    if len(selected) != len(reference):
        raise ValueError("Date-balanced allocation did not preserve the fixed total")
    return selected


def truth_records(module, stem: str, dataset_root: Path) -> list[Any]:
    values = module.truth_instances(stem, dataset_root)
    return values[-1]


def regenerate_split(module, model, stems: list[str], dataset_root: Path, device, threshold: float, area: int):
    probabilities, _, _ = module.predict(model, stems, dataset_root, device)
    rows: list[dict[str, Any]] = []
    edges: dict[str, set[str]] = {}
    for stem in stems:
        candidates, _, qualified_edges, _, _ = module.component_candidates(
            stem, probabilities[stem], dataset_root, threshold, area
        )
        for row in candidates:
            candidate_id = str(row["candidate_id"])
            if candidate_id in edges:
                raise ValueError(f"Duplicate candidate ID {candidate_id}")
            rows.append(row)
            edges[candidate_id] = set(qualified_edges.get(candidate_id, set()))
    del probabilities
    return rows, edges


def evaluate(
    module,
    selected: list[dict[str, Any]],
    stems: list[str],
    dataset_root: Path,
    edges: dict[str, set[str]],
    scene_output: list[dict[str, Any]] | None = None,
    scene_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_file[str(row["file"])].append(row)
    scene_rows: list[dict[str, Any]] = []
    for stem in sorted(stems):
        records = truth_records(module, stem, dataset_root)
        weed_n = sum(record.role == "weed" for record in records)
        rows = by_file.get(stem, [])
        qualified = [row for row in rows if str(row["candidate_truth"]) == "weed"]
        local_edges = {str(row["candidate_id"]): edges[str(row["candidate_id"])] for row in qualified}
        many = {instance_id for values in local_edges.values() for instance_id in values}
        one = int(module.one_to_one_count(qualified, local_edges))
        scene_rows.append(
            {
                "file": stem,
                "date": stem[:10],
                "weed_instances": weed_n,
                "accepted": len(rows),
                "qualified": len(qualified),
                "many": len(many),
                "one": one,
                "crop_scene": int(any(float(row["crop_fraction_candidate"]) > 0 for row in rows)),
            }
        )
    total_weed = sum(row["weed_instances"] for row in scene_rows)
    accepted = sum(row["accepted"] for row in scene_rows)
    qualified = sum(row["qualified"] for row in scene_rows)
    many = sum(row["many"] for row in scene_rows)
    one = sum(row["one"] for row in scene_rows)
    date_metrics = []
    for date in sorted({row["date"] for row in scene_rows}):
        group = [row for row in scene_rows if row["date"] == date]
        d_weed = sum(row["weed_instances"] for row in group)
        d_accepted = sum(row["accepted"] for row in group)
        date_metrics.append(
            {
                "precision": sum(row["qualified"] for row in group) / max(1, d_accepted),
                "one": sum(row["one"] for row in group) / max(1, d_weed),
            }
        )
    counts = [row["accepted"] for row in scene_rows]
    if scene_output is not None:
        metadata = {} if scene_metadata is None else scene_metadata
        scene_output.extend({**metadata, **row} for row in scene_rows)
    return {
        "accepted_candidates": accepted,
        "candidate_level_role_qualified_precision": qualified / max(1, accepted),
        "many_to_one_role_qualified_recall": many / max(1, total_weed),
        "one_to_one_role_qualified_recall": one / max(1, total_weed),
        "date_macro_one_to_one_recall": float(np.mean([row["one"] for row in date_metrics])),
        "worst_date_one_to_one_recall": float(np.min([row["one"] for row in date_metrics])),
        "worst_date_precision": float(np.min([row["precision"] for row in date_metrics])),
        "crop_overlap_scene_frequency": sum(row["crop_scene"] for row in scene_rows) / len(scene_rows),
        "minimum_candidates_per_tile": min(counts),
        "maximum_candidates_per_tile": max(counts),
    }


def quota_label(minimum: int, maximum: int | None) -> str:
    return f"min{minimum}_max{'unrestricted' if maximum is None else maximum}"


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    trainer = args.trainer.resolve()
    dataset_root = args.dataset_root.resolve()
    module = load_module(trainer)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact component replay")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    validation_rows_out: list[dict[str, Any]] = []
    test_rows_out: list[dict[str, Any]] = []
    selected_rows_out: list[dict[str, Any]] = []
    test_scene_rows_out: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []

    for raw_run_dir in args.run_dir:
        run_dir = raw_run_dir.resolve()
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        seed = int(summary["training"]["seed"])
        threshold = float(summary["selected_on_validation"]["threshold"])
        area = int(summary["selected_on_validation"]["minimum_area_pixels"])
        checkpoint = torch.load(run_dir / "best_validation_queue.pt", map_location=device, weights_only=False)
        model = module.build_model().to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        val_stems = module.read_split(dataset_root, "val")
        test_stems = module.read_split(dataset_root, "test")
        val_rows, val_edges = regenerate_split(module, model, val_stems, dataset_root, device, threshold, area)
        test_rows, test_edges = regenerate_split(module, model, test_stems, dataset_root, device, threshold, area)
        archived_rows = read_tsv(run_dir / "official_test_candidates.tsv")
        if {row["candidate_id"] for row in archived_rows} != {str(row["candidate_id"]) for row in test_rows}:
            raise ValueError(f"Exact test candidates differ for seed {seed}")
        fixed_val = fixed_k(val_rows)
        fixed_test = [row for row in archived_rows if int(row["accepted_K20"]) == 1]
        fixed_val_metrics = evaluate(module, fixed_val, val_stems, dataset_root, val_edges)

        grid_validation: list[dict[str, Any]] = []
        for minimum in MINIMUMS:
            for maximum in MAXIMUMS:
                if maximum is not None and minimum > maximum:
                    continue
                label = quota_label(minimum, maximum)
                val_selected = quota_allocate(val_rows, len(fixed_val), minimum, maximum)
                val_metrics = evaluate(module, val_selected, val_stems, dataset_root, val_edges)
                val_record = {
                    "seed": seed,
                    "policy": label,
                    "minimum_quota": minimum,
                    "maximum_quota": "unrestricted" if maximum is None else maximum,
                    **val_metrics,
                }
                validation_rows_out.append(val_record)
                grid_validation.append(val_record)
                test_selected = quota_allocate(test_rows, len(fixed_test), minimum, maximum)
                test_metrics = evaluate(
                    module,
                    test_selected,
                    test_stems,
                    dataset_root,
                    test_edges,
                    scene_output=test_scene_rows_out,
                    scene_metadata={"seed": seed, "policy": label},
                )
                test_rows_out.append(
                    {
                        "seed": seed,
                        "policy": label,
                        "minimum_quota": minimum,
                        "maximum_quota": "unrestricted" if maximum is None else maximum,
                        **test_metrics,
                    }
                )

        precision_floor = float(fixed_val_metrics["candidate_level_role_qualified_precision"])
        feasible = [
            row for row in grid_validation
            if float(row["candidate_level_role_qualified_precision"]) + 1e-12 >= precision_floor
        ]
        robust = max(
            feasible,
            key=lambda row: (
                float(row["worst_date_one_to_one_recall"]),
                float(row["one_to_one_role_qualified_recall"]),
                float(row["candidate_level_role_qualified_precision"]),
                -int(row["minimum_quota"]),
                -999999 if row["maximum_quota"] == "unrestricted" else -int(row["maximum_quota"]),
            ),
        )
        robust_test = next(
            row for row in test_rows_out if int(row["seed"]) == seed and row["policy"] == robust["policy"]
        )
        date_balanced_val = date_balanced_allocate(val_rows, fixed_val)
        date_balanced_test = date_balanced_allocate(test_rows, fixed_test)
        date_val_metrics = evaluate(module, date_balanced_val, val_stems, dataset_root, val_edges)
        date_test_metrics = evaluate(
            module,
            date_balanced_test,
            test_stems,
            dataset_root,
            test_edges,
            scene_output=test_scene_rows_out,
            scene_metadata={"seed": seed, "policy": "date_balanced_global"},
        )
        validation_rows_out.append({"seed": seed, "policy": "date_balanced_global", "minimum_quota": "", "maximum_quota": "", **date_val_metrics})
        test_rows_out.append({"seed": seed, "policy": "date_balanced_global", "minimum_quota": "", "maximum_quota": "", **date_test_metrics})
        selected_rows_out.append(
            {
                "seed": seed,
                "validation_precision_floor": precision_floor,
                "validation_selected_policy": robust["policy"],
                "validation_worst_date_one_to_one_recall": robust["worst_date_one_to_one_recall"],
                "validation_pooled_one_to_one_recall": robust["one_to_one_role_qualified_recall"],
                "test_candidate_level_role_qualified_precision": robust_test["candidate_level_role_qualified_precision"],
                "test_one_to_one_role_qualified_recall": robust_test["one_to_one_role_qualified_recall"],
                "test_worst_date_one_to_one_recall": robust_test["worst_date_one_to_one_recall"],
                "date_balanced_test_precision": date_test_metrics["candidate_level_role_qualified_precision"],
                "date_balanced_test_one_to_one_recall": date_test_metrics["one_to_one_role_qualified_recall"],
                "date_balanced_test_worst_date_one_to_one_recall": date_test_metrics["worst_date_one_to_one_recall"],
            }
        )
        input_rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "summary_sha256": sha256(run_dir / "summary.json"),
                "checkpoint_sha256": sha256(run_dir / "best_validation_queue.pt"),
                "candidate_sha256": sha256(run_dir / "official_test_candidates.tsv"),
            }
        )
        del model, checkpoint, val_rows, val_edges, test_rows, test_edges
        torch.cuda.empty_cache()
        gc.collect()

    observed = sorted(int(row["seed"]) for row in input_rows)
    expected = [20260810, 20260811, 20260812, 20260813, 20260814]
    if observed != expected:
        raise ValueError(f"Expected seeds {expected}, found {observed}")

    aggregate_rows: list[dict[str, Any]] = []
    policies = sorted({str(row["policy"]) for row in test_rows_out})
    metrics = (
        "candidate_level_role_qualified_precision",
        "one_to_one_role_qualified_recall",
        "date_macro_one_to_one_recall",
        "worst_date_one_to_one_recall",
        "crop_overlap_scene_frequency",
    )
    for policy in policies:
        group = [row for row in test_rows_out if row["policy"] == policy]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            aggregate_rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )

    write_tsv(output / "validation_quota_grid_per_seed.tsv", validation_rows_out)
    write_tsv(output / "test_quota_grid_per_seed.tsv", test_rows_out)
    write_tsv(output / "test_quota_grid_per_seed_scene.tsv", test_scene_rows_out)
    write_tsv(output / "validation_selected_robust_policy_per_seed.tsv", selected_rows_out)
    write_tsv(output / "test_quota_grid_aggregate.tsv", aggregate_rows)
    write_tsv(output / "input_manifest.tsv", input_rows)
    summary_out = {
        "analysis": "exact-mask quota-grid and date-balanced allocation audit",
        "status": "completed",
        "analysis_class": "post-test exploratory robustness analysis",
        "seeds": observed,
        "seed_identifier_note": "numeric values are random-seed identifiers, not execution dates",
        "quota_grid": {
            "minimum": list(MINIMUMS),
            "maximum": [value if value is not None else "unrestricted" for value in MAXIMUMS],
        },
        "validation_selection_rule": (
            "among quota policies with validation candidate-level role-qualified precision no lower than fixed K=20, "
            "maximize validation worst-date one-to-one recall, then pooled one-to-one recall and precision"
        ),
        "date_balanced_baseline": "preserve fixed-K candidate count within each acquisition date and rank globally inside each date",
        "matching": "binary component masks and exact qualified candidate-instance edges regenerated from every checkpoint",
        "inputs": input_rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary_out, indent=2) + "\n", encoding="utf-8")
    checksums = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.tsv":
            checksums.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_tsv(output / "SHA256SUMS.tsv", checksums)
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
