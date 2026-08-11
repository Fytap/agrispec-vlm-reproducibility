#!/usr/bin/env python3
"""Audit proposal definitions and compare original/commissioned queues.

The script separates spatial proposal recall, role-qualified proposal recall,
and queued instance recall. Target masks are read only for training-date
setting selection already recorded by the commissioning run and for offline
scoring; proposal maps and semantic scores are reused unchanged.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import (  # noqa: E402
    L2_GRID, average_precision, fit_logistic, predict, read_tsv, roc_auc, sha256,
)

DATES = ("2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15")
LOCAL = ("soil_probability_mean", "log1p_area_pixels", "mean_foreground_probability",
         "max_foreground_probability", "log_bbox_area", "log_bbox_aspect", "bbox_fill_fraction")
FEATURES = {"role_only": ("role_logit",), "local_only": LOCAL,
            "role_plus_local": ("role_logit", *LOCAL)}
KS = (1, 5, 10, 20)
IOU_GRID = (0.25, 0.50, 0.75)
SM_GRID = (0.10, 0.25, 0.50)
CONTRACT_GRID = (0.25, 0.50, 0.75)
PURITY_GRID = (0.50, 0.75, 0.90)
CROP_OVERLAP_GRID = (0.0, 0.01, 0.10, 0.50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for name in ("project-root", "archive", "manifest", "original-root", "grid-root",
                 "commissioned-summary", "config", "output-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--expected-archive-sha256", required=True)
    p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--expected-original-candidates-sha256", required=True)
    p.add_argument("--expected-commissioned-summary-sha256", required=True)
    return p.parse_args()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def load_truth(archive_path: Path, manifest: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(archive_path) as z:
        for row in manifest:
            prefix = "weedsgalore-dataset/"
            semantic = np.asarray(Image.open(io.BytesIO(z.read(prefix + row["semantic_relpath"]))))
            instances = np.asarray(Image.open(io.BytesIO(z.read(prefix + row["instance_relpath"]))))
            records = {}
            for tid in (int(v) for v in np.unique(instances) if int(v) > 0):
                mask = instances == tid; values = semantic[mask]
                crop, weed = int((values == 1).sum()), int((values > 1).sum())
                if crop + weed:
                    records[tid] = {"instance_id": f"{row['sample_id']}:instance:{tid}",
                                    "role": "crop" if crop >= weed else "weed", "area": int(mask.sum())}
            out[row["sample_id"]] = {"date": row["provisional_source_session_id"],
                                     "semantic": semantic, "instances": instances, "records": records}
    return out


def audit_setting(path: Path, manifest: list[dict[str, str]], truth: dict[str, dict[str, object]]) -> dict[str, object]:
    candidates = read_tsv(path / "candidate_components.tsv")
    semantic_scores = {r["candidate_id"]: r for r in read_tsv(path / "candidate_semantic_scores.tsv")}
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates: by_file[row["file"]].append(row)
    a = np.load(path / "candidate_component_label_maps.npz")
    maps, sample_ids = a["label_maps"], [str(v) for v in a["sample_ids"]]
    expected_ids = [r["sample_id"] for r in manifest]
    if sample_ids != expected_ids: raise ValueError(f"Map lineage mismatch: {path}")
    rows: list[dict[str, object]] = []
    matches: list[dict[str, object]] = []
    instances_out: list[dict[str, object]] = []
    qualified: dict[str, set[str]] = defaultdict(set)
    spatial: dict[str, set[str]] = defaultdict(set)
    roles: dict[str, str] = {}
    for frame_index, sample_id in enumerate(sample_ids):
        current = truth[sample_id]; date = str(current["date"]); sem = current["semantic"]; inst = current["instances"]
        labels = maps[frame_index]; local = {int(r["component_index"]): r for r in by_file[sample_id]}
        max_label = int(labels.max()); area = np.bincount(labels.ravel(), minlength=max_label + 1)
        crop_px = np.bincount(labels[sem == 1].ravel(), minlength=max_label + 1)
        weed_px = np.bincount(labels[sem > 1].ravel(), minlength=max_label + 1)
        contracts = {}
        for label, cand in local.items():
            total_area = max(1, int(area[label])); labeled = int(crop_px[label] + weed_px[label])
            coverage = labeled / total_area
            crop_purity, weed_purity = int(crop_px[label]) / max(1, labeled), int(weed_px[label]) / max(1, labeled)
            eligible = coverage >= .5 and max(crop_purity, weed_purity) >= .9
            target = ("crop" if crop_purity >= weed_purity else "weed") if eligible else "ambiguous_or_background"
            contracts[label] = (coverage, crop_purity, weed_purity, target)
            s = semantic_scores[cand["candidate_id"]]; width = float(cand["x1_exclusive"]) - float(cand["x0"])
            height = float(cand["y1_exclusive"]) - float(cand["y0"]); pixels = float(cand["area_pixels"]); eps = 1e-8
            rows.append({"candidate_id": cand["candidate_id"], "date": date, "file": sample_id,
                         "truth": target, "eligible": int(eligible), "candidate_labeled_coverage": coverage,
                         "crop_fraction_candidate": int(crop_px[label]) / total_area,
                         "weed_fraction_candidate": int(weed_px[label]) / total_area,
                         "role_logit": math.log((float(s["weed_probability_mean"]) + eps) /
                                                (float(s["crop_probability_mean"]) + eps)),
                         "soil_probability_mean": float(s["soil_probability_mean"]),
                         "log1p_area_pixels": math.log1p(pixels),
                         "mean_foreground_probability": float(cand["mean_foreground_probability"]),
                         "max_foreground_probability": float(cand["max_foreground_probability"]),
                         "log_bbox_area": math.log(max(1., width * height)),
                         "log_bbox_aspect": math.log(max(width, 1.) / max(height, 1.)),
                         "bbox_fill_fraction": pixels / max(1., width * height)})
        overlap_map = {}
        foreground = (inst > 0) & (labels > 0)
        if foreground.any():
            pairs, counts = np.unique(np.column_stack([inst[foreground], labels[foreground]]), axis=0, return_counts=True)
            overlap_map = {(int(p[0]), int(p[1])): int(n) for p, n in zip(pairs, counts, strict=True)}
        for tid, tr in current["records"].items():
            iid, role, iarea = tr["instance_id"], tr["role"], int(tr["area"]); roles[iid] = role
            local_matches = []
            for label, cand in local.items():
                ov = overlap_map.get((tid, label), 0)
                if ov == 0: continue
                coverage, cp, wp, _ = contracts[label]; carea = int(area[label]); purity = cp if role == "crop" else wp
                rec = {"instance_id": iid, "sample_id": sample_id, "date": date, "truth_role": role,
                       "candidate_id": cand["candidate_id"], "overlap_pixels": ov,
                       "instance_area_pixels": iarea, "candidate_area_pixels": carea,
                       "instance_coverage": ov / iarea, "candidate_area_share": ov / max(1, carea),
                       "iou": ov / max(1, iarea + carea - ov), "candidate_labeled_coverage": coverage,
                       "same_role_purity": purity}
                matches.append(rec); local_matches.append(rec)
                if rec["instance_coverage"] >= .5: spatial[cand["candidate_id"]].add(iid)
                if rec["instance_coverage"] >= .5 and coverage >= .5 and purity >= .9:
                    qualified[cand["candidate_id"]].add(iid)
            instances_out.append({"instance_id": iid, "sample_id": sample_id, "date": date, "truth_role": role,
                                  "best_instance_coverage": max((float(r["instance_coverage"]) for r in local_matches), default=0.),
                                  "best_iou": max((float(r["iou"]) for r in local_matches), default=0.),
                                  "spatial_recalled": int(any(float(r["instance_coverage"]) >= .5 for r in local_matches)),
                                  "role_qualified_recalled": int(any(float(r["instance_coverage"]) >= .5 and float(r["candidate_labeled_coverage"]) >= .5 and float(r["same_role_purity"]) >= .9 for r in local_matches))})
    return {"rows": rows, "matches": matches, "instances": instances_out, "qualified": qualified,
            "spatial": spatial, "roles": roles, "input_hashes": {
                "candidates": sha256(path / "candidate_components.tsv"),
                "semantic_scores": sha256(path / "candidate_semantic_scores.tsv"),
                "component_maps": sha256(path / "candidate_component_label_maps.npz")}}


def fit_outer(rows: list[dict[str, object]], held: str, features: tuple[str, ...]) -> tuple[dict[str, float], dict[str, object]]:
    train_dates = tuple(d for d in DATES if d != held); sensitivity = []
    def xy(ds: tuple[str, ...]):
        chosen = [r for r in rows if int(r["eligible"]) and r["date"] in ds]
        return np.asarray([[float(r[f]) for f in features] for r in chosen]), np.asarray([r["truth"] == "weed" for r in chosen], dtype=np.int64), chosen
    for l2 in L2_GRID:
        aucs = []
        for val in train_dates:
            tx, ty, _ = xy(tuple(d for d in train_dates if d != val)); vx, vy, _ = xy((val,))
            aucs.append(float(roc_auc(predict(fit_logistic(tx, ty, float(l2)), vx), vy)))
        sensitivity.append({"l2": float(l2), "mean_inner_date_auroc": float(np.mean(aucs))})
    best = max(sensitivity, key=lambda r: (r["mean_inner_date_auroc"], -abs(math.log10(r["l2"]))))
    tx, ty, _ = xy(train_dates); model = fit_logistic(tx, ty, float(best["l2"]))
    test = [r for r in rows if r["date"] == held]
    x = np.asarray([[float(r[f]) for f in features] for r in test])
    return {r["candidate_id"]: float(s) for r, s in zip(test, predict(model, x), strict=True)}, {"selected_l2": best["l2"], "inner": sensitivity}


def proposal_rows(variant: str, date: str, audit: dict[str, object]) -> list[dict[str, object]]:
    out = []
    for role in ("crop", "weed"):
        rows = [r for r in audit["instances"] if r["date"] == date and r["truth_role"] == role]
        rec = {"variant": variant, "held_out_date": date, "truth_role": role, "truth_instances": len(rows),
               "spatial_recalled_instances": sum(int(r["spatial_recalled"]) for r in rows),
               "spatial_proposal_recall": np.mean([int(r["spatial_recalled"]) for r in rows]) if rows else 0.,
               "role_qualified_recalled_instances": sum(int(r["role_qualified_recalled"]) for r in rows),
               "role_qualified_proposal_recall": np.mean([int(r["role_qualified_recalled"]) for r in rows]) if rows else 0.}
        for t in IOU_GRID: rec[f"proposal_ar_iou_{t:.2f}"] = np.mean([float(r["best_iou"]) >= t for r in rows]) if rows else 0.
        out.append(rec)
    return out


def queue(variant: str, model_name: str, date: str, k: int, audit: dict[str, object], scores: dict[str, float]) -> dict[str, object]:
    by_file: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in audit["rows"]:
        if r["date"] == date: by_file[str(r["file"])].append(r)
    accepted = [r for rs in by_file.values() for r in sorted(rs, key=lambda x: (-scores[x["candidate_id"]], x["candidate_id"]))[:k]]
    counts = Counter(str(r["truth"]) for r in accepted)
    qids = {iid for r in accepted for iid in audit["qualified"].get(r["candidate_id"], set())}
    sids = {iid for r in accepted for iid in audit["spatial"].get(r["candidate_id"], set())}
    weed_ids = {iid for iid, role in audit["roles"].items() if role == "weed" and any(iid.startswith(f + ":instance:") for f in by_file)}
    result = {"variant": variant, "model": model_name, "held_out_date": date, "k": k, "images": len(by_file),
              "accepted_candidates": len(accepted), "eligible_weed_candidates": counts["weed"],
              "eligible_crop_candidates": counts["crop"], "ambiguous_or_background_candidates": counts["ambiguous_or_background"],
              "all_candidate_precision": counts["weed"] / max(1, len(accepted)),
              "queued_spatial_weed_instance_recall": len(sids & weed_ids) / max(1, len(weed_ids)),
              "queued_role_qualified_weed_instance_recall": len(qids & weed_ids) / max(1, len(weed_ids)),
              "pure_crop_scene_frequency": len({r["file"] for r in accepted if r["truth"] == "crop"}) / max(1, len(by_file))}
    for t in CROP_OVERLAP_GRID:
        scenes = {r["file"] for r in accepted if (float(r["crop_fraction_candidate"]) > 0 if t == 0 else float(r["crop_fraction_candidate"]) >= t)}
        result[f"crop_overlap_scene_frequency_{t:.2f}"] = len(scenes) / max(1, len(by_file))
    return result


def sensitivity_rows(variant: str, date: str, audit: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    inst = [r for r in audit["instances"] if r["date"] == date]
    ids_by_role = {role: {r["instance_id"] for r in inst if r["truth_role"] == role} for role in ("crop", "weed")}
    matches = [r for r in audit["matches"] if r["date"] == date]
    contract = []
    for ic in CONTRACT_GRID:
        for lc in CONTRACT_GRID:
            for p in PURITY_GRID:
                hit = {r["instance_id"] for r in matches if float(r["instance_coverage"]) >= ic and float(r["candidate_labeled_coverage"]) >= lc and float(r["same_role_purity"]) >= p}
                for role in ("crop", "weed"):
                    denom = ids_by_role[role]
                    contract.append({"variant": variant, "held_out_date": date, "truth_role": role,
                                     "minimum_instance_coverage": ic, "minimum_candidate_labeled_coverage": lc,
                                     "minimum_same_role_purity": p, "truth_instances": len(denom),
                                     "role_qualified_proposal_recall": len(hit & denom) / max(1, len(denom))})
    sm = []
    for t in SM_GRID:
        split = Counter(); merged: dict[str, set[str]] = defaultdict(set)
        for r in matches:
            if float(r["instance_coverage"]) >= t:
                split[r["instance_id"]] += 1; merged[r["candidate_id"]].add(r["instance_id"])
        sm.append({"variant": variant, "held_out_date": date, "instance_coverage_threshold": t,
                   "split_instances": sum(n >= 2 for n in split.values()),
                   "merge_candidates": sum(len(v) >= 2 for v in merged.values())})
    return contract, sm


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""): raise RuntimeError("CPU-only experiment")
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    checks = [(args.archive, args.expected_archive_sha256), (args.manifest, args.expected_manifest_sha256),
              (args.original_root / "candidate_components.tsv", args.expected_original_candidates_sha256),
              (args.commissioned_summary, args.expected_commissioned_summary_sha256)]
    for path, expected in checks:
        if sha256(path.resolve()) != expected: raise ValueError(f"Input hash mismatch: {path}")
    started = time.perf_counter(); manifest = read_tsv(args.manifest.resolve()); truth = load_truth(args.archive.resolve(), manifest)
    commissioning = json.loads(args.commissioned_summary.read_text(encoding="utf-8"))
    selected = {d: commissioning["per_outer_date"][d]["setting"] for d in DATES}
    cache: dict[str, dict[str, object]] = {}
    def get(path: Path) -> dict[str, object]:
        key = str(path.resolve())
        if key not in cache: cache[key] = audit_setting(path.resolve(), manifest, truth)
        return cache[key]
    original = get(args.original_root)
    proposal, ranking, queues, contract, sm, predictions = [], [], [], [], [], []
    input_hashes = {"original": original["input_hashes"], "commissioned": {}}
    for variant in ("original", "nested_commissioned"):
        for date in DATES:
            audit = original if variant == "original" else get(args.grid_root / selected[date])
            if variant == "nested_commissioned": input_hashes["commissioned"][selected[date]] = audit["input_hashes"]
            proposal.extend(proposal_rows(variant, date, audit))
            cr, sr = sensitivity_rows(variant, date, audit); contract.extend(cr); sm.extend(sr)
            for model_name, feats in FEATURES.items():
                scores, artifact = fit_outer(audit["rows"], date, feats)
                eligible = [r for r in audit["rows"] if r["date"] == date and int(r["eligible"])]
                y = np.asarray([r["truth"] == "weed" for r in eligible], dtype=np.int64)
                vals = np.asarray([scores[r["candidate_id"]] for r in eligible])
                ranking.append({"variant": variant, "model": model_name, "held_out_date": date,
                                "eligible_candidates": len(eligible), "eligible_crop_candidates": int((y == 0).sum()),
                                "eligible_weed_candidates": int(y.sum()), "roc_auc": roc_auc(vals, y),
                                "average_precision": average_precision(vals, y, [r["candidate_id"] for r in eligible]),
                                "selected_l2": artifact["selected_l2"]})
                for k in KS: queues.append(queue(variant, model_name, date, k, audit, scores))
                for r in audit["rows"]:
                    if r["date"] == date:
                        predictions.append({"variant": variant, "model": model_name, "held_out_date": date,
                                            "candidate_id": r["candidate_id"], "file": r["file"],
                                            "dataset_role_target": r["truth"], "score": f"{scores[r['candidate_id']]:.10f}"})
    out.mkdir(parents=True)
    paths = {"proposal_layers": out / "proposal_recall_layers.tsv", "ranking": out / "candidate_ranking_by_date.tsv",
             "queue": out / "commissioning_queue_comparison.tsv", "contract": out / "matching_contract_sensitivity.tsv",
             "split_merge": out / "split_merge_sensitivity.tsv", "predictions": out / "outer_predictions.tsv"}
    for name, rows in (("proposal_layers", proposal), ("ranking", ranking), ("queue", queues),
                       ("contract", contract), ("split_merge", sm), ("predictions", predictions)): write_tsv(paths[name], rows)
    primary = [r for r in queues if r["model"] == "role_plus_local" and int(r["k"]) == 20]
    summary = {"run_id": args.run_id, "status": "completed_sat_major_revision_contract_audit",
               "scope": {"public_data_only": True, "sealed_test_read": False,
                         "proposal_generation_reads_target_truth": False,
                         "training_date_dense_masks_used_for_setting_selection": True,
                         "held_out_date_used_only_for_offline_scoring": True, "formal_risk_certificate": False},
               "definitions": {"spatial_proposal_recall": "one candidate covers >=50% of a truth instance; candidate role is ignored",
                               "role_qualified_proposal_recall": "spatial rule plus >=50% candidate labeled coverage and >=90% same-role purity",
                               "queued_instance_recall": "truth instance has a qualifying candidate within the fixed per-image prefix"},
               "selected_commissioned_settings": selected, "input_artifact_hashes": input_hashes,
               "primary_k20_date_macro": {v: {m: float(np.mean([float(r[m]) for r in primary if r["variant"] == v]))
                                                   for m in ("all_candidate_precision", "queued_spatial_weed_instance_recall",
                                                             "queued_role_qualified_weed_instance_recall", "pure_crop_scene_frequency",
                                                             "crop_overlap_scene_frequency_0.00")}
                                            for v in ("original", "nested_commissioned")},
               "outputs": {name + "_sha256": sha256(path) for name, path in paths.items()},
               "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version, "numpy": np.__version__},
               "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(args.config.resolve()),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
