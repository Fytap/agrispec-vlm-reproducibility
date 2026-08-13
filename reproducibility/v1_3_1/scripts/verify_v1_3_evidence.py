#!/usr/bin/env python3
"""Verify the public v1.3.1 corrected evidence layer using only the standard library."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tsv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def close(actual: str | float, expected: float, tol: float = 1e-12) -> None:
    value = float(actual)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"expected {expected}, observed {value}")


def row(rows: list[dict[str, str]], **keys: str) -> dict[str, str]:
    matches = [item for item in rows if all(item.get(key) == value for key, value in keys.items())]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {keys}, observed {len(matches)}")
    return matches[0]


def verify_contracts() -> None:
    grid = load_tsv("evidence/commissioning_contract/contract_grid_metrics.tsv")
    if len(grid) != 216:
        raise AssertionError(f"contract grid has {len(grid)} rows, expected 216")
    effects = load_tsv("evidence/commissioning_contract/effect_sign_stability.tsv")
    commissioned = row(
        effects,
        ranker_scheme="commissioned_only_shared",
        metric="one_to_one_recall",
    )
    if (commissioned["positive_contracts"], commissioned["zero_contracts"], commissioned["negative_contracts"]) != (
        "15",
        "0",
        "12",
    ):
        raise AssertionError("commissioned-only sign accounting changed")
    spatial = row(
        effects,
        ranker_scheme="commissioned_only_shared",
        metric="spatial_recall",
    )
    if spatial["positive_contracts"] != "27":
        raise AssertionError("commissioned spatial support is not positive in all 27 contracts")


def verify_merge_and_operator() -> None:
    merge = load_tsv("evidence/merge_aware/merge_aware_queue_summary.tsv")
    all_selected = row(merge, group="all_selected")
    if (all_selected["reviewed_candidates"], all_selected["merge_capacity_gap_instances"]) != ("518", "105"):
        raise AssertionError("merge-aware queue totals changed")
    close(all_selected["set_coverage_recall"], 0.26460280373831774)
    close(all_selected["one_to_one_recall"], 0.20327102803738317)

    summary = load_json("evidence/operator_extended/summary.json")
    if summary["items"] != 518 or summary["timing_used"] is not False:
        raise AssertionError("operator item count or no-timing rule changed")
    agreement = load_tsv("evidence/operator_extended/operator_agreement_extended.tsv")
    review = row(agreement, field="weed_reviewable", analysis="all_categories_nominal")
    close(review["exact_agreement"], 0.9420849420849421)
    close(review["gwet_ac1"], 0.9205980185271289)
    human_mask = load_tsv("evidence/operator_extended/human_mask_metrics.tsv")
    if human_mask[0]["analysis_population"] != "determinate_operator_consensus_only":
        raise AssertionError("human--mask analysis population changed")
    if (
        human_mask[0]["true_positive"],
        human_mask[0]["false_positive"],
        human_mask[0]["false_negative"],
        human_mask[0]["true_negative"],
    ) != ("249", "67", "52", "76"):
        raise AssertionError("human--mask determinate contingency changed")
    close(human_mask[0]["eligible_fraction_of_all_items"], 0.8571428571428571)
    close(human_mask[0]["mask_precision_against_human_consensus"], 0.7879746835443038)
    close(human_mask[0]["mask_recall_against_human_consensus"], 0.8272425249169435)
    close(human_mask[0]["f1"], 0.807131280388979)
    close(human_mask[0]["jaccard"], 0.6766304347826086)
    if human_mask[0]["unresolved_human_items"] != "74":
        raise AssertionError("unresolved operator count changed")


def verify_models() -> None:
    summary = load_json("evidence/model_standard_metrics/summary.json")
    if summary["checkpoint_runs"] != 11 or summary["primary_replay_failures"] != 0:
        raise AssertionError("checkpoint count or replay status changed")
    if summary["hardware"]["gpu"] != "NVIDIA H200":
        raise AssertionError("registered model audit hardware is not H200")
    replay = load_tsv("evidence/model_standard_metrics/primary_endpoint_replay_audit.tsv")
    if len(replay) != 66 or any(item["exact_within_1e_12"] != "1" for item in replay):
        raise AssertionError("the 66 primary replay checks are not exact")

    metrics = load_tsv("evidence/model_standard_metrics/model_standard_instance_metrics.tsv")
    mask = row(metrics, model="Mask R-CNN--ResNet50-FPN-v2")
    close(mask["mean_mAP_IoU_0.50_0.95"], 0.1064398140061506)
    close(mask["mean_queue_AR20_IoU_0.50"], 0.18613707165109036)

    contrasts = load_tsv("evidence/crossed_uncertainty/model_contrasts.tsv")
    primary = row(contrasts, contrast="maskrcnn_minus_unet", metric="one_to_one_recall")
    close(primary["point"], 0.013239875389408046)
    if not (float(primary["crossed_seed_image_lower_95"]) < 0.0 < float(primary["crossed_seed_image_upper_95"])):
        raise AssertionError("crossed Mask R-CNN minus U-Net interval no longer contains zero")


def verify_external_and_figures() -> None:
    locked = load_json("evidence/cwfid_continuous/summary.json")
    if locked["images"] != 60 or locked["locked_primary_result_changed"] is not False:
        raise AssertionError("locked CWFID endpoint metadata changed")
    seeds = load_tsv("evidence/cwfid_continuous/per_seed_continuous_score_summary.tsv")
    if len(seeds) != 5 or any(item["locked_images_with_any_candidate"] != "0" for item in seeds):
        raise AssertionError("locked CWFID candidate count changed")
    for index in range(1, 6):
        audit = load_json(f"source_data/Figure_{index}_text_bounds_audit.json")
        if audit["outside_canvas_count"] != 0:
            raise AssertionError(f"Figure {index} has text outside its canvas")


def verify_public_boundary() -> None:
    forbidden = [
        re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
        re.compile(r"/data/[A-Za-z0-9_.-]+/", re.IGNORECASE),
        re.compile(r"jms[-.]research", re.IGNORECASE),
        re.compile(r"ssh\s+[^\n]+@", re.IGNORECASE),
    ]
    readable = {".txt", ".md", ".json", ".tsv", ".csv", ".yaml", ".yml", ".py"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in readable:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in forbidden:
            if pattern.search(text):
                raise AssertionError(f"non-public path or infrastructure identifier in {path.relative_to(ROOT)}")


def verify_manifest() -> int:
    entries = load_tsv("manifests/v1_3_1_sha256.tsv")
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.is_file():
            raise AssertionError(f"manifest path is missing: {entry['path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise AssertionError(f"SHA-256 mismatch: {entry['path']}")
        if path.stat().st_size != int(entry["bytes"]):
            raise AssertionError(f"byte-count mismatch: {entry['path']}")
    return len(entries)


def main() -> None:
    verify_contracts()
    verify_merge_and_operator()
    verify_models()
    verify_external_and_figures()
    verify_public_boundary()
    manifest_entries = verify_manifest()
    print("status: verified")
    print("contract_rows: 216")
    print("primary_replay_checks: 66")
    print("operator_items: 518")
    print("cwfid_images: 60")
    print("figure_text_bounds: 5/5 clear")
    print(f"manifest_entries: {manifest_entries}")


if __name__ == "__main__":
    main()
