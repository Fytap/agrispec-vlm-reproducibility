#!/usr/bin/env python3
"""Verify the v1.2.x public evidence layer using the Python standard library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".svg",
    ".tex",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_manifest_bytes(path: Path, content: bytes) -> bytes:
    """Normalise text line endings while leaving binary artifacts byte-exact."""
    if path.suffix.lower() in TEXT_SUFFIXES and b"\x00" not in content:
        return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return content


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, observed {actual}")


def verify_manifest(root: Path) -> int:
    manifest = root / "manifests" / "PUBLIC_SHA256SUMS.tsv"
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", newline=""), delimiter="\t"))
    for row in rows:
        path = root / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        content = path.read_bytes()
        expected_size = int(row["bytes"])
        expected_sha256 = row["sha256"]
        if len(content) == expected_size and sha256_bytes(content) == expected_sha256:
            continue
        canonical = canonical_manifest_bytes(path, content)
        if len(canonical) == expected_size and sha256_bytes(canonical) == expected_sha256:
            continue
        raise AssertionError(f"size or checksum mismatch: {row['relative_path']}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    verified_files = verify_manifest(root)
    matched = load_json(root / "evidence" / "matched_burden" / "summary.json")
    strong = load_json(root / "evidence" / "strong_baselines" / "summary.json")
    semantic = load_json(root / "evidence" / "semantic_ranking" / "summary.json")
    external = load_json(root / "evidence" / "cwfid_external" / "consolidation" / "summary.json")
    operators = load_json(root / "evidence" / "operator_review" / "summary.json")

    original = matched["paired_common_K20"]["original_T095_A032"]
    commissioned = matched["paired_common_K20"]["commissioned_T070_A032"]
    assert original["accepted_candidates"] == commissioned["accepted_candidates"] == 471
    assert original["weed_instances"] == commissioned["weed_instances"] == 1712
    close(original["one_to_one_role_qualified_recall"], 0.0969626168224299)
    close(commissioned["one_to_one_role_qualified_recall"], 0.11565420560747663)

    contrasts = {(row["contrast"], row["metric"]): row for row in strong["primary_contrasts"]}
    mask_contrast = contrasts[("Mask-RCNN-ResNet50-FPN-v2_minus_ResNet18-UNet", "one_to_one_role_qualified_recall")]
    close(mask_contrast["estimate"], 0.013239875389408129)
    assert mask_contrast["ci95_low"] > 0

    semantic_recall = semantic["official_test_selected_policy_mean_sample_sd"]["one_to_one_role_qualified_recall"]
    close(semantic_recall["mean"], 0.2116822429906542)
    assert external["images"] == 60
    assert external["unique_weed_components"] == 331
    assert external["candidate_count_all_seeds"] == 0
    close(external["primary_endpoint_mean"], 0.0)
    assert operators["items"] == 518 and operators["scenes"] == 26
    close(operators["agreement"]["weed_reviewable"]["agreement"], 0.9420849420849421)
    close(operators["agreement"]["weed_reviewable"]["cohen_kappa"], 0.8930105750165236)

    key_rows = [
        ("matched_original_one_to_one_recall", original["one_to_one_role_qualified_recall"]),
        ("matched_commissioned_one_to_one_recall", commissioned["one_to_one_role_qualified_recall"]),
        ("mask_rcnn_minus_unet_one_to_one_recall", mask_contrast["estimate"]),
        ("semantic_probability_one_to_one_recall", semantic_recall["mean"]),
        ("cwfid_candidates_all_seeds", external["candidate_count_all_seeds"]),
        ("cwfid_one_to_one_recall", external["primary_endpoint_mean"]),
        ("operator_weed_reviewability_agreement", operators["agreement"]["weed_reviewable"]["agreement"]),
        ("operator_weed_reviewability_kappa", operators["agreement"]["weed_reviewable"]["cohen_kappa"]),
    ]
    with (output / "key_results.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "value"])
        writer.writerows(key_rows)

    report = {
        "status": "verified",
        "manifest_files_verified": verified_files,
        "evidence_summaries_loaded": 5,
        "primary_denominators": {"weedsgalore_test_instances": 1712, "cwfid_images": 60, "operator_items": 518},
    }
    with (output / "verification.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
