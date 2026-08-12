#!/usr/bin/env python3
"""Locked cross-dataset evaluation of five WeedsGalore-selected checkpoints.

Candidate generation and ranking use CWFID RGB only.  Public annotations are
loaded afterwards and are restricted to offline scoring and integrity audits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


COMMIT = "176612f3506c0ce70d85518cb82ddd938a6206c8"
SEEDS = (20260810, 20260811, 20260812, 20260813, 20260814)
PRIMARY_TRUTH_AREA = 16
TRUTH_AREA_GRID = (1, 16, 32, 64)
QUEUE_K = 20
INSTANCE_COVERAGE = 0.50
LABELED_COVERAGE = 0.50
ROLE_PURITY = 0.90
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
RUN_TEMPLATE = "P2_WEEDSGALORE_TARGET_SEMANTIC_H200_SEED_{seed}_20260812_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu-id", type=int, choices=(6, 7), required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_git_tree(dataset_root: Path) -> tuple[list[dict[str, Any]], list[tuple[str, Path, Path, Path]]]:
    inventory_path = dataset_root / "UPSTREAM_GIT_TREE.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("sha") != COMMIT or inventory.get("truncated"):
        raise ValueError("Frozen upstream tree identity failed")
    blobs = [item for item in inventory["tree"] if item["type"] == "blob"]
    if len(blobs) != 242 or sum(int(item["size"]) for item in blobs) != 90444113:
        raise ValueError("Frozen upstream tree inventory changed")
    lineage: list[dict[str, Any]] = []
    for item in sorted(blobs, key=lambda value: value["path"]):
        path = dataset_root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        actual_blob = git_blob_sha1(path)
        if actual_size != int(item["size"]) or actual_blob != item["sha"]:
            raise ValueError(f"Git blob mismatch: {item['path']}")
        role = (
            "rgb" if item["path"].startswith("images/")
            else "annotation" if item["path"].startswith("annotations/")
            else "vegetation_mask" if item["path"].startswith("masks/")
            else "documentation_or_split"
        )
        lineage.append({
            "relative_path": item["path"],
            "role": role,
            "bytes": actual_size,
            "upstream_git_blob_sha1": actual_blob,
            "sha256": sha256(path),
        })
    readme = (dataset_root / "README.md").read_text(encoding="utf-8")
    if "non-commercial research" not in readme or "cite our publication" not in readme:
        raise ValueError("Upstream licence/use notice was not found")

    def index_files(folder: str, suffix: str) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for path in sorted((dataset_root / folder).glob(f"*{suffix}")):
            key = path.name[: -len(suffix)]
            if key in result:
                raise ValueError(f"Duplicate key {key} in {folder}")
            result[key] = path
        return result

    images = index_files("images", "_image.png")
    annotations = index_files("annotations", "_annotation.png")
    masks = index_files("masks", "_mask.png")
    keys = sorted(set(images) & set(annotations) & set(masks))
    if len(keys) != 60 or set(keys) != set(images) or set(keys) != set(annotations) or set(keys) != set(masks):
        raise ValueError("Expected exactly 60 one-to-one RGB/annotation/mask triplets")
    return lineage, [(key, images[key], annotations[key], masks[key]) for key in keys]


def transform_geometry(width: int, height: int) -> tuple[int, int, int, int, int, int]:
    scale = 600.0 / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    left = (600 - resized_width) // 2
    right = 600 - resized_width - left
    top = (600 - resized_height) // 2
    bottom = 600 - resized_height - top
    return resized_width, resized_height, left, right, top, bottom


def load_rgb(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        rw, rh, left, right, top, bottom = transform_geometry(width, height)
        resized = np.asarray(image.resize((rw, rh), Image.Resampling.BILINEAR), dtype=np.uint8)
    canvas = np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    if canvas.shape != (600, 600, 3):
        raise ValueError(f"RGB canvas contract failed for {path}: {canvas.shape}")
    geometry = {
        "source_width": width,
        "source_height": height,
        "resized_width": rw,
        "resized_height": rh,
        "pad_left": left,
        "pad_right": right,
        "pad_top": top,
        "pad_bottom": bottom,
    }
    return canvas, geometry


def load_annotation(path: Path, geometry: dict[str, int]) -> tuple[np.ndarray, list[list[int]], int]:
    with Image.open(path) as source:
        annotation = source.convert("RGB").resize(
            (geometry["resized_width"], geometry["resized_height"]),
            Image.Resampling.NEAREST,
        )
        raw = np.asarray(annotation, dtype=np.uint8)
    raw = np.pad(
        raw,
        (
            (geometry["pad_top"], geometry["pad_bottom"]),
            (geometry["pad_left"], geometry["pad_right"]),
            (0, 0),
        ),
        mode="constant",
        constant_values=0,
    )
    # Frozen one-based channel contract: channel 2 (green) is crop and
    # channel 1 (red) is weed.
    crop = raw[..., 1] == 255
    weed = raw[..., 0] == 255
    overlap = int((crop & weed).sum())
    if overlap:
        raise ValueError(f"Crop/weed annotation channels overlap in {path}")
    semantic = np.zeros((600, 600), dtype=np.uint8)
    semantic[crop] = 1
    semantic[weed] = 2
    colors = np.unique(raw.reshape(-1, 3), axis=0).tolist()
    return semantic, colors, overlap


def build_model():
    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torchvision.models import resnet18

    class Block(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs):
            return self.layers(inputs)

    class ResNet18UNet(nn.Module):
        def __init__(self):
            super().__init__()
            encoder = resnet18(weights=None)
            self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
            self.pool = encoder.maxpool
            self.layer1 = encoder.layer1
            self.layer2 = encoder.layer2
            self.layer3 = encoder.layer3
            self.layer4 = encoder.layer4
            self.dec4 = Block(768, 256)
            self.dec3 = Block(384, 128)
            self.dec2 = Block(192, 64)
            self.dec1 = Block(128, 64)
            self.head = nn.Conv2d(64, 3, 1)

        def forward(self, inputs):
            output_size = inputs.shape[-2:]
            s0 = self.stem(inputs)
            s1 = self.layer1(self.pool(s0))
            s2 = self.layer2(s1)
            s3 = self.layer3(s2)
            s4 = self.layer4(s3)
            up = lambda value, reference: functional.interpolate(
                value, size=reference.shape[-2:], mode="bilinear", align_corners=False
            )
            d4 = self.dec4(torch.cat([up(s4, s3), s3], dim=1))
            d3 = self.dec3(torch.cat([up(d4, s2), s2], dim=1))
            d2 = self.dec2(torch.cat([up(d3, s1), s1], dim=1))
            d1 = self.dec1(torch.cat([up(d2, s0), s0], dim=1))
            return self.head(functional.interpolate(d1, size=output_size, mode="bilinear", align_corners=False))

    return ResNet18UNet()


def generate_candidates(seed: int, image_id: str, probabilities: np.ndarray, threshold: float, minimum_area: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    plant = 1.0 - probabilities[0]
    labels, count = ndimage.label(plant >= threshold, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    rows: list[dict[str, Any]] = []
    for component in range(1, count + 1):
        area = int(areas[component])
        if area < minimum_area:
            continue
        mask = labels == component
        ys, xs = np.nonzero(mask)
        plant_mean = float((probabilities[1][mask] + probabilities[2][mask]).mean())
        weed_mean = float(probabilities[2][mask].mean())
        normalized = probabilities[:, mask]
        entropy = float((-normalized * np.log(np.clip(normalized, 1e-8, 1.0))).sum(axis=0).mean())
        candidate_id = f"{image_id}:seed:{seed}:component:{component:05d}"
        rows.append({
            "candidate_id": candidate_id,
            "component": component,
            "area_pixels": area,
            "x0": int(xs.min()),
            "y0": int(ys.min()),
            "x1_exclusive": int(xs.max()) + 1,
            "y1_exclusive": int(ys.max()) + 1,
            "score": weed_mean / max(1e-8, plant_mean),
            "mean_weed_probability": weed_mean,
            "mean_entropy": entropy,
        })
    rows.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
    return labels, rows


def deterministic_matching(candidate_ids: list[str], edges: dict[str, set[str]]) -> dict[str, str]:
    owner: dict[str, str] = {}

    def augment(candidate_id: str, visited: set[str]) -> bool:
        for truth_id in sorted(edges.get(candidate_id, set())):
            if truth_id in visited:
                continue
            visited.add(truth_id)
            previous = owner.get(truth_id)
            if previous is None or augment(previous, visited):
                owner[truth_id] = candidate_id
                return True
        return False

    for candidate_id in sorted(candidate_ids):
        augment(candidate_id, set())
    return owner


def score_image(
    seed: int,
    image_id: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    candidates: list[dict[str, Any]],
    semantic: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    truth_labels, count = ndimage.label(semantic == 2, structure=np.ones((3, 3), dtype=np.uint8))
    truth_areas_raw = np.bincount(truth_labels.ravel(), minlength=count + 1)
    truth_areas = {f"{image_id}:weed:{value:05d}": int(truth_areas_raw[value]) for value in range(1, count + 1)}
    truth_by_value = {value: f"{image_id}:weed:{value:05d}" for value in range(1, count + 1)}
    accepted_ids = {str(row["candidate_id"]) for row in candidates[:QUEUE_K]}
    spatial_edges: dict[str, set[str]] = defaultdict(set)
    qualified_edges: dict[str, set[str]] = defaultdict(set)
    candidate_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, 1):
        candidate_id = str(candidate["candidate_id"])
        mask = labels == int(candidate["component"])
        area = int(candidate["area_pixels"])
        crop_pixels = int((semantic[mask] == 1).sum())
        weed_pixels = int((semantic[mask] == 2).sum())
        labeled_pixels = crop_pixels + weed_pixels
        labeled_coverage = labeled_pixels / max(1, area)
        crop_purity = crop_pixels / max(1, labeled_pixels)
        weed_purity = weed_pixels / max(1, labeled_pixels)
        role = (
            "weed" if labeled_coverage >= LABELED_COVERAGE and weed_purity >= ROLE_PURITY
            else "crop" if labeled_coverage >= LABELED_COVERAGE and crop_purity >= ROLE_PURITY
            else "ambiguous_or_background"
        )
        overlap_values, overlap_counts = np.unique(truth_labels[mask], return_counts=True)
        max_coverage = 0.0
        for value, overlap_count in zip(overlap_values.tolist(), overlap_counts.tolist()):
            if int(value) == 0:
                continue
            truth_id = truth_by_value[int(value)]
            coverage = int(overlap_count) / truth_areas[truth_id]
            max_coverage = max(max_coverage, coverage)
            if coverage >= INSTANCE_COVERAGE:
                spatial_edges[candidate_id].add(truth_id)
                if labeled_coverage >= LABELED_COVERAGE and weed_purity >= ROLE_PURITY:
                    qualified_edges[candidate_id].add(truth_id)
            if candidate_id in accepted_ids:
                overlap_rows.append({
                    "seed": seed,
                    "image_id": image_id,
                    "candidate_id": candidate_id,
                    "truth_component_id": truth_id,
                    "overlap_pixels": int(overlap_count),
                    "truth_component_area": truth_areas[truth_id],
                    "instance_coverage": coverage,
                    "spatial_edge": int(coverage >= INSTANCE_COVERAGE),
                    "qualified_edge": int(truth_id in qualified_edges.get(candidate_id, set())),
                })
        candidate_rows.append({
            "seed": seed,
            "image_id": image_id,
            "rank_within_image": rank,
            "accepted_K20": int(rank <= QUEUE_K),
            **{key: value for key, value in candidate.items() if key != "component"},
            "candidate_truth": role,
            "candidate_labeled_coverage": labeled_coverage,
            "candidate_crop_purity": crop_purity,
            "candidate_weed_purity": weed_purity,
            "crop_fraction_candidate": crop_pixels / max(1, area),
            "weed_fraction_candidate": weed_pixels / max(1, area),
            "maximum_weed_component_coverage": max_coverage,
        })

    accepted = candidate_rows[:QUEUE_K]
    accepted_list = [str(row["candidate_id"]) for row in accepted]
    sensitivity_rows: list[dict[str, Any]] = []
    primary: dict[str, Any] | None = None
    matched_primary_qualified: dict[str, str] = {}
    matched_primary_spatial: dict[str, str] = {}
    for minimum_area in TRUTH_AREA_GRID:
        eligible_truth = {truth_id for truth_id, area in truth_areas.items() if area >= minimum_area}
        filtered_spatial = {candidate_id: values & eligible_truth for candidate_id, values in spatial_edges.items()}
        filtered_qualified = {candidate_id: values & eligible_truth for candidate_id, values in qualified_edges.items()}
        matched_spatial = deterministic_matching(accepted_list, filtered_spatial)
        matched_qualified = deterministic_matching(accepted_list, filtered_qualified)
        spatial_many = set().union(*(filtered_spatial.get(candidate_id, set()) for candidate_id in accepted_list)) if accepted_list else set()
        qualified_many = set().union(*(filtered_qualified.get(candidate_id, set()) for candidate_id in accepted_list)) if accepted_list else set()
        record = {
            "seed": seed,
            "image_id": image_id,
            "truth_minimum_area": minimum_area,
            "weed_components": len(eligible_truth),
            "many_to_one_spatial_recalled": len(spatial_many),
            "many_to_one_role_qualified_recalled": len(qualified_many),
            "one_to_one_spatial_recalled": len(matched_spatial),
            "one_to_one_role_qualified_recalled": len(matched_qualified),
        }
        sensitivity_rows.append(record)
        if minimum_area == PRIMARY_TRUTH_AREA:
            primary = record
            matched_primary_spatial = matched_spatial
            matched_primary_qualified = matched_qualified
    if primary is None:
        raise RuntimeError("Primary truth-area result missing")
    for row in overlap_rows:
        truth_id = str(row["truth_component_id"])
        candidate_id = str(row["candidate_id"])
        row["matched_spatial"] = int(matched_primary_spatial.get(truth_id) == candidate_id)
        row["matched_role_qualified"] = int(matched_primary_qualified.get(truth_id) == candidate_id)

    eligible_weed = sum(str(row["candidate_truth"]) == "weed" for row in accepted)
    crop_overlap = int(any(float(row["crop_fraction_candidate"]) > 0 for row in accepted))
    pred = probabilities.argmax(axis=0)
    confusion = np.bincount((3 * semantic.ravel() + pred.ravel()), minlength=9).reshape(3, 3)
    tile = {
        "seed": seed,
        "image_id": image_id,
        "weed_components": primary["weed_components"],
        "candidates": len(candidate_rows),
        "accepted_candidates": len(accepted),
        "eligible_weed_candidates": eligible_weed,
        "many_to_one_spatial_recalled": primary["many_to_one_spatial_recalled"],
        "many_to_one_role_qualified_recalled": primary["many_to_one_role_qualified_recalled"],
        "one_to_one_spatial_recalled": primary["one_to_one_spatial_recalled"],
        "one_to_one_role_qualified_recalled": primary["one_to_one_role_qualified_recalled"],
        "crop_overlap_image": crop_overlap,
        "confusion_00": int(confusion[0, 0]),
        "confusion_01": int(confusion[0, 1]),
        "confusion_02": int(confusion[0, 2]),
        "confusion_10": int(confusion[1, 0]),
        "confusion_11": int(confusion[1, 1]),
        "confusion_12": int(confusion[1, 2]),
        "confusion_20": int(confusion[2, 0]),
        "confusion_21": int(confusion[2, 1]),
        "confusion_22": int(confusion[2, 2]),
    }
    return tile, candidate_rows, overlap_rows, sensitivity_rows, Counter({key: int(value) for key, value in tile.items() if key.startswith("confusion_")})


def aggregate_tiles(rows: list[dict[str, Any]], indices: np.ndarray) -> dict[str, float]:
    selected = [rows[int(index)] for index in indices]
    weed = sum(int(row["weed_components"]) for row in selected)
    accepted = sum(int(row["accepted_candidates"]) for row in selected)
    eligible = sum(int(row["eligible_weed_candidates"]) for row in selected)
    confusion = np.asarray([[sum(int(row[f"confusion_{i}{j}"]) for row in selected) for j in range(3)] for i in range(3)])
    result = {
        "images": float(len(selected)),
        "weed_components": float(weed),
        "accepted_candidates": float(accepted),
        "mean_candidates_per_image": accepted / max(1, len(selected)),
        "all_candidate_precision": eligible / max(1, accepted),
        "many_to_one_spatial_recall": sum(int(row["many_to_one_spatial_recalled"]) for row in selected) / max(1, weed),
        "many_to_one_role_qualified_recall": sum(int(row["many_to_one_role_qualified_recalled"]) for row in selected) / max(1, weed),
        "one_to_one_spatial_recall": sum(int(row["one_to_one_spatial_recalled"]) for row in selected) / max(1, weed),
        "one_to_one_role_qualified_recall": sum(int(row["one_to_one_role_qualified_recalled"]) for row in selected) / max(1, weed),
        "crop_overlap_image_frequency": sum(int(row["crop_overlap_image"]) for row in selected) / max(1, len(selected)),
    }
    ious = []
    for role, index in (("background", 0), ("crop", 1), ("weed", 2)):
        true_positive = int(confusion[index, index])
        denominator = true_positive + int(confusion[:, index].sum() - true_positive) + int(confusion[index, :].sum() - true_positive)
        value = true_positive / max(1, denominator)
        result[f"{role}_iou"] = value
        ious.append(value)
    result["mean_iou"] = float(np.mean(ious))
    return result


def percentile(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values[np.isfinite(values)], [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise RuntimeError("Frozen physical GPU assignment mismatch")
    import torch

    root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    config = args.config.resolve()
    contract = args.dataset_contract.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    lineage, triplets = audit_git_tree(dataset_root)
    write_tsv(output / "data_lineage.tsv", lineage)
    input_rows: list[dict[str, Any]] = []
    for image_id, image_path, annotation_path, mask_path in triplets:
        with Image.open(image_path) as image:
            width, height = image.size
        with Image.open(annotation_path) as annotation:
            annotation_width, annotation_height = annotation.size
        with Image.open(mask_path) as mask:
            mask_width, mask_height = mask.size
        if (width, height) != (annotation_width, annotation_height) or (width, height) != (mask_width, mask_height):
            raise ValueError(f"Unaligned upstream dimensions for {image_id}")
        input_rows.append({
            "image_id": image_id,
            "source_width": width,
            "source_height": height,
            "rgb_relative_path": image_path.relative_to(dataset_root).as_posix(),
            "annotation_relative_path": annotation_path.relative_to(dataset_root).as_posix(),
            "vegetation_mask_relative_path": mask_path.relative_to(dataset_root).as_posix(),
            "rgb_sha256": sha256(image_path),
            "annotation_sha256": sha256(annotation_path),
            "vegetation_mask_sha256": sha256(mask_path),
        })
    write_tsv(output / "paired_input_audit.tsv", input_rows)

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    all_tiles: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    all_overlaps: list[dict[str, Any]] = []
    all_sensitivity: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    annotation_audit: dict[str, dict[str, Any]] = {}
    seed_diagnostics: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = checkpoint_root / RUN_TEMPLATE.format(seed=seed)
        summary_path = run_dir / "summary.json"
        checkpoint_path = run_dir / "best_validation_queue.pt"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_checkpoint_hash = str(summary["model"]["checkpoint_sha256"])
        actual_checkpoint_hash = sha256(checkpoint_path)
        if actual_checkpoint_hash != expected_checkpoint_hash:
            raise ValueError(f"Checkpoint hash mismatch for seed {seed}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint["seed"]) != seed:
            raise ValueError(f"Checkpoint seed mismatch for {seed}")
        selection = checkpoint["selection"]
        threshold = float(selection["threshold"])
        minimum_area = int(selection["minimum_area_pixels"])
        model = build_model()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        checkpoint_rows.append({
            "seed": seed,
            "checkpoint_relative_path": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": actual_checkpoint_hash,
            "validation_selected_epoch": int(selection["epoch"]),
            "validation_selected_threshold": threshold,
            "validation_selected_minimum_area": minimum_area,
        })

        torch.cuda.reset_peak_memory_stats(device)
        inference_started = time.perf_counter()
        seed_candidates: list[dict[str, Any]] = []
        for image_id, image_path, annotation_path, _ in triplets:
            # RGB-only candidate generation and sorting are complete before the
            # annotation is decoded for offline scoring below.
            rgb, geometry = load_rgb(image_path)
            normalized = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                probabilities = model(tensor).softmax(dim=1)[0].float().cpu().numpy()
            labels, candidates = generate_candidates(seed, image_id, probabilities, threshold, minimum_area)

            semantic, colors, channel_overlap = load_annotation(annotation_path, geometry)
            if image_id not in annotation_audit:
                annotation_audit[image_id] = {
                    "image_id": image_id,
                    **geometry,
                    "unique_annotation_rgb_values": json.dumps(colors, separators=(",", ":")),
                    "crop_weed_channel_overlap_pixels": channel_overlap,
                    "crop_pixels_on_canvas": int((semantic == 1).sum()),
                    "weed_pixels_on_canvas": int((semantic == 2).sum()),
                }
            tile, candidate_rows, overlap_rows, sensitivity_rows, _ = score_image(
                seed, image_id, probabilities, labels, candidates, semantic
            )
            all_tiles.append(tile)
            all_candidates.extend(candidate_rows)
            seed_candidates.extend(candidate_rows)
            all_overlaps.extend(overlap_rows)
            all_sensitivity.extend(sensitivity_rows)
        inference_seconds = time.perf_counter() - inference_started
        targets = np.asarray([row["candidate_truth"] == "weed" for row in seed_candidates], dtype=np.int64)
        scores = np.asarray([float(row["score"]) for row in seed_candidates], dtype=np.float64)
        diagnostic = {
            "seed": seed,
            "all_candidates": len(seed_candidates),
            "candidate_AUC_diagnostic": float(roc_auc_score(targets, scores)) if len(np.unique(targets)) == 2 else float("nan"),
            "candidate_AP_diagnostic": float(average_precision_score(targets, scores)) if targets.any() else float("nan"),
            "inference_and_scoring_seconds": inference_seconds,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        seed_diagnostics.append(diagnostic)
        del model, checkpoint
        torch.cuda.empty_cache()

    write_tsv(output / "checkpoint_audit.tsv", checkpoint_rows)
    write_tsv(output / "annotation_semantics_audit.tsv", list(annotation_audit.values()))
    write_tsv(output / "per_image_seed_metrics.tsv", all_tiles)
    write_tsv(output / "candidate_diagnostics.tsv", all_candidates)
    write_tsv(output / "accepted_candidate_overlap_matching_audit.tsv", all_overlaps)
    write_tsv(output / "truth_component_area_sensitivity.tsv", all_sensitivity)

    tiles_by_seed = {seed: sorted([row for row in all_tiles if int(row["seed"]) == seed], key=lambda row: row["image_id"]) for seed in SEEDS}
    image_ids = [row["image_id"] for row in tiles_by_seed[SEEDS[0]]]
    if len(image_ids) != 60 or any([row["image_id"] for row in tiles_by_seed[seed]] != image_ids for seed in SEEDS):
        raise ValueError("Seed/image pairing contract failed")
    point_indices = np.arange(len(image_ids), dtype=np.int64)
    seed_points = {seed: aggregate_tiles(tiles_by_seed[seed], point_indices) for seed in SEEDS}
    metrics = (
        "all_candidate_precision",
        "many_to_one_spatial_recall",
        "many_to_one_role_qualified_recall",
        "one_to_one_spatial_recall",
        "one_to_one_role_qualified_recall",
        "crop_overlap_image_frequency",
        "mean_candidates_per_image",
        "weed_iou",
        "mean_iou",
    )
    rng = np.random.default_rng(20260812)
    replicates = 10000
    seed_bootstrap = {seed: {metric: np.empty(replicates) for metric in metrics} for seed in SEEDS}
    mean_bootstrap = {metric: np.empty(replicates) for metric in metrics}
    for replicate in range(replicates):
        sampled = rng.integers(0, len(image_ids), size=len(image_ids))
        estimates = {seed: aggregate_tiles(tiles_by_seed[seed], sampled) for seed in SEEDS}
        for metric in metrics:
            for seed in SEEDS:
                seed_bootstrap[seed][metric][replicate] = estimates[seed][metric]
            mean_bootstrap[metric][replicate] = float(np.mean([estimates[seed][metric] for seed in SEEDS]))
    uncertainty_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for metric in metrics:
            low, high = percentile(seed_bootstrap[seed][metric])
            uncertainty_rows.append({
                "summary_unit": f"seed_{seed}",
                "metric": metric,
                "estimate": seed_points[seed][metric],
                "ci95_low": low,
                "ci95_high": high,
                "cluster_unit": "image",
                "clusters": 60,
                "bootstrap_replicates": replicates,
            })
    mean_points = {metric: float(np.mean([seed_points[seed][metric] for seed in SEEDS])) for metric in metrics}
    for metric in metrics:
        low, high = percentile(mean_bootstrap[metric])
        uncertainty_rows.append({
            "summary_unit": "five_seed_mean",
            "metric": metric,
            "estimate": mean_points[metric],
            "ci95_low": low,
            "ci95_high": high,
            "cluster_unit": "image",
            "clusters": 60,
            "bootstrap_replicates": replicates,
        })
    write_tsv(output / "image_cluster_bootstrap.tsv", uncertainty_rows)
    write_tsv(output / "seed_candidate_diagnostics.tsv", seed_diagnostics)

    sensitivity_summary: list[dict[str, Any]] = []
    for minimum_area in TRUTH_AREA_GRID:
        for seed in SEEDS:
            rows = [row for row in all_sensitivity if int(row["seed"]) == seed and int(row["truth_minimum_area"]) == minimum_area]
            weeds = sum(int(row["weed_components"]) for row in rows)
            sensitivity_summary.append({
                "seed": seed,
                "truth_minimum_area": minimum_area,
                "images": len(rows),
                "weed_components": weeds,
                "many_to_one_spatial_recall": sum(int(row["many_to_one_spatial_recalled"]) for row in rows) / max(1, weeds),
                "many_to_one_role_qualified_recall": sum(int(row["many_to_one_role_qualified_recalled"]) for row in rows) / max(1, weeds),
                "one_to_one_spatial_recall": sum(int(row["one_to_one_spatial_recalled"]) for row in rows) / max(1, weeds),
                "one_to_one_role_qualified_recall": sum(int(row["one_to_one_role_qualified_recalled"]) for row in rows) / max(1, weeds),
            })
    write_tsv(output / "truth_component_area_sensitivity_summary.tsv", sensitivity_summary)

    primary_rows = [row for row in uncertainty_rows if row["summary_unit"] == "five_seed_mean" and row["metric"] == "one_to_one_role_qualified_recall"]
    summary = {
        "run_id": args.run_id,
        "status": "completed_locked_cross_dataset_evaluation",
        "source_revision": COMMIT,
        "images": 60,
        "seeds": list(SEEDS),
        "queue_policy": "fixed_K20_per_image",
        "external_tuning": False,
        "primary_endpoint": primary_rows[0],
        "seed_point_estimates": seed_points,
        "five_seed_mean_point_estimates": mean_points,
        "optimization_seed_sample_sd": {
            metric: float(np.std([seed_points[seed][metric] for seed in SEEDS], ddof=1)) for metric in metrics
        },
        "candidate_diagnostics": seed_diagnostics,
        "inputs": {
            "config_sha256": sha256(config),
            "dataset_contract_sha256": sha256(contract),
            "script_sha256": sha256(Path(__file__).resolve()),
            "upstream_tree_inventory_sha256": sha256(dataset_root / "UPSTREAM_GIT_TREE.json"),
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "physical_gpu_id": args.physical_gpu_id,
            "gpu": torch.cuda.get_device_name(0),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "governance": {
            "analysis_class": "locked_external_evaluation",
            "external_labels_for_training_or_selection": False,
            "candidate_generation_before_annotation_decode": True,
            "negative_transfer_preserved": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "SHA256SUMS.tsv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("path\tsha256\n")
        for path in sorted(output.iterdir()):
            if path.name != "SHA256SUMS.tsv":
                stream.write(f"{path.name}\t{sha256(path)}\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"LOCKED_EXTERNAL_EVALUATION_FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
