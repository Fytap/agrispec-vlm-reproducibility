#!/usr/bin/env python3
"""Train and evaluate a target-mask semantic queue baseline.

The official training split supplies dense-mask supervision. Epoch and
component settings are selected on official validation by the final K=20
one-to-one queue metric. Official-test labels are opened only after that
selection and are never used for threshold, epoch, component, candidate-ID,
or queue-policy selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


THRESHOLDS = (0.30, 0.50, 0.70)
MIN_AREAS = (16, 32, 64)
IOU_THRESHOLDS = (0.25, 0.50, 0.75)
ROLE_INSTANCE_COVERAGE = 0.50
ROLE_LABELED_COVERAGE = 0.50
ROLE_PURITY = 0.90
MEAN_BURDEN_LIMIT = 100.0
QUEUE_K = 20
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--physical-gpu-ids", default="0,1,2,5,6,7")
    parser.add_argument("--training-tensor-cache", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def extract_public_archive(archive: Path, cache_dir: Path, expected_hash: str) -> Path:
    marker = cache_dir / "EXTRACTED_FROM_SHA256.txt"
    dataset = cache_dir / "weedsgalore-dataset"
    if marker.is_file() and dataset.is_dir():
        if marker.read_text(encoding="utf-8").strip() != expected_hash:
            raise ValueError("Existing extraction cache hash does not match the frozen archive")
        return dataset
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"Nonempty unverified cache: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(cache_dir)
    marker.write_text(expected_hash + "\n", encoding="utf-8")
    if not dataset.is_dir():
        raise ValueError("Extracted dataset root is missing")
    return dataset


def read_split(dataset_root: Path, split: str) -> list[str]:
    values = (dataset_root / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    expected = {"train": 104, "val": 26, "test": 26}[split]
    if len(values) != expected or len(values) != len(set(values)):
        raise ValueError(f"Unexpected official {split} split")
    return values


def paths_for_stem(root: Path, stem: str) -> dict[str, Path]:
    date = stem[:10]
    base = root / date
    return {
        "r": base / "images" / f"{stem}_R.png",
        "g": base / "images" / f"{stem}_G.png",
        "b": base / "images" / f"{stem}_B.png",
        "semantic": base / "semantics" / f"{stem}.png",
        "instances": base / "instances" / f"{stem}.png",
    }


def load_rgb(paths: dict[str, Path]) -> np.ndarray:
    channels = []
    for key in ("r", "g", "b"):
        with Image.open(paths[key]) as image:
            channels.append(np.asarray(image))
    rgb = np.stack(channels, axis=-1)
    if rgb.shape != (600, 600, 3):
        raise ValueError(f"Unexpected RGB shape: {rgb.shape}")
    return rgb


def load_semantic(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        raw = np.asarray(image)
    target = np.zeros(raw.shape, dtype=np.int64)
    target[raw == 1] = 1
    target[raw > 1] = 2
    return target


class TargetDataset:
    """Factory namespace; the actual torch Dataset is built after torch import."""

    @staticmethod
    def build(
        stems: list[str],
        root: Path,
        train: bool,
        seed: int,
        tensor_cache: Path | None = None,
    ):
        import torch
        from torch.utils.data import Dataset

        cached_rgb = None
        cached_semantic = None
        if train and tensor_cache is not None:
            manifest = json.loads((tensor_cache / "manifest.json").read_text(encoding="utf-8"))
            if manifest["stems"] != stems:
                raise ValueError("Training tensor-cache stems differ from the official training split")
            cached_rgb = np.load(tensor_cache / "rgb_uint16.npy", mmap_mode="r")
            cached_semantic = np.load(tensor_cache / "semantic_uint8.npy", mmap_mode="r")
            if cached_rgb.shape != (len(stems), 600, 600, 3):
                raise ValueError("Unexpected cached RGB tensor shape")
            if cached_semantic.shape != (len(stems), 600, 600):
                raise ValueError("Unexpected cached semantic tensor shape")

        class DatasetImpl(Dataset):
            def __len__(self) -> int:
                return len(stems)

            def __getitem__(self, index: int):
                stem = stems[index]
                if cached_rgb is None:
                    paths = paths_for_stem(root, stem)
                    rgb = load_rgb(paths)
                    target = load_semantic(paths["semantic"])
                else:
                    rgb = np.asarray(cached_rgb[index])
                    target = np.asarray(cached_semantic[index], dtype=np.int64)
                if train:
                    worker = torch.utils.data.get_worker_info()
                    worker_seed = 0 if worker is None else worker.seed
                    rng = np.random.default_rng(seed + index * 1000003 + int(worker_seed % 1000000007))
                    y0 = int(rng.integers(0, 600 - 512 + 1))
                    x0 = int(rng.integers(0, 600 - 512 + 1))
                    rgb = rgb[y0 : y0 + 512, x0 : x0 + 512]
                    target = target[y0 : y0 + 512, x0 : x0 + 512]
                    if rng.random() < 0.5:
                        rgb, target = rgb[:, ::-1], target[:, ::-1]
                    if rng.random() < 0.5:
                        rgb, target = rgb[::-1], target[::-1]
                x = rgb.astype(np.float32) / 255.0
                x = (x - IMAGENET_MEAN) / IMAGENET_STD
                x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
                y = torch.from_numpy(np.ascontiguousarray(target))
                return x, y, stem

        return DatasetImpl()


def build_model():
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

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

        def forward(self, x):
            return self.layers(x)

    class ResNet18UNet(nn.Module):
        def __init__(self):
            super().__init__()
            encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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

        def forward(self, x):
            output_size = x.shape[-2:]
            s0 = self.stem(x)
            s1 = self.layer1(self.pool(s0))
            s2 = self.layer2(s1)
            s3 = self.layer3(s2)
            s4 = self.layer4(s3)
            up = lambda value, reference: F.interpolate(
                value, size=reference.shape[-2:], mode="bilinear", align_corners=False
            )
            d4 = self.dec4(torch.cat([up(s4, s3), s3], dim=1))
            d3 = self.dec3(torch.cat([up(d4, s2), s2], dim=1))
            d2 = self.dec2(torch.cat([up(d3, s1), s1], dim=1))
            d1 = self.dec1(torch.cat([up(d2, s0), s0], dim=1))
            return self.head(F.interpolate(d1, size=output_size, mode="bilinear", align_corners=False))

    return ResNet18UNet()


def semantic_metrics(probabilities: dict[str, np.ndarray], root: Path) -> dict[str, float]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for stem, probs in probabilities.items():
        truth = load_semantic(paths_for_stem(root, stem)["semantic"])
        pred = probs.argmax(axis=0)
        flat = 3 * truth.ravel() + pred.ravel()
        confusion += np.bincount(flat, minlength=9).reshape(3, 3)
    ious = []
    result: dict[str, float] = {}
    for role, index in (("background", 0), ("crop", 1), ("weed", 2)):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        iou = tp / max(1, tp + fp + fn)
        result[f"{role}_iou"] = iou
        ious.append(iou)
    result["mean_iou"] = float(np.mean(ious))
    result["plant_role_mean_iou"] = float(np.mean(ious[1:]))
    return result


@dataclass
class TruthInstance:
    instance_id: str
    role: str
    area: int
    value: int


@lru_cache(maxsize=64)
def truth_instances(stem: str, root: Path) -> tuple[np.ndarray, np.ndarray, list[TruthInstance]]:
    paths = paths_for_stem(root, stem)
    semantic = load_semantic(paths["semantic"])
    with Image.open(paths["instances"]) as image:
        instance_map = np.asarray(image)
    maximum = int(instance_map.max())
    areas = np.bincount(instance_map.ravel(), minlength=maximum + 1)
    crop_counts = np.bincount(instance_map[semantic == 1].ravel(), minlength=maximum + 1)
    weed_counts = np.bincount(instance_map[semantic == 2].ravel(), minlength=maximum + 1)
    records: list[TruthInstance] = []
    for value in range(1, maximum + 1):
        crop = int(crop_counts[value])
        weed = int(weed_counts[value])
        if crop + weed == 0:
            continue
        records.append(
            TruthInstance(
                instance_id=f"{stem}:instance:{value}",
                role="crop" if crop >= weed else "weed",
                area=int(areas[value]),
                value=value,
            )
        )
    return semantic, instance_map, records


def component_candidates(
    stem: str,
    probs: np.ndarray,
    root: Path,
    threshold: float,
    minimum_area: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, dict[str, set[str]]],
    dict[str, float],
]:
    semantic, instance_map, instances = truth_instances(stem, root)
    instance_by_value = {record.value: record for record in instances}
    plant_probability = 1.0 - probs[0]
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, count = ndimage.label(plant_probability >= threshold, structure=structure)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    candidates: list[dict[str, Any]] = []
    spatial_edges: dict[str, set[str]] = defaultdict(set)
    qualified_edges: dict[str, set[str]] = defaultdict(set)
    iou_edges: dict[str, dict[str, set[str]]] = {
        f"{value:.2f}": defaultdict(set) for value in IOU_THRESHOLDS
    }
    best_iou: dict[str, float] = {record.instance_id: 0.0 for record in instances if record.role == "weed"}
    for component in range(1, count + 1):
        area = int(areas[component])
        if area < minimum_area:
            continue
        mask = labels == component
        ys, xs = np.nonzero(mask)
        crop_pixels = int((semantic[mask] == 1).sum())
        weed_pixels = int((semantic[mask] == 2).sum())
        labeled_pixels = crop_pixels + weed_pixels
        labeled_coverage = labeled_pixels / max(1, area)
        crop_purity = crop_pixels / max(1, labeled_pixels)
        weed_purity = weed_pixels / max(1, labeled_pixels)
        candidate_role = (
            "weed"
            if labeled_coverage >= ROLE_LABELED_COVERAGE and weed_purity >= ROLE_PURITY
            else "crop"
            if labeled_coverage >= ROLE_LABELED_COVERAGE and crop_purity >= ROLE_PURITY
            else "ambiguous_or_background"
        )
        plant_mean = float((probs[1][mask] + probs[2][mask]).mean())
        weed_mean = float(probs[2][mask].mean())
        score = weed_mean / max(1e-8, plant_mean)
        candidate_id = f"{stem}:semantic:T{int(round(threshold * 100)):03d}:A{minimum_area:03d}:C{component:05d}"
        row = {
            "candidate_id": candidate_id,
            "file": stem,
            "date": stem[:10],
            "area_pixels": area,
            "x0": int(xs.min()),
            "y0": int(ys.min()),
            "x1_exclusive": int(xs.max()) + 1,
            "y1_exclusive": int(ys.max()) + 1,
            "score": score,
            "candidate_truth": candidate_role,
            "candidate_labeled_coverage": labeled_coverage,
            "candidate_crop_purity": crop_purity,
            "candidate_weed_purity": weed_purity,
            "crop_fraction_candidate": crop_pixels / max(1, area),
            "weed_fraction_candidate": weed_pixels / max(1, area),
        }
        candidates.append(row)
        overlap_values, overlap_counts = np.unique(instance_map[mask], return_counts=True)
        for instance_value, overlap_count in zip(overlap_values.tolist(), overlap_counts.tolist()):
            record = instance_by_value.get(int(instance_value))
            if record is None:
                continue
            overlap = int(overlap_count)
            coverage = overlap / record.area
            union = area + record.area - overlap
            iou = overlap / max(1, union)
            if record.role == "weed":
                best_iou[record.instance_id] = max(best_iou[record.instance_id], iou)
                if coverage >= ROLE_INSTANCE_COVERAGE:
                    spatial_edges[candidate_id].add(record.instance_id)
                    if labeled_coverage >= ROLE_LABELED_COVERAGE and weed_purity >= ROLE_PURITY:
                        qualified_edges[candidate_id].add(record.instance_id)
                for value in IOU_THRESHOLDS:
                    if iou >= value:
                        iou_edges[f"{value:.2f}"][candidate_id].add(record.instance_id)
    candidates.sort(key=lambda row: (-float(row["score"]), str(row["candidate_id"])))
    return candidates, spatial_edges, qualified_edges, iou_edges, best_iou


def one_to_one_count(accepted: list[dict[str, Any]], edges: dict[str, set[str]]) -> int:
    instance_ids = sorted({instance for row in accepted for instance in edges.get(str(row["candidate_id"]), set())})
    if not accepted or not instance_ids:
        return 0
    index = {value: i for i, value in enumerate(instance_ids)}
    rows: list[int] = []
    cols: list[int] = []
    for row_index, row in enumerate(accepted):
        for instance in edges.get(str(row["candidate_id"]), set()):
            rows.append(row_index)
            cols.append(index[instance])
    if not rows:
        return 0
    matrix = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(accepted), len(instance_ids)))
    matching = maximum_bipartite_matching(matrix, perm_type="column")
    return int((matching >= 0).sum())


def evaluate_probabilities(
    probabilities: dict[str, np.ndarray],
    root: Path,
    threshold: float,
    minimum_area: int,
    include_candidate_rows: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    all_candidates: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []
    totals = Counter()
    crop_fractions: list[float] = []
    dates: dict[str, Counter] = defaultdict(Counter)
    for stem in sorted(probabilities):
        candidates, spatial_edges, qualified_edges, iou_edges, _ = component_candidates(
            stem, probabilities[stem], root, threshold, minimum_area
        )
        accepted = candidates[:QUEUE_K]
        _, _, instances = truth_instances(stem, root)
        weed_ids = {record.instance_id for record in instances if record.role == "weed"}
        crop_ids = {record.instance_id for record in instances if record.role == "crop"}
        spatial = set().union(*(spatial_edges.get(str(row["candidate_id"]), set()) for row in accepted)) if accepted else set()
        qualified = set().union(*(qualified_edges.get(str(row["candidate_id"]), set()) for row in accepted)) if accepted else set()
        one_to_one = one_to_one_count(accepted, qualified_edges)
        eligible_weed = sum(str(row["candidate_truth"]) == "weed" for row in accepted)
        crop_overlap = [float(row["crop_fraction_candidate"]) for row in accepted]
        crop_fractions.extend(crop_overlap)
        row = {
            "file": stem,
            "date": stem[:10],
            "weed_instances": len(weed_ids),
            "crop_instances": len(crop_ids),
            "candidates": len(candidates),
            "accepted_candidates": len(accepted),
            "eligible_weed_candidates": eligible_weed,
            "all_candidate_precision": eligible_weed / max(1, len(accepted)),
            "many_to_one_spatial_recalled": len(spatial),
            "many_to_one_role_qualified_recalled": len(qualified),
            "one_to_one_role_qualified_recalled": one_to_one,
            "crop_overlap_scene": int(any(value > 0 for value in crop_overlap)),
            "maximum_crop_fraction": max(crop_overlap, default=0.0),
        }
        tile_rows.append(row)
        date = dates[stem[:10]]
        for key in (
            "weed_instances",
            "crop_instances",
            "candidates",
            "accepted_candidates",
            "eligible_weed_candidates",
            "many_to_one_spatial_recalled",
            "many_to_one_role_qualified_recalled",
            "one_to_one_role_qualified_recalled",
            "crop_overlap_scene",
        ):
            totals[key] += int(row[key])
            date[key] += int(row[key])
        totals["images"] += 1
        date["images"] += 1
        for value in IOU_THRESHOLDS:
            recalled_ids = set().union(*(
                iou_edges[f"{value:.2f}"].get(str(row["candidate_id"]), set()) for row in accepted
            )) if accepted else set()
            recalled = len(recalled_ids & weed_ids)
            totals[f"iou_{value:.2f}_recalled"] += recalled
            date[f"iou_{value:.2f}_recalled"] += recalled
        if include_candidate_rows:
            for rank, candidate in enumerate(candidates, 1):
                all_candidates.append({**candidate, "rank_within_image": rank, "accepted_K20": int(rank <= QUEUE_K)})
    summary: dict[str, Any] = {
        "images": totals["images"],
        "weed_instances": totals["weed_instances"],
        "crop_instances": totals["crop_instances"],
        "candidates": totals["candidates"],
        "candidates_per_image": totals["candidates"] / max(1, totals["images"]),
        "accepted_candidates": totals["accepted_candidates"],
        "eligible_weed_candidates": totals["eligible_weed_candidates"],
        "K20_all_candidate_precision": totals["eligible_weed_candidates"] / max(1, totals["accepted_candidates"]),
        "K20_many_to_one_spatial_recall": totals["many_to_one_spatial_recalled"] / max(1, totals["weed_instances"]),
        "K20_many_to_one_role_qualified_recall": totals["many_to_one_role_qualified_recalled"] / max(1, totals["weed_instances"]),
        "K20_one_to_one_role_qualified_recall": totals["one_to_one_role_qualified_recalled"] / max(1, totals["weed_instances"]),
        "K20_crop_overlap_scene_frequency": totals["crop_overlap_scene"] / max(1, totals["images"]),
        "queued_crop_fraction_median": float(np.median(crop_fractions)) if crop_fractions else 0.0,
        "queued_crop_fraction_q75": float(np.quantile(crop_fractions, 0.75)) if crop_fractions else 0.0,
        "queued_crop_fraction_q95": float(np.quantile(crop_fractions, 0.95)) if crop_fractions else 0.0,
        "queued_crop_fraction_maximum": max(crop_fractions, default=0.0),
    }
    for value in IOU_THRESHOLDS:
        summary[f"K20_AR_IoU_{value:.2f}"] = totals[f"iou_{value:.2f}_recalled"] / max(1, totals["weed_instances"])
    date_rows: list[dict[str, Any]] = []
    for date, values in sorted(dates.items()):
        date_rows.append({
            "date": date,
            "images": values["images"],
            "weed_instances": values["weed_instances"],
            "candidates": values["candidates"],
            "accepted_candidates": values["accepted_candidates"],
            "precision": values["eligible_weed_candidates"] / max(1, values["accepted_candidates"]),
            "many_to_one_spatial_recall": values["many_to_one_spatial_recalled"] / max(1, values["weed_instances"]),
            "many_to_one_role_qualified_recall": values["many_to_one_role_qualified_recalled"] / max(1, values["weed_instances"]),
            "one_to_one_role_qualified_recall": values["one_to_one_role_qualified_recalled"] / max(1, values["weed_instances"]),
            "crop_overlap_scene_frequency": values["crop_overlap_scene"] / max(1, values["images"]),
        })
    return summary, tile_rows, all_candidates + date_rows


def predict(model, stems: list[str], root: Path, device) -> tuple[dict[str, np.ndarray], float, int]:
    import torch

    model.eval()
    output: dict[str, np.ndarray] = {}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for stem in stems:
            rgb = load_rgb(paths_for_stem(root, stem)).astype(np.float32) / 255.0
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
            x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            output[stem] = logits.softmax(dim=1)[0].float().cpu().numpy()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    return output, elapsed, peak


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    physical = [int(value) for value in args.physical_gpu_ids.split(",")]
    expected_visible = ",".join(str(value) for value in physical)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible or world_size != len(physical):
        raise RuntimeError("Frozen GPU assignment mismatch")
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    root = args.project_root.resolve()
    archive = args.archive.resolve()
    config = args.config.resolve()
    output = args.output_dir.resolve()
    cache = args.cache_dir.resolve()
    if sha256(archive) != args.expected_archive_sha256:
        raise ValueError("Frozen public archive hash mismatch")
    exists = bool(output.exists())
    if distributed:
        exists_tensor = torch.tensor(int(rank == 0 and exists), device=device)
        dist.broadcast(exists_tensor, src=0)
        exists = bool(int(exists_tensor))
    if exists:
        raise FileExistsError(f"Refusing to overwrite {output}")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        dataset_root = extract_public_archive(archive, cache, args.expected_archive_sha256)
    if distributed:
        dist.barrier()
    dataset_root = cache / "weedsgalore-dataset"
    train_stems = read_split(dataset_root, "train")
    val_stems = read_split(dataset_root, "val")
    test_stems = read_split(dataset_root, "test")
    if set(train_stems) & set(val_stems) or set(train_stems) & set(test_stems) or set(val_stems) & set(test_stems):
        raise ValueError("Official split overlap")

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    train_dataset = TargetDataset.build(
        train_stems,
        dataset_root,
        train=True,
        seed=args.seed,
        tensor_cache=args.training_tensor_cache,
    )
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    loader_workers = 0 if os.name == "nt" and world_size == 1 else 2
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        sampler=sampler,
        num_workers=loader_workers,
        pin_memory=True,
        persistent_workers=loader_workers > 0,
        drop_last=False,
    )
    model = build_model().to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    base_model = model.module if distributed else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    class_weights = torch.tensor([0.15, 1.0, 1.0], device=device)
    history: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    best_key: tuple[Any, ...] | None = None
    best_selection: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        aggregate = torch.zeros(2, device=device, dtype=torch.float64)
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                cross_entropy = F.cross_entropy(logits, y, weight=class_weights)
                probabilities = logits.softmax(dim=1)
                one_hot = F.one_hot(y, 3).permute(0, 3, 1, 2).float()
                intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum(dim=(0, 2, 3))
                denominator = probabilities[:, 1:].sum(dim=(0, 2, 3)) + one_hot[:, 1:].sum(dim=(0, 2, 3))
                dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
                loss = cross_entropy + dice_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            aggregate += torch.tensor([float(loss.detach()), 1.0], device=device)
        if distributed:
            dist.all_reduce(aggregate)
            dist.barrier()
        if rank == 0:
            validation_probabilities, validation_seconds, validation_peak = predict(
                base_model, val_stems, dataset_root, device
            )
            seg = semantic_metrics(validation_probabilities, dataset_root)
            epoch_rows: list[dict[str, Any]] = []
            for threshold in THRESHOLDS:
                for minimum_area in MIN_AREAS:
                    queue_summary, _, _ = evaluate_probabilities(
                        validation_probabilities, dataset_root, threshold, minimum_area
                    )
                    feasible = float(queue_summary["candidates_per_image"]) <= MEAN_BURDEN_LIMIT
                    record = {
                        "epoch": epoch,
                        "threshold": threshold,
                        "minimum_area_pixels": minimum_area,
                        "feasible_mean_burden": int(feasible),
                        **seg,
                        **queue_summary,
                    }
                    epoch_rows.append(record)
                    selection_rows.append(record)
                    if feasible:
                        key = (
                            float(record["K20_one_to_one_role_qualified_recall"]),
                            float(record["K20_many_to_one_role_qualified_recall"]),
                            float(record["K20_all_candidate_precision"]),
                            -float(record["candidates_per_image"]),
                            float(threshold),
                            int(minimum_area),
                            -int(epoch),
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_selection = record
                            torch.save(
                                {
                                    "model_state": base_model.state_dict(),
                                    "selection": record,
                                    "epoch": epoch,
                                    "seed": args.seed,
                                },
                                output / "best_validation_queue.pt",
                            )
            chosen_this_epoch = max(
                (row for row in epoch_rows if int(row["feasible_mean_burden"])),
                key=lambda row: (
                    float(row["K20_one_to_one_role_qualified_recall"]),
                    float(row["K20_many_to_one_role_qualified_recall"]),
                    float(row["K20_all_candidate_precision"]),
                    -float(row["candidates_per_image"]),
                    float(row["threshold"]),
                    int(row["minimum_area_pixels"]),
                ),
            )
            history_row = {
                "epoch": epoch,
                "mean_train_loss": float(aggregate[0] / aggregate[1]),
                "validation_seconds": validation_seconds,
                "validation_peak_memory_bytes": validation_peak,
                "validation_mean_iou": seg["mean_iou"],
                "validation_plant_role_mean_iou": seg["plant_role_mean_iou"],
                "epoch_selected_threshold": chosen_this_epoch["threshold"],
                "epoch_selected_minimum_area": chosen_this_epoch["minimum_area_pixels"],
                "epoch_K20_one_to_one_role_qualified_recall": chosen_this_epoch["K20_one_to_one_role_qualified_recall"],
                "global_best_epoch_so_far": int(best_selection["epoch"] if best_selection else epoch),
            }
            history.append(history_row)
            print(json.dumps(history_row), flush=True)
        if distributed:
            dist.barrier()

    if rank == 0:
        if best_selection is None:
            raise RuntimeError("No validation candidate setting satisfied the registered mean burden")
        checkpoint = torch.load(output / "best_validation_queue.pt", map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["model_state"])
        selected_threshold = float(best_selection["threshold"])
        selected_area = int(best_selection["minimum_area_pixels"])
        # The official test is opened only below, after epoch and setting selection is complete.
        test_probabilities, test_seconds, test_peak = predict(base_model, test_stems, dataset_root, device)
        test_segmentation = semantic_metrics(test_probabilities, dataset_root)
        test_summary, test_tile_rows, combined_rows = evaluate_probabilities(
            test_probabilities,
            dataset_root,
            selected_threshold,
            selected_area,
            include_candidate_rows=True,
        )
        test_candidate_rows = [row for row in combined_rows if "candidate_id" in row]
        test_date_rows = [row for row in combined_rows if "candidate_id" not in row]
        write_tsv(output / "training_history.tsv", history)
        write_tsv(output / "validation_epoch_setting_selection.tsv", selection_rows)
        write_tsv(output / "official_test_tile_metrics.tsv", test_tile_rows)
        write_tsv(output / "official_test_date_metrics.tsv", test_date_rows)
        write_tsv(output / "official_test_candidates.tsv", test_candidate_rows)
        governance = {
            "analysis_class": "post_test_seed_stability_analysis",
            "reason": "independent-seed stability analysis under equal target-mask access",
            "pre_test_frozen_items": [],
            "post_test_registered_before_this_run": {
                "config": str(config),
                "config_sha256": sha256(config),
                "selection_role": "official validation only",
                "selection_metric": "K20 one-to-one role-qualified weed-instance recall",
                "thresholds": THRESHOLDS,
                "minimum_areas": MIN_AREAS,
                "epochs": args.epochs,
                "seed": args.seed,
            },
            "test_labels_not_used_for": [
                "epoch selection",
                "foreground threshold selection",
                "minimum component area selection",
                "candidate-ID selection",
                "training updates",
            ],
            "test_use": "offline scoring after validation-only selection",
        }
        (output / "test_governance.json").write_text(json.dumps(governance, indent=2) + "\n", encoding="utf-8")
        result = {
            "run_id": args.run_id,
            "status": "completed_post_test_seed_stability_analysis",
            "git_commit": git_commit(root),
            "inputs": {
                "archive": str(archive),
                "archive_sha256": sha256(archive),
                "config": str(config),
                "config_sha256": sha256(config),
                "training_tensor_cache": str(args.training_tensor_cache.resolve()) if args.training_tensor_cache else None,
                "training_tensor_cache_manifest_sha256": (
                    sha256(args.training_tensor_cache / "manifest.json") if args.training_tensor_cache else None
                ),
            },
            "training": {
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size_per_gpu": args.batch_size_per_gpu,
                "world_size": world_size,
                "global_batch_size": args.batch_size_per_gpu * world_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            "split_counts": {"train": len(train_stems), "validation": len(val_stems), "test": len(test_stems)},
            "model": {
                "architecture": "ResNet18-UNet",
                "encoder_initialization": "ImageNet-1K ResNet18 public weights",
                "classes": ["background", "crop", "weed"],
                "checkpoint_sha256": sha256(output / "best_validation_queue.pt"),
            },
            "selected_on_validation": best_selection,
            "official_test_semantic_metrics": test_segmentation,
            "official_test_queue_metrics": test_summary,
            "runtime": {
                "test_total_seconds": test_seconds,
                "test_seconds_per_tile": test_seconds / len(test_stems),
                "test_peak_gpu_memory_bytes": test_peak,
                "run_wall_seconds": time.perf_counter() - run_started,
                "physical_gpu_ids_training": physical,
                "evaluation_local_device": physical[0],
            },
            "software": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
            "governance": governance,
            "limitations": [
                "This is a post-test descriptive stability analysis rather than a new confirmatory test.",
                "The official split contains spatial patches from one field and does not support cross-farm inference.",
                "Training seeds are the independent units for the stability summary.",
            ],
        }
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
