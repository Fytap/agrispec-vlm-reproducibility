#!/usr/bin/env python3
"""Eight-GPU, nonsealed RGB binary-foreground proposal training.

This script reads *only* ``specialist_train`` supervision.  Crop/weed masks
are collapsed to foreground solely for supervised patch selection and targets;
they are never used to generate inference candidates.  A separate future
specialist-validation diagnostic must choose any proposal threshold.
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
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


CROP_IDS = {10000, 10001, 10002}
WEED_IDS = {2} | set(range(20000, 20012)) | set(range(20100, 20106))
IGNORE_INDEX = 255


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def foreground(values: np.ndarray) -> np.ndarray:
    return np.isin(values, list(CROP_IDS | WEED_IDS)).astype(np.float32)


def crop_pad(array: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    height, width = array.shape[:2]
    crop = array[y0 : min(height, y0 + size), x0 : min(width, x0 + size)]
    bottom, right = size - crop.shape[0], size - crop.shape[1]
    if bottom == 0 and right == 0:
        return crop
    return np.pad(crop, ((0, bottom), (0, right), (0, 0)), mode="edge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--supervision-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--train-patches", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size-per-gpu", type=int, default=64)
    parser.add_argument("--num-workers-per-gpu", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--physical-gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.distributed import DistributedSampler
    from torchvision.models import ResNet18_Weights, resnet18

    rank, world_size, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    physical_gpus = [int(item) for item in args.physical_gpu_ids.split(",")]
    expected_visible = ",".join(str(item) for item in physical_gpus)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible or world_size != len(physical_gpus):
        raise RuntimeError("CUDA_VISIBLE_DEVICES/world size does not match the frozen physical GPU assignment")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if args.dry_run:
        print(json.dumps({"ddp_dry_run": True, "rank": rank, "world_size": world_size, "local_rank": local_rank, "physical_gpu": physical_gpus[local_rank]}), flush=True)
        dist.barrier(); dist.destroy_process_group(); return
    project_root, data_root = args.project_root.resolve(), args.data_root.resolve()
    manifest_path, config_path, output_dir = args.supervision_manifest.resolve(), args.config.resolve(), args.output_dir.resolve()
    exists_error = torch.tensor(int(rank == 0 and output_dir.exists()), device=device)
    dist.broadcast(exists_error, src=0)
    if int(exists_error):
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    rows = read_tsv(manifest_path)
    if any(row["split"] == "sealed_test" for row in rows):
        raise ValueError("This run refuses a manifest containing sealed rows")
    train_sources = [row for row in rows if row["split"] == "specialist_train"]
    if len(train_sources) != 7050 or any(row["split"] != "specialist_train" for row in train_sources):
        raise ValueError("Unexpected specialist_train source-manifest contract")
    crop_sources = [row for row in train_sources if int(row["crop_pixels"]) > 0]
    weed_sources = [row for row in train_sources if int(row["weed_pixels"]) > 0]
    if not crop_sources or not weed_sources:
        raise ValueError("Both crop and weed supervised sources are required")

    def build_specs() -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for index in range(args.train_patches):
            role, label_ids, candidates = ("crop", CROP_IDS, crop_sources) if index % 2 == 0 else ("weed", WEED_IDS, weed_sources)
            rng = np.random.default_rng(args.seed * 1000003 + index)
            row = candidates[int(rng.integers(len(candidates)))]
            with Image.open(data_root / row["imap_file"]) as image:
                values = np.asarray(image)
            positions = np.argwhere(np.isin(values, list(label_ids)))
            y, x = positions[int(rng.integers(len(positions)))]
            height, width = values.shape
            specs.append({"patch_index": index, "sampling_role": role, "session_id": row["session_id"], "file": row["file"], "imap_file": row["imap_file"], "x0": int(np.clip(x - args.patch_size // 2, 0, max(0, width - args.patch_size))), "y0": int(np.clip(y - args.patch_size // 2, 0, max(0, height - args.patch_size)))})
        return specs

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
        write_tsv(output_dir / "train_source_manifest.tsv", ["session_id", "split", "file", "bytes", "imap_file", "crop_pixels", "weed_pixels"], train_sources)
        write_tsv(output_dir / "train_foreground_patch_specs.tsv", ["patch_index", "sampling_role", "session_id", "file", "imap_file", "x0", "y0"], build_specs())
    dist.barrier()
    specs = read_tsv(output_dir / "train_foreground_patch_specs.tsv")

    def image_tensor(image: np.ndarray) -> Any:
        value = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
        return (value - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    class PatchDataset(Dataset):
        def __len__(self) -> int:
            return len(specs)
        def __getitem__(self, index: int) -> tuple[Any, Any]:
            spec = specs[index]
            with Image.open(data_root / spec["file"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            with Image.open(data_root / spec["imap_file"]) as image:
                target = foreground(np.asarray(image))
            rgb = crop_pad(rgb, int(spec["x0"]), int(spec["y0"]), args.patch_size)
            target = target[int(spec["y0"]) : int(spec["y0"]) + args.patch_size, int(spec["x0"]) : int(spec["x0"]) + args.patch_size]
            if target.shape != (args.patch_size, args.patch_size):
                raise ValueError("Training target patch unexpectedly crossed image boundary")
            return image_tensor(rgb), torch.from_numpy(np.ascontiguousarray(target))

    class ResNet18UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
            self.pool, self.layer1, self.layer2, self.layer3, self.layer4 = encoder.maxpool, encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4
            def block(inputs: int, outputs: int) -> nn.Sequential:
                return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True), nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))
            self.d4, self.d3, self.d2, self.d1 = block(768, 256), block(384, 128), block(192, 64), block(128, 64)
            self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))
        def forward(self, x: Any) -> Any:
            shape = x.shape[-2:]; s0 = self.stem(x); s1 = self.layer1(self.pool(s0)); s2 = self.layer2(s1); s3 = self.layer3(s2); s4 = self.layer4(s3)
            d4 = self.d4(torch.cat([functional.interpolate(s4, size=s3.shape[-2:], mode="bilinear", align_corners=False), s3], 1)); d3 = self.d3(torch.cat([functional.interpolate(d4, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1)); d2 = self.d2(torch.cat([functional.interpolate(d3, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], 1)); d1 = self.d1(torch.cat([functional.interpolate(d2, size=s0.shape[-2:], mode="bilinear", align_corners=False), s0], 1))
            return self.head(functional.interpolate(d1, size=shape, mode="bilinear", align_corners=False)).squeeze(1)

    def dice_loss(logits: Any, targets: Any) -> Any:
        probabilities = logits.sigmoid()
        return 1.0 - (2.0 * (probabilities * targets).sum() + 1.0) / (probabilities.sum() + targets.sum() + 1.0)

    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank); torch.cuda.manual_seed_all(args.seed + rank)
    sampler = DistributedSampler(PatchDataset(), num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(PatchDataset(), batch_size=args.batch_size_per_gpu, sampler=sampler, num_workers=args.num_workers_per_gpu, pin_memory=True, persistent_workers=args.num_workers_per_gpu > 0)
    model = DistributedDataParallel(ResNet18UNet().to(device), device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    positive_weight = torch.tensor(30.0, device=device)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch); model.train(); total_loss = torch.zeros(2, device=device)
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images); loss = functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight) + dice_loss(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch} rank={rank}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); total_loss += torch.tensor([float(loss.detach()), 1.0], device=device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            record = {"epoch": float(epoch), "mean_train_loss": float(total_loss[0] / total_loss[1])}; history.append(record); print(json.dumps(record), flush=True)
    dist.barrier()
    if rank == 0:
        checkpoint_path = output_dir / "last.pt"
        torch.save({"model_state": model.module.state_dict(), "epochs": args.epochs, "history": history}, checkpoint_path)
        cached_weight = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
        summary = {"run_id": args.run_id, "status": "completed_nonsealed_ddp_binary_proposal_training_no_validation_read", "scope": {"public_data_only": True, "sealed_test_read": False, "specialist_val_read": False, "controller_policy_risk_read": False, "train_split_only": "specialist_train", "model_input": "RGB_only", "iMap_use": "supervised_patch_selection_and_target_only"}, "inputs": {"supervision_manifest": str(manifest_path), "supervision_manifest_sha256": sha256_file(manifest_path), "config": str(config_path), "config_sha256": sha256_file(config_path), "train_source_manifest_sha256": sha256_file(output_dir / "train_source_manifest.tsv"), "train_patch_specs_sha256": sha256_file(output_dir / "train_foreground_patch_specs.tsv"), "encoder_weight_cache_sha256": sha256_file(cached_weight)}, "model": {"architecture": "custom_resnet18_unet", "encoder_weights": "torchvision_ResNet18_Weights.IMAGENET1K_V1_cached", "classes": ["background", "foreground"]}, "run_parameters": {"seed": args.seed, "epochs": args.epochs, "patch_size": args.patch_size, "train_patches": args.train_patches, "batch_size_per_gpu": args.batch_size_per_gpu, "world_size": world_size, "physical_gpu_ids": physical_gpus, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, "runtime": {"python": sys.version, "torch": torch.__version__, "torchvision": __import__("torchvision").__version__, "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version()}, "history": history, "checkpoint_sha256": sha256_file(checkpoint_path), "limitations": ["This training run creates a binary foreground proposal model, not crop/weed role claims or controller output.", "No validation, policy selection, risk certification, controller data, or sealed data was read.", "A later independently registered specialist-val threshold diagnostic is required before nonsealed controller inference."], "script_sha256": sha256_file(Path(__file__).resolve()), "git_commit": git_commit(project_root)}
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
