#!/usr/bin/env python3
"""Train a validation-selected DeepLabV3+ queue baseline on WeedsGalore.

The script deliberately reuses the exact candidate, matching, and K=20
evaluation contract of the archived ResNet18-UNet baseline.  Official-test
labels are read only after the epoch and component settings have been selected
on the official validation split.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import train_evaluate_p2_weedsgalore_target_semantic_seed_stability_h200 as common
except ModuleNotFoundError:  # local publication-backup layout
    backup_scripts = (
        Path(__file__).resolve().parents[1]
        / "reproducibility_backups"
        / "H200_seed_stability_20260812_v3"
        / "scripts"
    )
    sys.path.insert(0, str(backup_scripts))
    import train_evaluate_p2_weedsgalore_target_semantic_seed_stability_h200 as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--physical-gpu-id", type=int, choices=(6, 7), required=True)
    parser.add_argument("--training-tensor-cache", type=Path)
    parser.add_argument("--encoder-weights", type=Path, required=True)
    parser.add_argument("--expected-encoder-sha256", required=True)
    return parser.parse_args()


def build_model(encoder_weights: Path, expected_sha256: str):
    import segmentation_models_pytorch as smp
    import torch

    if common.sha256(encoder_weights) != expected_sha256:
        raise ValueError("Frozen ResNet50 encoder hash mismatch")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=3,
        activation=None,
    )
    # This torchvision-hosted ResNet-50 checkpoint uses PyTorch's legacy tar
    # serialization.  ``weights_only=True`` cannot read that container.  The
    # frozen SHA-256 is verified immediately above, before unpickling.
    state = torch.load(encoder_weights, map_location="cpu", weights_only=False)
    state = {key: value for key, value in state.items() if not key.startswith("fc.")}
    encoder_keys = set(model.encoder.state_dict())
    state_keys = set(state)
    missing_keys = sorted(encoder_keys - state_keys)
    unexpected_keys = sorted(state_keys - encoder_keys)
    incompatible_missing = [key for key in missing_keys if not key.endswith("num_batches_tracked")]
    if incompatible_missing or unexpected_keys:
        raise ValueError(
            f"Unexpected encoder load contract: missing={incompatible_missing}, "
            f"unexpected={unexpected_keys}"
        )
    # SMP's ResNet encoder overrides ``load_state_dict`` and returns ``None``;
    # verify the key contract explicitly above, then load strictly.
    model.encoder.load_state_dict(state, strict=False)
    class StridePaddedModel(torch.nn.Module):
        """Pad 600 px inputs to the encoder stride and crop logits back."""

        def __init__(self, segmentation_model: torch.nn.Module) -> None:
            super().__init__()
            self.segmentation_model = segmentation_model

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            height, width = inputs.shape[-2:]
            pad_height = (-height) % 16
            pad_width = (-width) % 16
            if pad_height or pad_width:
                inputs = torch.nn.functional.pad(
                    inputs,
                    (0, pad_width, 0, pad_height),
                    mode="reflect",
                )
            logits = self.segmentation_model(inputs)
            return logits[..., :height, :width]

    return StridePaddedModel(model)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    expected_visible = str(args.physical_gpu_id)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RuntimeError("Frozen GPU assignment mismatch")

    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    root = args.project_root.resolve()
    archive = args.archive.resolve()
    config = args.config.resolve()
    output = args.output_dir.resolve()
    cache = args.cache_dir.resolve()
    if common.sha256(archive) != args.expected_archive_sha256:
        raise ValueError("Frozen public archive hash mismatch")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    dataset_root = common.extract_public_archive(archive, cache, args.expected_archive_sha256)
    train_stems = common.read_split(dataset_root, "train")
    val_stems = common.read_split(dataset_root, "val")
    test_stems = common.read_split(dataset_root, "test")
    if set(train_stems) & set(val_stems) or set(train_stems) & set(test_stems) or set(val_stems) & set(test_stems):
        raise ValueError("Official split overlap")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    dataset = common.TargetDataset.build(
        train_stems,
        dataset_root,
        train=True,
        seed=args.seed,
        tensor_cache=args.training_tensor_cache,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )
    encoder_weights = args.encoder_weights.resolve()
    model = build_model(encoder_weights, args.expected_encoder_sha256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    class_weights = torch.tensor([0.15, 1.0, 1.0], device=device)
    history: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    best_key: tuple[Any, ...] | None = None
    best_selection: dict[str, Any] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = 0
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
            loss_sum += float(loss.detach())
            batches += 1

        validation_probabilities, validation_seconds, validation_peak = common.predict(
            model, val_stems, dataset_root, device
        )
        segmentation = common.semantic_metrics(validation_probabilities, dataset_root)
        epoch_rows: list[dict[str, Any]] = []
        for threshold in common.THRESHOLDS:
            for minimum_area in common.MIN_AREAS:
                queue_summary, _, _ = common.evaluate_probabilities(
                    validation_probabilities, dataset_root, threshold, minimum_area
                )
                feasible = float(queue_summary["candidates_per_image"]) <= common.MEAN_BURDEN_LIMIT
                record = {
                    "epoch": epoch,
                    "threshold": threshold,
                    "minimum_area_pixels": minimum_area,
                    "feasible_mean_burden": int(feasible),
                    **segmentation,
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
                                "architecture": "DeepLabV3Plus-resnet50",
                                "model_state": model.state_dict(),
                                "selection": record,
                                "epoch": epoch,
                                "seed": args.seed,
                            },
                            output / "best_validation_queue.pt",
                        )
        chosen = max(
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
            "mean_train_loss": loss_sum / max(1, batches),
            "validation_seconds": validation_seconds,
            "validation_peak_memory_bytes": validation_peak,
            "validation_mean_iou": segmentation["mean_iou"],
            "validation_plant_role_mean_iou": segmentation["plant_role_mean_iou"],
            "epoch_selected_threshold": chosen["threshold"],
            "epoch_selected_minimum_area": chosen["minimum_area_pixels"],
            "epoch_K20_one_to_one_role_qualified_recall": chosen["K20_one_to_one_role_qualified_recall"],
            "global_best_epoch_so_far": int(best_selection["epoch"] if best_selection else epoch),
        }
        history.append(history_row)
        print(json.dumps(history_row), flush=True)

    if best_selection is None:
        raise RuntimeError("No validation setting satisfied the burden contract")
    checkpoint = torch.load(output / "best_validation_queue.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(best_selection["threshold"])
    minimum_area = int(best_selection["minimum_area_pixels"])
    test_probabilities, test_seconds, test_peak = common.predict(model, test_stems, dataset_root, device)
    test_segmentation = common.semantic_metrics(test_probabilities, dataset_root)
    test_summary, test_tile_rows, combined_rows = common.evaluate_probabilities(
        test_probabilities, dataset_root, threshold, minimum_area, include_candidate_rows=True
    )
    candidate_rows = [row for row in combined_rows if "candidate_id" in row]
    date_rows = [row for row in combined_rows if "candidate_id" not in row]
    common.write_tsv(output / "training_history.tsv", history)
    common.write_tsv(output / "validation_epoch_setting_selection.tsv", selection_rows)
    common.write_tsv(output / "official_test_tile_metrics.tsv", test_tile_rows)
    common.write_tsv(output / "official_test_date_metrics.tsv", date_rows)
    common.write_tsv(output / "official_test_candidates.tsv", candidate_rows)
    governance = {
        "analysis_class": "post_test_strong_baseline_extension",
        "selection_role": "official validation only",
        "test_use": "offline scoring after all selection",
        "external_data_used": False,
        "test_labels_not_used_for": ["training", "epoch selection", "component setting selection"],
    }
    (output / "test_governance.json").write_text(json.dumps(governance, indent=2) + "\n", encoding="utf-8")
    result = {
        "run_id": args.run_id,
        "status": "completed_post_test_strong_baseline_extension",
        "git_commit": common.git_commit(root),
        "inputs": {
            "archive_sha256": common.sha256(archive),
            "config": str(config),
            "config_sha256": common.sha256(config),
            "script_sha256": common.sha256(Path(__file__).resolve()),
            "encoder_weights": str(encoder_weights),
            "encoder_weights_sha256": common.sha256(encoder_weights),
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "model": {
            "architecture": "DeepLabV3Plus-resnet50",
            "implementation": f"segmentation-models-pytorch {smp.__version__}",
            "encoder_initialization": "ImageNet-1K public ResNet50 weights",
            "checkpoint_sha256": common.sha256(output / "best_validation_queue.pt"),
        },
        "selected_on_validation": best_selection,
        "official_test_semantic_metrics": test_segmentation,
        "official_test_queue_metrics": test_summary,
        "runtime": {
            "test_total_seconds": test_seconds,
            "test_seconds_per_tile": test_seconds / len(test_stems),
            "test_peak_gpu_memory_bytes": test_peak,
            "run_wall_seconds": time.perf_counter() - started,
            "physical_gpu_id": args.physical_gpu_id,
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "governance": governance,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "SHA256SUMS.tsv").open("w", encoding="utf-8", newline="") as stream:
        stream.write("path\tsha256\n")
        for path in sorted(output.iterdir()):
            if path.name != "SHA256SUMS.tsv":
                stream.write(f"{path.name}\t{common.sha256(path)}\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
