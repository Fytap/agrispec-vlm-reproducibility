#!/usr/bin/env python3
"""Extract frozen DINOv2 embeddings for complete WeedsGalore RGB tiles.

These embeddings are used only to reconstruct repeated acquisition locations
within each official spatial split for cluster-aware uncertainty analysis.
No semantic or instance truth is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("project-root", "data-root", "manifest", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    import timm
    import torch
    from torchvision.transforms import v2

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id) or not torch.cuda.is_available():
        raise RuntimeError("GPU assignment mismatch")
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = args.manifest.resolve()
    if sha256(manifest_path) != args.expected_manifest_sha256:
        raise ValueError("Manifest hash mismatch")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))

    device = torch.device("cuda:0")
    torch.manual_seed(20260810)
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(0)
    transform = v2.Compose([
        v2.ToImage(), v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    sample_ids: list[str] = []
    batches: list[np.ndarray] = []
    tensors: list[torch.Tensor] = []
    data_root = args.data_root.resolve()

    def flush() -> None:
        if not tensors:
            return
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = model(torch.stack(tensors).to(device)).float().cpu().numpy()
        batches.append(values.astype(np.float32, copy=False))
        tensors.clear()

    for row in manifest:
        bands = json.loads(row["source_image_relpaths_json"])
        channels = [np.asarray(Image.open(data_root / bands[name]), dtype=np.float32) / 65535.0 for name in ("R", "G", "B")]
        tensors.append(transform(np.clip(np.stack(channels, axis=-1) * 255.0, 0, 255).astype(np.uint8)))
        sample_ids.append(row["sample_id"])
        if len(tensors) >= args.batch_size:
            flush()
    flush()
    matrix = np.concatenate(batches, axis=0)
    if matrix.shape[0] != 156 or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Scene embedding coverage mismatch")

    state_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state_hash.update(name.encode())
        state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    output.mkdir(parents=True)
    artifact = output / "dinov2_scene_embeddings.npz"
    np.savez_compressed(artifact, sample_ids=np.asarray(sample_ids), embeddings=matrix)
    summary = {
        "run_id": args.run_id,
        "status": "completed_label_free_dinov2_scene_embedding_extraction",
        "scope": {"public_data_only": True, "sealed_test_read": False, "truth_or_target_labels_read": False, "use": "within-official-split repeated-location reconstruction only", "encoder_weights_frozen": True},
        "model": {"timm_identifier": args.model_name, "pretraining": "DINOv2 self-supervised LVD-142M", "state_dict_sha256": state_hash.hexdigest(), "input_size": [224, 224], "crop": "complete 600x600 tile resized without padding"},
        "embedding": {"samples": len(sample_ids), "dimension": int(matrix.shape[1]), "dtype": "float32"},
        "inputs": {"manifest_sha256": sha256(manifest_path)},
        "outputs": {"embedding_npz_sha256": sha256(artifact)},
        "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "timm": timm.__version__, "physical_gpu_id": args.physical_gpu_id, "gpu_name": torch.cuda.get_device_name(0), "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(0)},
        "script_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
