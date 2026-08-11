#!/usr/bin/env python3
"""Extract truth-free ROI embeddings from the frozen semantic encoder."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
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
    for name in ("project-root", "data-root", "manifest", "semantic-checkpoint", "candidate-root", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-semantic-checkpoint-sha256", required=True)
    parser.add_argument("--expected-candidates-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torchvision.models import resnet18
    from torchvision.ops import roi_align

    root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    paths = {
        "manifest": args.manifest.resolve(),
        "semantic_checkpoint": args.semantic_checkpoint.resolve(),
        "candidates": (args.candidate_root.resolve() / "candidate_components.tsv"),
    }
    expected = {
        "manifest": args.expected_manifest_sha256,
        "semantic_checkpoint": args.expected_semantic_checkpoint_sha256,
        "candidates": args.expected_candidates_sha256,
    }
    for name, path in paths.items():
        if sha256(path) != expected[name]:
            raise ValueError(f"hash mismatch: {name}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id) or not torch.cuda.is_available():
        raise RuntimeError("GPU assignment mismatch")

    with paths["manifest"].open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    with paths["candidates"].open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle, delimiter="\t"))
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_file[row["file"]].append(row)

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            encoder = resnet18(weights=None)
            self.s = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
            self.p, self.l1, self.l2, self.l3, self.l4 = encoder.maxpool, encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            stem = self.s(image)
            return self.l4(self.l3(self.l2(self.l1(self.p(stem)))))

    device = torch.device("cuda:0")
    encoder = Encoder().to(device)
    state = torch.load(paths["semantic_checkpoint"], map_location="cpu", weights_only=False)["model_state"]
    encoder_state = {key: value for key, value in state.items() if key.startswith(("s.", "p.", "l1.", "l2.", "l3.", "l4."))}
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected or [name for name in missing if not name.startswith("p.")]:
        raise ValueError({"missing": missing, "unexpected": unexpected})
    encoder.eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    candidate_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        for row in manifest:
            bands = json.loads(row["source_image_relpaths_json"])
            arrays = [np.asarray(Image.open(data_root / bands[channel]), dtype=np.float32) / 65535.0 for channel in ("R", "G", "B")]
            rgb = np.stack(arrays, axis=-1)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            tensor = ((tensor - mean) / std)[None].to(device)
            feature_map = encoder(tensor)
            frame_candidates = by_file[row["sample_id"]]
            if not frame_candidates:
                continue
            boxes = torch.tensor(
                [[0.0, float(item["x0"]), float(item["y0"]), float(item["x1_exclusive"]), float(item["y1_exclusive"])] for item in frame_candidates],
                dtype=torch.float32,
                device=device,
            )
            spatial_scale = feature_map.shape[-1] / tensor.shape[-1]
            pooled = roi_align(feature_map, boxes, output_size=(2, 2), spatial_scale=spatial_scale, sampling_ratio=2, aligned=True)
            pooled = functional.adaptive_avg_pool2d(pooled, 1).flatten(1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32, copy=False))
            candidate_ids.extend(item["candidate_id"] for item in frame_candidates)

    matrix = np.concatenate(embeddings, axis=0)
    if matrix.shape != (len(candidates), 512) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError({"shape": matrix.shape, "candidates": len(candidates), "unique_ids": len(set(candidate_ids))})
    output_dir.mkdir(parents=True)
    artifact = output_dir / "frozen_encoder_roi_embeddings.npz"
    np.savez_compressed(artifact, candidate_ids=np.asarray(candidate_ids), embeddings=matrix)
    summary = {
        "run_id": args.run_id,
        "status": "completed_truth_free_frozen_encoder_embedding_extraction",
        "scope": {
            "public_data_only": True,
            "sealed_test_read": False,
            "truth_or_labels_read": False,
            "candidate_generation_reused_without_modification": True,
            "encoder_weights_frozen": True,
        },
        "inputs": {name + "_sha256": sha256(path) for name, path in paths.items()},
        "embedding": {"dimension": 512, "roi_output_size": [2, 2], "candidates": len(candidate_ids), "dtype": "float32"},
        "outputs": {"embedding_npz_sha256": sha256(artifact)},
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "physical_gpu_id": args.physical_gpu_id,
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
        },
        "script_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
