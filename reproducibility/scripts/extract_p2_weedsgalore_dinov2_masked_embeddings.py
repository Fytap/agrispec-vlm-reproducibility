#!/usr/bin/env python3
"""Extract DINOv2 embeddings after removing proposal-external context.

The component mask is produced by the RGB proposal specialist. Target semantic
or instance labels are never read. Pixels outside the connected component but
inside its tight bounding box are filled with the per-image RGB median.
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
    for name in ("project-root", "data-root", "manifest", "candidate-root", "config", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-candidates-sha256", required=True)
    parser.add_argument("--expected-component-maps-sha256", required=True)
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    import timm
    import torch
    from torchvision.transforms import v2

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id) or not torch.cuda.is_available():
        raise RuntimeError("GPU assignment mismatch")
    root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    manifest_path = args.manifest.resolve()
    candidate_path = args.candidate_root.resolve() / "candidate_components.tsv"
    map_path = args.candidate_root.resolve() / "candidate_component_label_maps.npz"
    checks = (
        (manifest_path, args.expected_manifest_sha256),
        (candidate_path, args.expected_candidates_sha256),
        (map_path, args.expected_component_maps_sha256),
    )
    for path, expected in checks:
        if sha256(path) != expected:
            raise ValueError(f"Input hash mismatch: {path}")

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    with candidate_path.open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle, delimiter="\t"))
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_file[row["file"]].append(row)

    component_archive = np.load(map_path)
    label_maps = component_archive["label_maps"]
    sample_ids = [str(value) for value in component_archive["sample_ids"]]
    manifest_ids = [row["sample_id"] for row in manifest]
    if sample_ids != manifest_ids or label_maps.shape[0] != len(manifest):
        raise ValueError("Component-map lineage mismatch")

    device = torch.device("cuda:0")
    torch.manual_seed(20260810)
    torch.cuda.get_device_properties(0)
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(0)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    candidate_ids: list[str] = []
    chunks: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    batch_ids: list[str] = []

    def flush() -> None:
        if not batch:
            return
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = model(torch.stack(batch).to(device)).float().cpu().numpy()
        chunks.append(features.astype(np.float32, copy=False))
        candidate_ids.extend(batch_ids)
        batch.clear()
        batch_ids.clear()

    data_root = args.data_root.resolve()
    for frame_index, record in enumerate(manifest):
        bands = json.loads(record["source_image_relpaths_json"])
        arrays = [np.asarray(Image.open(data_root / bands[channel]), dtype=np.float32) / 65535.0 for channel in ("R", "G", "B")]
        image = np.clip(np.stack(arrays, axis=-1) * 255.0, 0, 255).astype(np.uint8)
        fill = np.median(image.reshape(-1, 3), axis=0).astype(np.uint8)
        labels = label_maps[frame_index]
        for candidate in by_file[record["sample_id"]]:
            x0, y0, x1, y1 = (int(float(candidate[key])) for key in ("x0", "y0", "x1_exclusive", "y1_exclusive"))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = max(x0 + 1, x1), max(y0 + 1, y1)
            crop = image[y0:y1, x0:x1].copy()
            component_index = int(candidate["component_index"])
            mask = labels[y0:y1, x0:x1] == component_index
            crop[~mask] = fill
            batch.append(transform(crop))
            batch_ids.append(candidate["candidate_id"])
            if len(batch) >= args.batch_size:
                flush()
    flush()

    matrix = np.concatenate(chunks, axis=0)
    if matrix.shape[0] != len(candidates) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError({"shape": matrix.shape, "ids": len(candidate_ids), "candidates": len(candidates)})
    state_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state_hash.update(name.encode())
        state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())

    output_dir.mkdir(parents=True)
    artifact = output_dir / "dinov2_component_masked_embeddings.npz"
    np.savez_compressed(artifact, candidate_ids=np.asarray(candidate_ids), embeddings=matrix)
    summary = {
        "run_id": args.run_id,
        "status": "completed_label_free_dinov2_component_masked_embedding_extraction",
        "scope": {
            "public_data_only": True,
            "sealed_test_read": False,
            "truth_or_target_labels_read": False,
            "mask_source": "RGB proposal connected-component map",
            "encoder_weights_frozen": True,
        },
        "crop_contract": {
            "box": "tight proposal bounding box without padding",
            "inside": "RGB pixels inside the proposal component",
            "outside_fill": "per-image RGB median",
        },
        "model": {
            "timm_identifier": args.model_name,
            "pretraining": "DINOv2 self-supervised LVD-142M",
            "state_dict_sha256": state_hash.hexdigest(),
            "input_size": [224, 224],
        },
        "embedding": {"candidates": len(candidate_ids), "dimension": int(matrix.shape[1]), "dtype": "float32"},
        "inputs": {
            "manifest_sha256": sha256(manifest_path),
            "candidates_sha256": sha256(candidate_path),
            "component_maps_sha256": sha256(map_path),
        },
        "outputs": {"embedding_npz_sha256": sha256(artifact)},
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "timm": timm.__version__,
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
