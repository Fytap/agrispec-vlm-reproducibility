#!/usr/bin/env python3
"""Extract label-free DINOv2 embeddings for RGB proposal boxes."""
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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    for name in ("project-root", "data-root", "manifest", "candidate-root", "config", "output-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--physical-gpu-id", type=int, required=True)
    p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--expected-candidates-sha256", required=True)
    p.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args(); started = time.perf_counter()
    import torch
    import timm
    from torchvision.transforms import v2
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id) or not torch.cuda.is_available():
        raise RuntimeError("GPU assignment mismatch")
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    manifest_path = args.manifest.resolve(); candidate_path = args.candidate_root.resolve() / "candidate_components.tsv"
    if sha256(manifest_path) != args.expected_manifest_sha256 or sha256(candidate_path) != args.expected_candidates_sha256:
        raise ValueError("Input hash mismatch")
    with manifest_path.open(encoding="utf-8", newline="") as h: manifest = list(csv.DictReader(h, delimiter="\t"))
    with candidate_path.open(encoding="utf-8", newline="") as h: candidates = list(csv.DictReader(h, delimiter="\t"))
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates: by_file[row["file"]].append(row)
    device = torch.device("cuda:0"); torch.manual_seed(20260810)
    # Establish the CUDA context before resetting counters. Some recent torch
    # builds reject reset_peak_memory_stats on a merely enumerated device.
    torch.cuda.get_device_properties(0)
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0, img_size=224).to(device).eval()
    torch.cuda.reset_peak_memory_stats(0)
    for p in model.parameters(): p.requires_grad_(False)
    transform = v2.Compose([v2.ToImage(), v2.Resize((224, 224), antialias=True), v2.ToDtype(torch.float32, scale=True),
                            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    ids: list[str] = []; chunks: list[np.ndarray] = []; batch: list[torch.Tensor] = []; batch_ids: list[str] = []
    def flush() -> None:
        if not batch: return
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feature = model(torch.stack(batch).to(device)).float().cpu().numpy()
        chunks.append(feature.astype(np.float32, copy=False)); ids.extend(batch_ids); batch.clear(); batch_ids.clear()
    data_root = args.data_root.resolve()
    for record in manifest:
        bands = json.loads(record["source_image_relpaths_json"])
        arrays = [np.asarray(Image.open(data_root / bands[c]), dtype=np.float32) / 65535.0 for c in ("R", "G", "B")]
        image = np.clip(np.stack(arrays, axis=-1) * 255.0, 0, 255).astype(np.uint8)
        for cand in by_file[record["sample_id"]]:
            x0, y0, x1, y1 = (int(float(cand[k])) for k in ("x0", "y0", "x1_exclusive", "y1_exclusive"))
            crop = image[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
            batch.append(transform(crop)); batch_ids.append(cand["candidate_id"])
            if len(batch) >= args.batch_size: flush()
    flush(); matrix = np.concatenate(chunks, axis=0)
    if matrix.shape[0] != len(candidates) or len(ids) != len(set(ids)): raise ValueError({"shape": matrix.shape, "ids": len(ids)})
    state_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state_hash.update(name.encode()); state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    out.mkdir(parents=True); artifact = out / "dinov2_roi_embeddings.npz"
    np.savez_compressed(artifact, candidate_ids=np.asarray(ids), embeddings=matrix)
    summary = {"run_id": args.run_id, "status": "completed_label_free_dinov2_embedding_extraction",
               "scope": {"public_data_only": True, "sealed_test_read": False, "truth_or_target_labels_read": False,
                         "encoder_weights_frozen": True},
               "model": {"timm_identifier": args.model_name, "pretraining": "DINOv2 self-supervised LVD-142M",
                         "pretrained_cfg": {k: str(v) for k, v in model.pretrained_cfg.items() if k in ("url", "hf_hub_id", "architecture", "tag")},
                         "state_dict_sha256": state_hash.hexdigest(), "input_size": [224, 224]},
               "embedding": {"candidates": len(ids), "dimension": int(matrix.shape[1]), "dtype": "float32"},
               "inputs": {"manifest_sha256": sha256(manifest_path), "candidates_sha256": sha256(candidate_path)},
               "outputs": {"embedding_npz_sha256": sha256(artifact)},
               "runtime": {"wall_seconds": time.perf_counter() - started, "python": sys.version,
                           "numpy": np.__version__, "torch": torch.__version__, "timm": timm.__version__,
                           "physical_gpu_id": args.physical_gpu_id, "gpu_name": torch.cuda.get_device_name(0),
                           "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(0)},
               "script_sha256": sha256(Path(__file__).resolve()), "config_sha256": sha256(args.config.resolve()),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
