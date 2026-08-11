#!/usr/bin/env python3
"""Sanitize and verify the public text outputs of the revision-8 semantic run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PUBLIC_FILES = (
    "summary.json",
    "test_governance.json",
    "training_history.tsv",
    "validation_epoch_setting_selection.tsv",
    "official_test_tile_metrics.tsv",
    "official_test_date_metrics.tsv",
    "official_test_candidates.tsv",
)
PROJECT_MARKER = "/agrispec-vlm/"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        if PROJECT_MARKER in normalized:
            return normalized.split(PROJECT_MARKER, 1)[1]
        if normalized.startswith("/tmp/"):
            return "RUNTIME_CACHE/" + Path(normalized).name
    return value


def main() -> None:
    args = arguments()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    missing = [name for name in PUBLIC_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing completed-run outputs: " + ", ".join(missing))
    output.mkdir(parents=True)

    input_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    for name in PUBLIC_FILES:
        src = source / name
        dst = output / name
        input_hashes[name] = sha256(src)
        if src.suffix == ".json":
            data = sanitize(json.loads(src.read_text(encoding="utf-8")))
            dst.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            shutil.copy2(src, dst)
        output_hashes[name] = sha256(dst)

    public_text = "\n".join(
        (output / name).read_text(encoding="utf-8", errors="replace") for name in PUBLIC_FILES
    )
    forbidden = (
        "/data/",
        bytes((106, 109, 115, 45)).decode("ascii"),
        bytes((35, 104, 101, 114, 97, 35)).decode("ascii"),
    )
    found = [token for token in forbidden if token in public_text]
    if found:
        raise ValueError("Forbidden internal token remained after sanitization: " + ", ".join(found))

    manifest = {
        "status": "sanitized_completed_run_text_outputs",
        "files": list(PUBLIC_FILES),
        "private_input_sha256": input_hashes,
        "public_output_sha256": output_hashes,
        "checkpoint_excluded": True,
        "raw_log_excluded": True,
    }
    (output / "public_output_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
