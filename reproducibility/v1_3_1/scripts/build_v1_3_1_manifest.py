#!/usr/bin/env python3
"""Build the deterministic raw-byte manifest for the v1.3.1 evidence layer."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manifests" / "v1_3_1_sha256.tsv"


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path != OUTPUT
    )
    rows = []
    for path in files:
        payload = path.read_bytes()
        rows.append({
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "path": path.relative_to(ROOT).as_posix(),
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sha256", "bytes", "path"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"{OUTPUT}: {len(rows)} files")


if __name__ == "__main__":
    main()
