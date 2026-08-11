#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for a directory tree."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    base = args.base.resolve(strict=True)
    output = base / args.output_name
    files = sorted(
        (path for path in base.rglob("*") if path.is_file() and path != output),
        key=lambda path: path.relative_to(base).as_posix(),
    )
    rows = ['"relative_path"\t"bytes"\t"sha256"']
    for path in files:
        relative = path.relative_to(base).as_posix()
        rows.append(f'"{relative}"\t"{path.stat().st_size}"\t"{file_sha256(path)}"')
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{output}: {len(files)} files")


if __name__ == "__main__":
    main()
