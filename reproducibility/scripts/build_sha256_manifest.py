#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for a directory tree."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob matched against POSIX relative paths; may be repeated.",
    )
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
    def included(path: Path) -> bool:
        if path == output:
            return False
        relative = path.relative_to(base).as_posix()
        if ".git" in path.relative_to(base).parts:
            return False
        return not any(fnmatch.fnmatch(relative, pattern) for pattern in args.exclude)

    files = sorted(
        (path for path in base.rglob("*") if path.is_file() and included(path)),
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
