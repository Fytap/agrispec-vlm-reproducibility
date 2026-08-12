#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for a directory tree."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
from pathlib import Path


TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".svg",
    ".tex",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


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


def manifest_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES and b"\x00" not in content:
        return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return content


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
        content = manifest_bytes(path)
        rows.append(f'"{relative}"\t"{len(content)}"\t"{hashlib.sha256(content).hexdigest()}"')
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(rows) + "\n")
    print(f"{output}: {len(files)} files")


if __name__ == "__main__":
    main()
