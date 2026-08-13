#!/usr/bin/env python3
"""Audit a public submission package and create SHA-256 manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path


TEXT_SUFFIXES = {".tex", ".bib", ".txt", ".md", ".json", ".tsv", ".csv", ".yaml", ".yml"}
FORBIDDEN_PATTERNS = {
    "credential_wording": re.compile(r"(?i)\b(password|passwd|api[_ -]?key|secret[_ -]?key)\s*[:=]"),
    "ssh_command": re.compile(r"(?i)\bssh\s+[^\n]+@"),
    "internal_jump_host": re.compile(r"(?i)jms-research|zhuo\.chen#hera"),
    "local_absolute_path": re.compile(r"(?i)\b[A-Z]:\\"),
    "private_ipv4": re.compile(r"\b(?:10\.|127\.|169\.254\.|192\.168\.)\d{1,3}(?:\.\d{1,3}){2}\b"),
}
FORBIDDEN_ARCHIVE_SUFFIXES = {".aux", ".log", ".blg", ".out", ".spl", ".env", ".pem", ".key"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scan_text(path: Path) -> list[dict]:
    hits = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for name, pattern in FORBIDDEN_PATTERNS.items():
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            hits.append({"file": str(path), "line": line, "pattern": name})
    return hits


def audit_archives(root: Path) -> list[dict]:
    reports = []
    for path in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            bad_crc = archive.testzip()
            names = archive.namelist()
            forbidden = [
                name
                for name in names
                if Path(name).suffix.lower() in FORBIDDEN_ARCHIVE_SUFFIXES
                or ".git/" in name.replace("\\", "/")
            ]
            reports.append(
                {
                    "archive": path.relative_to(root).as_posix(),
                    "entries": len(names),
                    "bad_crc_entry": bad_crc,
                    "forbidden_entries": forbidden,
                }
            )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    root = args.package.resolve()
    qa = root / "05_QA_AND_MANIFEST"
    qa.mkdir(parents=True, exist_ok=True)

    text_hits = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text_hits.extend(scan_text(path))
    archives = audit_archives(root)
    archive_failures = [r for r in archives if r["bad_crc_entry"] or r["forbidden_entries"]]
    scan = {
        "status": "PASS" if not text_hits and not archive_failures else "FAIL",
        "text_hits": text_hits,
        "archives": archives,
    }
    scan_path = qa / "PUBLIC_PACKAGE_SCAN.json"
    scan_path.write_text(json.dumps(scan, indent=2) + "\n", encoding="utf-8")

    excluded = {"MANIFEST_SHA256.tsv", "MANIFEST.json"}
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest_tsv = qa / "MANIFEST_SHA256.tsv"
    with manifest_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sha256", "bytes", "path"], delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    manifest_json = qa / "MANIFEST.json"
    manifest_json.write_text(
        json.dumps(
            {
                "package": root.name,
                "file_count_excluding_manifests": len(records),
                "public_scan_status": scan["status"],
                "files": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": scan["status"], "files": len(records), "archives": archives}, indent=2))
    if scan["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
