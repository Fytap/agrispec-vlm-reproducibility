#!/usr/bin/env python3
"""Render submission PDFs, create contact sheets, and record page-level QA data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_contact_sheet(images: list[Path], output: Path, columns: int = 4) -> None:
    opened = [Image.open(path).convert("RGB") for path in images]
    if not opened:
        return
    thumb_w = 360
    ratio = thumb_w / opened[0].width
    thumb_h = int(opened[0].height * ratio)
    label_h = 28
    rows = (len(opened) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (path, img) in enumerate(zip(images, opened)):
        row, col = divmod(idx, columns)
        x, y = col * thumb_w, row * (thumb_h + label_h)
        thumb = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x, y + label_h))
        draw.text((x + 6, y + 7), path.stem, fill="#24313A", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def audit_pdf(label: str, path: Path, render_root: Path, pdftoppm: Path) -> dict:
    render_dir = render_root / label
    render_dir.mkdir(parents=True, exist_ok=True)
    prefix = render_dir / "page"
    subprocess.run(
        [str(pdftoppm), "-png", "-r", "105", str(path), str(prefix)],
        check=True,
        capture_output=True,
    )
    images = sorted(render_dir.glob("page-*.png"))
    make_contact_sheet(images, render_root / f"{label}_contact_sheet.jpg")
    pages = []
    with pdfplumber.open(path) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            words = page.extract_words()
            pages.append(
                {
                    "page": index,
                    "width_pt": page.width,
                    "height_pt": page.height,
                    "text_characters": len(text),
                    "word_count": len(words),
                    "blank_page_flag": len(text.strip()) < 20 and len(page.images) == 0,
                    "image_count": len(page.images),
                }
            )
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "page_count": len(pages),
        "rendered_page_count": len(images),
        "pages": pages,
        "blank_page_count": sum(int(page["blank_page_flag"]) for page in pages),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="append", required=True, help="label=path")
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pdftoppm", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for item in args.pdf:
        label, raw_path = item.split("=", 1)
        reports.append(audit_pdf(label, Path(raw_path), args.render_root, args.pdftoppm))
    payload = {"status": "PASS" if all(r["blank_page_count"] == 0 for r in reports) else "REVIEW", "documents": reports}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "documents": [{"label": r["label"], "pages": r["page_count"], "blank_pages": r["blank_page_count"]} for r in reports]}, indent=2))


if __name__ == "__main__":
    main()
