#!/usr/bin/env python3
"""Build the three-panel SAT submission Figure 5 from archived CWFID evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text


MM = 1 / 25.4
WIDTH = 183 * MM
INK = "#24323D"
MUTED = "#66737E"
GRID = "#D9E0E4"
BLUE = "#3B6FB6"
TEAL = "#1F9E89"
ORANGE = "#E6863B"
PURPLE = "#756BB1"
RED = "#C44E52"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 7.6,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "agrispec-vlm-sat-figure5-v1",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(length=3, width=0.7)
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.13, 1.09, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
    ax.set_title(title, loc="left", fontweight="bold", pad=9)


def text_bounds_audit(fig: plt.Figure, output: Path) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    outside: list[str] = []
    for item in fig.findobj(match=lambda artist: isinstance(artist, Text)):
        if not item.get_visible() or not item.get_text().strip():
            continue
        bbox = item.get_window_extent(renderer=renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > canvas_width + 1 or bbox.y1 > canvas_height + 1:
            outside.append(item.get_text())
    report = {
        "figure": "Figure_5",
        "outside_canvas_count": len(outside),
        "outside_canvas_text": outside,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if outside:
        raise RuntimeError(f"Figure 5 contains out-of-bounds text: {outside}")
    return report


def export(fig: plt.Figure, output_dir: Path) -> dict[str, str]:
    paths = {
        "Figure_5.pdf": output_dir / "Figure_5.pdf",
        "Figure_5.svg": output_dir / "Figure_5.svg",
        "Figure_5.png": output_dir / "Figure_5.png",
        "Figure_5.tiff": output_dir / "Figure_5.tiff",
    }
    fig.savefig(paths["Figure_5.svg"], metadata={"Date": None, "Creator": "AgriSpec-VLM"})
    svg_path = paths["Figure_5.svg"]
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")
    fig.savefig(
        paths["Figure_5.pdf"],
        metadata={"CreationDate": None, "ModDate": None, "Creator": "AgriSpec-VLM"},
    )
    fig.savefig(paths["Figure_5.png"], dpi=300, metadata={"Software": "AgriSpec-VLM"})
    fig.savefig(paths["Figure_5.tiff"], dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    return {name: sha256(path) for name, path in paths.items()}


def update_manifest(output_dir: Path, hashes: dict[str, str], audit: dict[str, object]) -> None:
    manifest_path = output_dir / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["backend"] = "Python/matplotlib"
    manifest.setdefault("figures", {})["Figure_5"] = hashes
    manifest.setdefault("source_data", {})["Figure_5_source_data.tsv"] = sha256(
        output_dir / "Figure_5_source_data.tsv"
    )
    manifest.setdefault("text_bounds", {})["Figure_5"] = audit
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = repo / "reproducibility" / "v1_3_1" / "evidence" / "cwfid_continuous"
    seeds = read_tsv(evidence / "per_seed_continuous_score_summary.tsv")
    sweep = read_tsv(evidence / "threshold_sweep.tsv")
    if len(seeds) != 5:
        raise ValueError(f"Expected five checkpoints, found {len(seeds)}")
    low_by_seed = {
        int(row["seed"]): row
        for row in sweep
        if abs(float(row["diagnostic_threshold"]) - 0.01) < 1e-12
    }
    if len(low_by_seed) != 5:
        raise ValueError(f"Expected five threshold-0.01 rows, found {len(low_by_seed)}")

    checkpoint = np.arange(1, 6)
    y = np.arange(5)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(WIDTH, 2.75),
        gridspec_kw={"left": 0.065, "right": 0.985, "bottom": 0.30, "top": 0.84, "wspace": 0.42},
    )

    auc = np.asarray([float(row["image_macro_plant_pixel_auc"]) for row in seeds])
    auc_sd = np.asarray([float(row["image_sample_sd_plant_pixel_auc"]) for row in seeds])
    ap = np.asarray([float(row["image_macro_plant_pixel_ap"]) for row in seeds])
    ap_sd = np.asarray([float(row["image_sample_sd_plant_pixel_ap"]) for row in seeds])
    ax = axes[0]
    panel_title(ax, "a", "Continuous plant discrimination")
    ax.errorbar(auc, y - 0.10, xerr=auc_sd, fmt="o", ms=4.2, lw=1.0, capsize=2.2, color=BLUE, label="Pixel AUROC")
    ax.errorbar(ap, y + 0.10, xerr=ap_sd, fmt="s", ms=3.8, lw=1.0, capsize=2.2, color=ORANGE, label="Pixel AP")
    ax.axvline(0.5, color=MUTED, lw=0.75, ls="--", alpha=0.7)
    ax.set_yticks(y, [str(value) for value in checkpoint])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.73)
    ax.set_xlabel("Image-macro score")
    ax.set_ylabel("Checkpoint")
    style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.75)
    ax.grid(axis="y", visible=False)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.30),
        ncol=2,
        handlelength=1.4,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    q99 = np.asarray([float(row["image_macro_foreground_probability_q99"]) for row in seeds])
    maximum = np.asarray([float(row["image_macro_foreground_probability_max"]) for row in seeds])
    selected_threshold = np.asarray([float(row["locked_validation_threshold"]) for row in seeds])
    ax = axes[1]
    panel_title(ax, "b", "Foreground response and threshold")
    ax.scatter(q99, y - 0.14, s=22, color=BLUE, marker="o", label="q99", zorder=3)
    ax.scatter(maximum, y, s=25, color=TEAL, marker="^", label="Maximum", zorder=3)
    ax.scatter(selected_threshold, y + 0.14, s=30, color=RED, marker="|", linewidths=2.0, label="Selected threshold", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.007, 1.0)
    ax.set_xticks([0.01, 0.1, 1.0], ["0.01", "0.1", "1"])
    ax.set_yticks(y, [str(value) for value in checkpoint])
    ax.invert_yaxis()
    ax.set_xlabel("Foreground probability")
    ax.set_ylabel("Checkpoint")
    style_axis(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.75)
    ax.grid(axis="y", visible=False)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.30),
        ncol=2,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.35,
    )

    ordered_low = [low_by_seed[int(row["seed"])] for row in seeds]
    spatial = np.asarray([float(row["spatial_recall"]) for row in ordered_low])
    qualified = np.asarray([float(row["one_to_one_recall"]) for row in ordered_low])
    crop_exposure = np.asarray([float(row["crop_exposure_image_frequency"]) for row in ordered_low])
    ax = axes[2]
    panel_title(ax, "c", "Low-threshold queue outcomes")
    width = 0.23
    ax.bar(checkpoint - width, spatial, width, color=BLUE, label="Spatial recall", zorder=3)
    ax.bar(checkpoint, qualified, width, color=PURPLE, label="Qualified 1:1", zorder=3)
    ax.bar(checkpoint + width, crop_exposure, width, color=ORANGE, label="Crop exposure", zorder=3)
    ax.set_xticks(checkpoint)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Rate at threshold 0.01")
    style_axis(ax)
    positive_qualified = np.flatnonzero(qualified > 0)
    for index in positive_qualified:
        ax.text(checkpoint[index], qualified[index] + 0.025, f"{qualified[index]:.3f}", ha="center", va="bottom", fontsize=5.8, color=PURPLE)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.30),
        ncol=2,
        handlelength=1.4,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    source_rows: list[dict[str, object]] = []
    for index, (seed_row, low_row) in enumerate(zip(seeds, ordered_low), start=1):
        source_rows.append(
            {
                "checkpoint": index,
                "seed": int(seed_row["seed"]),
                "images": int(seed_row["images"]),
                "plant_pixel_auc": float(seed_row["image_macro_plant_pixel_auc"]),
                "plant_pixel_auc_image_sd": float(seed_row["image_sample_sd_plant_pixel_auc"]),
                "plant_pixel_ap": float(seed_row["image_macro_plant_pixel_ap"]),
                "plant_pixel_ap_image_sd": float(seed_row["image_sample_sd_plant_pixel_ap"]),
                "foreground_probability_q99": float(seed_row["image_macro_foreground_probability_q99"]),
                "foreground_probability_max": float(seed_row["image_macro_foreground_probability_max"]),
                "validation_selected_threshold": float(seed_row["locked_validation_threshold"]),
                "queue_threshold": float(low_row["diagnostic_threshold"]),
                "spatial_recall": float(low_row["spatial_recall"]),
                "role_qualified_one_to_one_recall": float(low_row["one_to_one_recall"]),
                "candidate_precision": float(low_row["candidate_precision"]),
                "crop_exposure_image_frequency": float(low_row["crop_exposure_image_frequency"]),
                "candidates": int(low_row["candidates"]),
            }
        )
    write_tsv(output / "Figure_5_source_data.tsv", source_rows)
    audit = text_bounds_audit(fig, output / "Figure_5_text_bounds_audit.json")
    hashes = export(fig, output)
    plt.close(fig)
    update_manifest(output, hashes, audit)
    print(json.dumps({"figure": "Figure_5", "hashes": hashes, "text_bounds": audit}, indent=2))


if __name__ == "__main__":
    main()
