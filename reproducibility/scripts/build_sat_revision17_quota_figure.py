#!/usr/bin/env python3
"""Build the quota-sensitivity and spatial-block robustness figure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text


MM = 1 / 25.4
INK = "#20272E"
TEAL = "#287C7E"
BLUE = "#376A9A"
ORANGE = "#D27935"
PALE = "#E7ECEF"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.3,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quota-dir", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def panel(ax: plt.Axes, value: str) -> None:
    ax.text(-0.12, 1.08, value, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom")


def heatmap(ax: plt.Axes, frame: pd.DataFrame, metric: str, title: str, vmin: float, vmax: float) -> None:
    minima = [0, 5, 10, 20]
    maxima = [20, 30, 50, 999]
    matrix = np.full((len(minima), len(maxima)), np.nan)
    for i, minimum in enumerate(minima):
        for j, maximum in enumerate(maxima):
            policy = f"min{minimum}_max{'unrestricted' if maximum == 999 else maximum}"
            row = frame[frame.policy == policy]
            if not row.empty:
                matrix[i, j] = float(row.iloc[0][metric])
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                color = "white" if value > (vmin + 0.62 * (vmax - vmin)) else INK
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.8, color=color)
    ax.set_xticks(range(4), ["20", "30", "50", "None"])
    ax.set_yticks(range(4), ["0", "5", "10", "20"])
    ax.set_xlabel("Maximum candidates per tile")
    ax.set_ylabel("Minimum candidates per tile")
    ax.set_title(title, loc="left", fontweight="bold")
    return image


def audit(fig: plt.Figure, output: Path) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    outside = []
    for artist in fig.findobj(lambda item: isinstance(item, Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer=renderer)
        if box.x0 < canvas.x0 - 1 or box.y0 < canvas.y0 - 1 or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1:
            outside.append(artist.get_text().replace("\n", " / "))
    payload = {"outside_canvas_count": len(outside), "outside_canvas": outside}
    (output / "Figure_6_text_layout_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if outside:
        raise RuntimeError(f"Outside-canvas text: {outside}")
    return payload


def export(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(base.with_suffix(".svg"), facecolor="white")
    fig.savefig(base.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, facecolor="white")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_seed = pd.read_csv(args.quota_dir / "test_quota_grid_per_seed.tsv", sep="\t")
    patch = pd.read_csv(args.patch_dir / "per_seed_patch_effects.tsv", sep="\t")

    quota = (
        per_seed[per_seed.policy.str.startswith("min")]
        .groupby(["policy", "minimum_quota", "maximum_quota"], as_index=False)
        .agg(
            pooled_recall=("one_to_one_role_qualified_recall", "mean"),
            worst_date_recall=("worst_date_one_to_one_recall", "mean"),
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 122 * MM))
    axa, axb, axc, axd = axes.ravel()
    heatmap(axa, quota, "pooled_recall", "Mean pooled 1:1 recall", 0.19, 0.27)
    heatmap(axb, quota, "worst_date_recall", "Mean worst-date 1:1 recall", 0.0, 0.12)
    panel(axa, "a")
    panel(axb, "b")

    fixed = per_seed[per_seed.policy == "min0_max20"].sort_values("seed")
    balanced = per_seed[per_seed.policy == "date_balanced_global"].sort_values("seed")
    seed_labels = [str(value)[-2:] for value in fixed.seed]
    x = np.arange(len(fixed))
    for metric, color, marker, label in (
        ("candidate_level_role_qualified_precision", BLUE, "o", "Candidate precision"),
        ("one_to_one_role_qualified_recall", TEAL, "s", "1:1 recall"),
    ):
        delta = balanced[metric].to_numpy() - fixed[metric].to_numpy()
        axc.plot(x, delta, marker=marker, ms=4.5, lw=1.4, color=color, label=label)
    axc.axhline(0, color=INK, lw=0.8)
    axc.set_xticks(x, seed_labels)
    axc.set_xlabel("Training-seed suffix")
    axc.set_ylabel("Date-balanced minus fixed")
    axc.set_title("Paired pooled effects", loc="left", fontweight="bold")
    axc.legend(loc="upper right", fontsize=7.0)
    axc.grid(axis="y", color=PALE, lw=0.7)
    panel(axc, "c")

    current = patch[patch.policy == "date_balanced_global"].copy()
    patch_order = ["test_capture_index_patch_proxy_0", "test_capture_index_patch_proxy_1"]
    positions = {name: index for index, name in enumerate(patch_order)}
    rng = np.random.default_rng(7)
    for patch_name in patch_order:
        rows = current[current.patch_proxy == patch_name]
        center = positions[patch_name]
        jitter = rng.uniform(-0.06, 0.06, len(rows))
        axd.scatter(center + jitter, rows.one_to_one_recall_difference_vs_fixed, s=21, color=ORANGE, edgecolor="white", linewidth=0.4, zorder=3)
        mean = rows.one_to_one_recall_difference_vs_fixed.mean()
        sd = rows.one_to_one_recall_difference_vs_fixed.std(ddof=1)
        axd.errorbar(center, mean, yerr=sd, fmt="D", ms=5, color=INK, capsize=3, lw=1.2, zorder=4)
    axd.axhline(0, color=INK, lw=0.8)
    axd.set_xticks([0, 1], ["Patch proxy 0", "Patch proxy 1"])
    axd.set_ylabel("Date-balanced minus fixed\n1:1 recall")
    axd.set_title("Spatial-block heterogeneity", loc="left", fontweight="bold")
    axd.grid(axis="y", color=PALE, lw=0.7)
    axd.text(0.02, 0.04, "points: seeds; diamond: mean ± SD", transform=axd.transAxes, fontsize=6.7)
    panel(axd, "d")

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.94, wspace=0.34, hspace=0.46)
    audit_payload = audit(fig, args.output_dir)
    export(fig, args.output_dir / "Figure_6")
    plt.close(fig)

    source = per_seed[
        per_seed.policy.isin(["min0_max20", "min5_max50", "date_balanced_global"])
    ].copy()
    source.to_csv(args.output_dir / "Figure_6_source_data.tsv", sep="\t", index=False)
    patch.to_csv(args.output_dir / "Figure_6_patch_source_data.tsv", sep="\t", index=False)
    manifest = {
        "figure": "Figure_6",
        "backend": "Python/matplotlib",
        "target_width_mm": 183,
        "target_height_mm": 122,
        "core_conclusion": "Quota choices trade pooled recovery against weakest-date coverage, and date-balanced allocation has heterogeneous spatial-block effects.",
        "inputs": {
            "quota_seed_table_sha256": sha256(args.quota_dir / "test_quota_grid_per_seed.tsv"),
            "patch_effect_table_sha256": sha256(args.patch_dir / "per_seed_patch_effects.tsv"),
        },
        "audit": audit_payload,
        "exports": {path.name: sha256(path) for path in sorted(args.output_dir.glob("Figure_6.*"))},
    }
    (args.output_dir / "Figure_6_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
