#!/usr/bin/env python3
"""Build the SAT Revision 8 queue-allocation figure from versioned audit TSVs."""

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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.2
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.75
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.65
plt.rcParams["ytick.major.width"] = 0.65

POLICIES = [
    "fixed_K20_per_image",
    "global_top_N520",
    "bounded_global_min5_max50_N520",
    "validation_score_threshold_transferred",
]
LABELS = ["Fixed K=20", "Global top-N", "Bounded global", "Val. threshold"]
COLORS = ["#767676", "#0F4D92", "#42949E", "#E4CCD8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    audit = args.audit_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    pooled = pd.read_csv(audit / "official_test_queue_policy_comparison.tsv", sep="\t").set_index("policy").loc[POLICIES]
    tiles = pd.read_csv(audit / "official_test_queue_policy_per_tile.tsv", sep="\t")
    strata = pd.read_csv(audit / "official_test_queue_policy_date_patch.tsv", sep="\t")
    crop = pd.read_csv(audit / "official_test_crop_exposure_severity.tsv", sep="\t").set_index("policy").loc[POLICIES]

    fixed = tiles[tiles.policy == POLICIES[0]].set_index("file")
    global_n = tiles[tiles.policy == POLICIES[1]].set_index("file")
    common = fixed.index.intersection(global_n.index)
    delta = (
        global_n.loc[common, "one_to_one_role_qualified_recall"]
        - fixed.loc[common, "one_to_one_role_qualified_recall"]
    ).sort_values()

    date_rows = strata[(strata.aggregation == "acquisition_date") & strata.policy.isin(POLICIES[:2])].copy()
    dates = sorted(date_rows.stratum.unique())

    # Versioned source data for every plotted panel.
    pooled.reset_index().to_csv(output / "Figure_5a_source_data.tsv", sep="\t", index=False)
    delta.rename("global_minus_fixed_one_to_one_qualified_recall").reset_index().to_csv(
        output / "Figure_5b_source_data.tsv", sep="\t", index=False
    )
    date_rows.to_csv(output / "Figure_5c_source_data.tsv", sep="\t", index=False)
    crop.reset_index().to_csv(output / "Figure_5d_source_data.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.205, 4.724), constrained_layout=False)
    axa, axb, axc, axd = axes.flat

    # a: pooled queue metrics.
    metric_cols = [
        "all_candidate_queue_precision",
        "queued_role_qualified_weed_instance_recall",
        "one_to_one_role_qualified_recall",
    ]
    metric_labels = ["Precision", "Qualified recall\n(many-to-one)", "Qualified recall\n(one-to-one)"]
    x = np.arange(len(metric_cols))
    width = 0.19
    for i, (policy, label, color) in enumerate(zip(POLICIES, LABELS, COLORS)):
        values = pooled.loc[policy, metric_cols].astype(float).to_numpy()
        bars = axa.bar(
            x + (i - 1.5) * width,
            values,
            width=width,
            color=color,
            edgecolor="#303030",
            linewidth=0.45,
            hatch="///" if policy == POLICIES[3] else None,
            label=label,
        )
        for bar, value in zip(bars, values):
            axa.text(bar.get_x() + bar.get_width() / 2, value + 0.013, f"{value:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
    axa.set_xticks(x, metric_labels)
    axa.set_ylim(0, 0.90)
    axa.set_ylabel("Proportion")
    axa.legend(ncol=1, loc="upper right", fontsize=6, handlelength=1.2, labelspacing=0.25)
    panel_label(axa, "a")

    # b: paired tile-level differences.
    values = delta.to_numpy()
    colors = np.where(values > 1e-12, "#2E9E44", np.where(values < -1e-12, "#B64342", "#A8A8A8"))
    axb.axhline(0, color="#606060", linewidth=0.8, linestyle="--", zorder=1)
    axb.vlines(np.arange(1, len(values) + 1), 0, values, color=colors, linewidth=0.8, alpha=0.75, zorder=2)
    axb.scatter(np.arange(1, len(values) + 1), values, c=colors, s=18, edgecolor="white", linewidth=0.35, zorder=3)
    axb.set_xlim(0.2, len(values) + 0.8)
    axb.set_ylim(min(-0.15, values.min() - 0.03), max(0.43, values.max() + 0.03))
    axb.set_xlabel("Test tile (sorted by paired difference)")
    axb.set_ylabel("Global minus fixed\none-to-one qualified recall")
    improved = int((values > 1e-12).sum())
    unchanged = int((np.abs(values) <= 1e-12).sum())
    worsened = int((values < -1e-12).sum())
    axb.text(0.03, 0.96, f"Improved {improved} | unchanged {unchanged} | worsened {worsened}\nmedian 0.000; range {values.min():.3f} to {values.max():.3f}", transform=axb.transAxes, ha="left", va="top", fontsize=6.1)
    panel_label(axb, "b")

    # c: per-date heterogeneous response.
    xx = np.arange(len(dates))
    for offset, policy, label, color in [(-0.18, POLICIES[0], LABELS[0], COLORS[0]), (0.18, POLICIES[1], LABELS[1], COLORS[1])]:
        rows = date_rows[date_rows.policy == policy].set_index("stratum").loc[dates]
        axc.bar(xx + offset, rows.queued_role_qualified_weed_instance_recall, width=0.34, color=color, edgecolor="#303030", linewidth=0.45, label=label)
    fixed_counts = date_rows[date_rows.policy == POLICIES[0]].set_index("stratum").loc[dates].accepted_candidates.astype(int)
    global_counts = date_rows[date_rows.policy == POLICIES[1]].set_index("stratum").loc[dates].accepted_candidates.astype(int)
    tick_labels = [f"{d[5:]}\nN={f}/{g}" for d, f, g in zip(dates, fixed_counts, global_counts)]
    axc.set_xticks(xx, tick_labels)
    axc.set_ylim(0, 0.54)
    axc.set_ylabel("Qualified recall (many-to-one)")
    axc.set_xlabel("Acquisition date (fixed/global accepted)")
    axc.legend(loc="upper left", fontsize=6)
    panel_label(axc, "c")

    # d: crop-exposure severity rather than scene presence alone.
    scene = pooled.loc[POLICIES, "crop_overlap_scene_frequency_0.00"].astype(float).to_numpy()
    instance = crop.loc[POLICIES, "crop_instance_coverage_rate_at_0.50"].astype(float).to_numpy()
    xxx = np.arange(len(POLICIES))
    axd.bar(xxx - 0.17, scene, width=0.34, color="#E9A6A1", edgecolor="#303030", linewidth=0.45, label="Scenes with any crop overlap")
    axd.bar(xxx + 0.17, instance, width=0.34, color="#7884B4", edgecolor="#303030", linewidth=0.45, label="Crop instances covered >=50%")
    axd.set_xticks(xxx, ["Fixed", "Global", "Bounded", "Val.\nthresh."])
    axd.set_ylim(0, 0.62)
    axd.set_ylabel("Frequency / instance fraction")
    axd.legend(loc="upper right", fontsize=6)
    panel_label(axd, "d")

    for ax in axes.flat:
        ax.tick_params(labelsize=6.5, length=2.5)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.105, top=0.96, wspace=0.30, hspace=0.38)

    base = output / "Figure_5"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest_path = output / "figure_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"backend": "Python/matplotlib", "editable_svg_text": True, "figures": {}}
    manifest.setdefault("figures", {})["Figure_5"] = {
        "exports": {
            extension: sha256(output / f"Figure_5.{extension}")
            for extension in ("pdf", "png", "svg", "tiff")
        },
        "source_data": {
            panel: {
                "path": f"Figure_5{panel}_source_data.tsv",
                "sha256": sha256(output / f"Figure_5{panel}_source_data.tsv"),
            }
            for panel in ("a", "b", "c", "d")
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
