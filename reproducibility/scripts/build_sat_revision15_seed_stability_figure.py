#!/usr/bin/env python3
"""Build the five-seed target-semantic stability figure in Python.

Figure contract
---------------
Core conclusion: independent training seeds quantify model variability while
queue allocation changes precision, one-to-one recovery, and crop exposure.
Archetype: quantitative grid with one model panel and three policy panels.
Statistics: n=5 training seeds; points are seeds; bars are mean +/- sample SD.
Exports: editable SVG/PDF plus 600-dpi TIFF and a PNG visual-QA preview.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Mandatory editable-text rules from the selected Python figure workflow.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7


POLICIES = (
    "fixed_K20_per_tile",
    "bounded_global_min5_max50",
    "global_top_equal_fixed_budget",
    "validation_global_top520_score_threshold",
)
POLICY_LABELS = ("Fixed 20/tile", "Bounded 5–50", "Global equal-N", "Val. threshold")
POLICY_COLORS = ("#7884B4", "#D07C8F", "#A8A8A8", "#55A6A6")
# Override a legacy encoding-corrupted label with an explicit Unicode en dash.
POLICY_LABELS = ("Fixed 20/tile", "Bounded 5\u201350", "Global equal-N", "Val. threshold")
SEED_JITTER = np.asarray([-0.13, -0.065, 0.0, 0.065, 0.13])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")


def metric_panel(
    ax: plt.Axes,
    policy_rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    label: str,
    percent: bool = False,
) -> None:
    panel_label(ax, label)
    for index, (policy, color) in enumerate(zip(POLICIES, POLICY_COLORS)):
        group = sorted((row for row in policy_rows if row["policy"] == policy), key=lambda row: int(row["seed"]))
        values = np.asarray([float(row[metric]) for row in group])
        if percent:
            values *= 100.0
        if len(values) != 5:
            raise ValueError(f"Expected five values for {policy}/{metric}, found {len(values)}")
        x = np.full(5, index, dtype=float) + SEED_JITTER
        ax.scatter(x, values, s=18, facecolor=color, edgecolor="white", linewidth=0.45, zorder=3)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        ax.errorbar(index, mean, yerr=sd, fmt="_", markersize=10, markeredgewidth=1.3,
                    color="#272727", elinewidth=1.0, capsize=2.4, zorder=4)
    ax.set_xticks(range(4), POLICY_LABELS, rotation=24, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.margins(x=0.08)


def model_panel(ax: plt.Axes, model_rows: list[dict[str, str]]) -> None:
    panel_label(ax, "a")
    ordered = sorted(model_rows, key=lambda row: int(row["seed"]))
    if len(ordered) != 5 or len({int(row["seed"]) for row in ordered}) != 5:
        raise ValueError("Model panel requires five unique training seeds")
    values = np.asarray([100.0 * float(row["test_semantic_mean_iou"]) for row in ordered])
    seeds = [str(row["seed"])[-2:] for row in ordered]
    x = np.arange(5)
    ax.scatter(x, values, s=25, facecolor="#484878", edgecolor="white", linewidth=0.5, zorder=3)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    ax.axhline(mean, color="#484878", linewidth=1.0, zorder=1)
    ax.fill_between([-0.45, 4.45], mean - sd, mean + sd, color="#B4C0E4", alpha=0.35, linewidth=0, zorder=0)
    ax.text(4.42, mean, f" mean {mean:.2f}%\n SD {sd:.2f}", ha="right", va="bottom", color="#484878", fontsize=6.3)
    ax.set_xticks(x, seeds)
    ax.set_xlabel("Training-seed suffix")
    ax.set_ylabel("Semantic mIoU (%)")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.set_xlim(-0.45, 4.45)


def export(fig: plt.Figure, base: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "svg": base.with_suffix(".svg"),
        "pdf": base.with_suffix(".pdf"),
        "tiff": base.with_suffix(".tiff"),
        "png": base.with_suffix(".png"),
    }
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight", pad_inches=0.03, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", pad_inches=0.03)
    return {
        extension: {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for extension, path in paths.items()
    }


def main() -> None:
    args = parse_args()
    aggregate = args.aggregate_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    model_path = aggregate / "per_seed_model.tsv"
    policy_path = aggregate / "per_seed_policy.tsv"
    summary_path = aggregate / "summary.json"
    model_rows = read_tsv(model_path)
    policy_rows = read_tsv(policy_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or int(summary.get("n_seeds", 0)) != 5:
        raise ValueError("Aggregate must contain five completed seeds")
    if sorted(int(seed) for seed in summary["seeds"]) != [20260810, 20260811, 20260812, 20260813, 20260814]:
        raise ValueError("Unexpected seed set")
    if len(policy_rows) != 20:
        raise ValueError(f"Expected 20 seed-policy rows, found {len(policy_rows)}")

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 118 / 25.4), constrained_layout=True)
    model_panel(axes[0, 0], model_rows)
    metric_panel(axes[0, 1], policy_rows, "candidate_precision", "Role-qualified precision", "b")
    metric_panel(axes[1, 0], policy_rows, "one_to_one_qualified_recall", "1:1 weed-instance recall", "c")
    metric_panel(axes[1, 1], policy_rows, "crop_overlap_scene_frequency", "Crop-exposed scenes (%)", "d", percent=True)
    fig.align_ylabels(axes[:, 0])

    source_rows: list[dict[str, Any]] = []
    for row in model_rows:
        source_rows.append({"panel": "a", "seed": row["seed"], "policy": "model", "metric": "test_semantic_mean_iou", "value": row["test_semantic_mean_iou"]})
    for panel, metric in (("b", "candidate_precision"), ("c", "one_to_one_qualified_recall"), ("d", "crop_overlap_scene_frequency")):
        for row in policy_rows:
            source_rows.append({"panel": panel, "seed": row["seed"], "policy": row["policy"], "metric": metric, "value": row[metric]})
    source_path = output / "Figure_7_source_data.tsv"
    with source_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(source_rows)

    exports = export(fig, output / "Figure_7")
    plt.close(fig)
    svg_text = (output / "Figure_7.svg").read_text(encoding="utf-8")
    if "<text" not in svg_text or "Training-seed suffix" not in svg_text:
        raise ValueError("SVG editable-text audit failed")
    manifest = {
        "figure": "Figure_7",
        "backend": "Python/matplotlib only",
        "archetype": "quantitative grid",
        "core_conclusion": "independent training seeds quantify model variability while queue allocation changes precision, one-to-one recovery, and crop exposure",
        "independent_unit": "training seed",
        "n": 5,
        "center": "arithmetic mean",
        "spread": "sample standard deviation across seeds",
        "tests": "none",
        "aggregate_summary_sha256": sha256(summary_path),
        "model_source_sha256": sha256(model_path),
        "policy_source_sha256": sha256(policy_path),
        "figure_source_data_sha256": sha256(source_path),
        "exports": exports,
        "editable_svg_text_audit": "passed",
    }
    (output / "Figure_7_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
