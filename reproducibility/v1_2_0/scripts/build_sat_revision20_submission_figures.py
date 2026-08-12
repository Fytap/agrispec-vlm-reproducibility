#!/usr/bin/env python3
"""Build submission figures for the SAT instance-support revision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


MM = 1 / 25.4
WIDTH = 183 * MM
INK = "#24323D"
MUTED = "#66737E"
GRID = "#D9E0E4"
BLUE = "#3B6FB6"
TEAL = "#2A9D8F"
ORANGE = "#E6863B"
RED = "#C44E52"
PURPLE = "#756BB1"
LIGHT_BLUE = "#DDEAF5"
LIGHT_TEAL = "#DDF1EC"
LIGHT_ORANGE = "#F8E8D7"
LIGHT_RED = "#F4DDE0"
LIGHT_GREY = "#EFF2F4"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--remote-tree", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(path)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def style_axis(ax: plt.Axes, grid: str = "y") -> None:
    ax.grid(axis=grid, color=GRID, linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)


def panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.set_title(title, loc="left", fontweight="bold", pad=8)


def export(fig: plt.Figure, out: Path, stem: str) -> dict[str, str]:
    paths = {}
    for ext, kw in [
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        p = out / f"{stem}.{ext}"
        fig.savefig(p, **kw)
        paths[p.name] = sha256(p)
    return paths


def text_bounds_audit(fig: plt.Figure, out: Path, stem: str) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        b = text.get_window_extent(renderer)
        if b.x0 < bounds.x0 - 2 or b.y0 < bounds.y0 - 2 or b.x1 > bounds.x1 + 2 or b.y1 > bounds.y1 + 2:
            outside.append(text.get_text().replace("\n", " / "))
    result = {"figure": stem, "outside_canvas_count": len(outside), "outside_canvas_text": outside}
    (out / f"{stem}_text_bounds_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def rounded(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], color: str, text: str, subtitle: str = "") -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=color, edgecolor=INK, linewidth=0.65,
        )
    )
    ax.text(x + w / 2, y + h * 0.61, text, ha="center", va="center", fontsize=7.2, fontweight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.25, subtitle, ha="center", va="center", fontsize=6.1, color=MUTED)


def arrow(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float], color: str = INK, dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=8, linewidth=0.8,
            linestyle="--" if dashed else "-", color=color,
        )
    )


def figure1(out: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    fig = plt.figure(figsize=(WIDTH, 4.85))
    ax = fig.add_axes([0.03, 0.41, 0.94, 0.53])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.01, "a", fontsize=10, fontweight="bold", va="bottom")
    ax.text(0.035, 1.01, "Finite review queue and plant-level accounting", fontsize=9, fontweight="bold", va="bottom")
    xs = [0.01, 0.19, 0.37, 0.55, 0.73]
    labels = [
        ("Target RGB", "model input", LIGHT_GREY),
        ("Candidate formation", "components", LIGHT_BLUE),
        ("Role qualification", "coverage + purity", LIGHT_TEAL),
        ("Ranking", "model score", LIGHT_ORANGE),
        ("Fixed queue", "$K=20$ regions", LIGHT_RED),
    ]
    for x, (lab, sub, col) in zip(xs, labels):
        rounded(ax, (x, 0.51), (0.145, 0.23), col, lab, sub)
    for x in xs[:-1]:
        arrow(ax, (x + 0.147, 0.625), (x + 0.178, 0.625))
    rounded(ax, (0.89, 0.51), (0.095, 0.23), "#F9E9B9", "Reviewer", "retain / reject\n/ escalate")
    arrow(ax, (0.877, 0.625), (0.89, 0.625))

    for x, lab, col in [
        (0.19, "Spatial\nsupport", BLUE),
        (0.37, "Role-qualified\nsupport", TEAL),
        (0.73, "One-to-one\nplant recovery", RED),
    ]:
        arrow(ax, (x + 0.0725, 0.50), (x + 0.0725, 0.31), color=col, dashed=True)
        rounded(ax, (x + 0.005, 0.10), (0.135, 0.18), "white", lab)
    ax.text(0.01, 0.015, "Test truth is unavailable until each queue is locked; matching is performed only for offline evaluation.", fontsize=6.6, color=RED, fontweight="bold")

    ax2 = fig.add_axes([0.04, 0.06, 0.44, 0.27])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(-0.02, 1.02, "b", fontsize=10, fontweight="bold", va="bottom")
    ax2.text(0.055, 1.02, "Declared data roles", fontsize=9, fontweight="bold", va="bottom")
    rounded(ax2, (0.00, 0.49), (0.23, 0.29), LIGHT_BLUE, "SugarBeets", "source specialists")
    rounded(ax2, (0.31, 0.49), (0.23, 0.29), LIGHT_TEAL, "WeedsGalore", "104 / 26 / 26")
    rounded(ax2, (0.67, 0.49), (0.23, 0.29), LIGHT_RED, "CWFID", "60 locked images")
    arrow(ax2, (0.24, 0.635), (0.30, 0.635), color=BLUE)
    arrow(ax2, (0.55, 0.635), (0.66, 0.635), color=RED, dashed=True)
    ax2.text(0.785, 0.32, "No external fitting\nor threshold selection", ha="center", va="center", fontsize=6.4, color=RED, fontweight="bold")
    ax2.text(0.115, 0.20, "train", ha="center", fontsize=6.4, color=MUTED)
    ax2.text(0.425, 0.20, "develop + score", ha="center", fontsize=6.4, color=MUTED)
    ax2.text(0.785, 0.13, "score once", ha="center", fontsize=6.4, color=MUTED)

    ax3 = fig.add_axes([0.56, 0.06, 0.40, 0.27])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.text(-0.04, 1.02, "c", fontsize=10, fontweight="bold", va="bottom")
    ax3.text(0.04, 1.02, "Truth-access contract", fontsize=9, fontweight="bold", va="bottom")
    cols = [0.45, 0.64, 0.81, 0.96]
    for x, t in zip(cols, ["RGB /\noutputs", "Train\ntruth", "Validation\ntruth", "Test\ntruth"]):
        ax3.text(x, 0.80, t, ha="center", va="center", fontsize=6.3, color=MUTED)
    stages = [
        ("Proposal selection", [1, 1, 0, 0]),
        ("Model selection", [1, 1, 1, 0]),
        ("Queue formation", [1, 0, 0, 0]),
        ("Offline scoring", [0, 0, 0, 1]),
    ]
    ys = [0.61, 0.44, 0.27, 0.10]
    for (name, access), y in zip(stages, ys):
        ax3.text(0.0, y, name, va="center", fontsize=6.5)
        for x, on in zip(cols, access):
            ax3.scatter([x], [y], s=30 if on else 13, color=TEAL if on else "white", edgecolor=TEAL if on else GRID, linewidth=0.7)

    rows = [
        {"panel": "a", "stage": x[0], "detail": x[1]} for x in [(l[0], l[1]) for l in labels]
    ] + [
        {"panel": "b", "stage": "SugarBeets", "detail": "source specialist training"},
        {"panel": "b", "stage": "WeedsGalore", "detail": "104 train, 26 validation, 26 test"},
        {"panel": "b", "stage": "CWFID", "detail": "60 images, locked scoring only"},
    ]
    write_tsv(out / "Figure_1_source_data.tsv", rows)
    audit = text_bounds_audit(fig, out, "Figure_1")
    hashes = export(fig, out, "Figure_1")
    plt.close(fig)
    return hashes, rows


def figure2(root: Path, out: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    base = root / "results/p2_development/P2_WEEDSGALORE_MATCHED_BURDEN_ANALYSIS_20260812_v3"
    boot = root / "results/p2_development/P2_WEEDSGALORE_MATCHED_BURDEN_UNCERTAINTY_20260812_v1"
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    variant_ci = read_tsv(boot / "variant_metric_cluster_bootstrap.tsv")
    contrast = read_tsv(boot / "paired_contrast_cluster_bootstrap.tsv")
    curve = read_tsv(base / "paired_common_k_curve.tsv")
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 5.65), gridspec_kw={"hspace": 0.48, "wspace": 0.36})

    ax = axs[0, 0]
    panel(ax, "a", "Raw proposal support and burden")
    settings = ["Original", "Commissioned"]
    x = np.arange(2)
    spatial = [0.2973, 0.5713]
    qualified = [0.1928, 0.3026]
    w = 0.32
    ax.bar(x - w / 2, spatial, w, color=BLUE, label="Spatial recall", zorder=3)
    ax.bar(x + w / 2, qualified, w, color=TEAL, label="Role-qualified recall", zorder=3)
    ax.set_xticks(x, settings)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Proposal recall")
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False)
    for i, b in enumerate([30.85, 90.42]):
        ax.text(i, max(spatial[i], qualified[i]) + 0.035, f"{b:.2f} cand./tile", ha="center", fontsize=6.6, color=MUTED)

    ax = axs[0, 1]
    panel(ax, "b", "Matched 471-candidate queues")
    metrics = ["all_candidate_precision", "many_to_one_spatial_recall", "many_to_one_role_qualified_recall", "one_to_one_role_qualified_recall"]
    labs = ["Precision", "Spatial", "Qualified", "1:1"]
    ypos = np.arange(len(metrics))[::-1]
    for variant, lab, col, shift in [
        ("original_T095_A032", "Original", BLUE, 0.10),
        ("commissioned_T070_A032", "Commissioned", ORANGE, -0.10),
    ]:
        rows = {r["metric"]: r for r in variant_ci if r["variant"] == variant}
        vals = np.array([float(rows[m]["estimate"]) for m in metrics])
        lo = np.array([float(rows[m]["ci95_low"]) for m in metrics])
        hi = np.array([float(rows[m]["ci95_high"]) for m in metrics])
        ax.errorbar(vals, ypos + shift, xerr=[vals - lo, hi - vals], fmt="o", color=col, capsize=2.5, label=lab, zorder=3)
    ax.set_yticks(ypos, labs)
    ax.set_xlim(0, 0.76)
    ax.set_xlabel("Estimate (95% image-bootstrap CI)")
    style_axis(ax, "x")
    ax.legend(loc="lower right", frameon=False)

    ax = axs[1, 0]
    panel(ax, "c", "Paired commissioning effects")
    cm = ["all_candidate_precision", "many_to_one_spatial_recall", "many_to_one_role_qualified_recall", "one_to_one_role_qualified_recall", "crop_overlap_scene_frequency"]
    clab = ["Precision", "Spatial", "Qualified", "1:1", "Crop exposure"]
    rows = {r["metric"]: r for r in contrast}
    y = np.arange(len(cm))[::-1]
    vals = np.array([float(rows[m]["estimate"]) for m in cm])
    lo = np.array([float(rows[m]["ci95_low"]) for m in cm])
    hi = np.array([float(rows[m]["ci95_high"]) for m in cm])
    ax.axvline(0, color=INK, linewidth=0.7)
    colors = [ORANGE if v < 0 else TEAL for v in vals]
    for yi, v, l, h, col in zip(y, vals, lo, hi, colors):
        ax.errorbar(v, yi, xerr=[[v - l], [h - v]], fmt="o", color=col, capsize=2.5, zorder=3)
    ax.set_yticks(y, clab)
    ax.set_xlabel("Commissioned minus original")
    ax.set_xlim(-0.16, 0.14)
    style_axis(ax, "x")

    ax = axs[1, 1]
    panel(ax, "d", "One-to-one recall across review burden")
    for variant, lab, col in [
        ("original_T095_A032", "Original", BLUE),
        ("commissioned_T070_A032", "Commissioned", ORANGE),
    ]:
        rows = [r for r in curve if r["variant"] == variant and int(r["requested_k"]) <= 20]
        xs = [float(r["mean_candidates_per_image"]) for r in rows]
        ys = [float(r["one_to_one_role_qualified_recall"]) for r in rows]
        ax.plot(xs, ys, color=col, linewidth=1.8, label=lab)
    ax.set_xlabel("Selected candidates per tile")
    ax.set_ylabel("One-to-one recall")
    ax.set_xlim(0, 20.5)
    ax.set_ylim(0, 0.13)
    style_axis(ax)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.06, "AURBC 0.0470 → 0.0578", transform=ax.transAxes, ha="right", fontsize=6.6, color=MUTED)

    source_rows = []
    for r in variant_ci:
        source_rows.append({"panel": "b", **r})
    for r in contrast:
        source_rows.append({"panel": "c", **r})
    for r in curve:
        if int(r["requested_k"]) <= 20:
            source_rows.append({"panel": "d", **r})
    write_tsv(out / "Figure_2_source_data.tsv", source_rows)
    text_bounds_audit(fig, out, "Figure_2")
    hashes = export(fig, out, "Figure_2")
    plt.close(fig)
    return hashes, source_rows


def figure3(project: Path, remote: Path, out: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    strong = remote / "results/p2_development/P2_WEEDSGALORE_STRONG_BASELINE_AGGREGATE_20260812_v1"
    ranking = remote / "results/p2_development/P2_WEEDSGALORE_SEMANTIC_RANKING_DIAGNOSTICS_20260812_v3"
    per_run = read_tsv(strong / "per_run_metrics.tsv")
    contrasts = read_tsv(strong / "paired_model_contrasts_image_cluster_bootstrap.tsv")
    selected = json.loads((ranking / "summary.json").read_text(encoding="utf-8"))
    policy_rows = read_tsv(ranking / "official_test_policy_metrics.tsv")
    selected_rows = [r for r in policy_rows if r["policy"] == selected["selected_policies"][r["seed"]]]

    model_order = ["ResNet18-UNet", "DeepLabV3Plus-ResNet50", "Mask-RCNN-ResNet50-FPN-v2", "Probability ranking"]
    model_labels = ["U-Net", "DeepLab\nV3+", "Mask\nR-CNN", "Probability\nranking"]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in per_run:
        groups[r["model"]].append(r)
    groups["Probability ranking"] = selected_rows
    colors = [BLUE, ORANGE, TEAL, PURPLE]
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 5.7), gridspec_kw={"hspace": 0.50, "wspace": 0.36})

    for ax, metric, label, letter, title, ylim in [
        (axs[0, 0], "all_candidate_precision", "Candidate precision", "a", "Precision at fixed $K=20$", (0.60, 0.84)),
        (axs[0, 1], "one_to_one_role_qualified_recall", "One-to-one recall", "b", "Distinct-plant recovery", (0.175, 0.23)),
    ]:
        panel(ax, letter, title)
        x = np.arange(len(model_order))
        means = []
        for i, (m, col) in enumerate(zip(model_order, colors)):
            vals = [float(r[metric]) for r in groups[m]]
            means.append(np.mean(vals))
            ax.bar(i, np.mean(vals), color=col, alpha=0.78, width=0.66, zorder=2)
            jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(i + jitter, vals, s=18, color=INK, facecolor="white", linewidth=0.7, zorder=4)
        ax.set_xticks(x, model_labels)
        ax.set_ylabel(label)
        ax.set_ylim(*ylim)
        style_axis(ax)

    ax = axs[1, 0]
    panel(ax, "c", "Paired one-to-one recall contrasts")
    crows = [r for r in contrasts if r["metric"] == "one_to_one_role_qualified_recall"]
    y = np.arange(2)[::-1]
    names = ["DeepLabV3+", "Mask R-CNN"]
    ax.axvline(0, color=INK, linewidth=0.7)
    for yi, r, col in zip(y, crows, [ORANGE, TEAL]):
        v, lo, hi = map(float, [r["estimate"], r["ci95_low"], r["ci95_high"]])
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color=col, capsize=3, zorder=3)
    ax.set_yticks(y, names)
    ax.set_xlim(-0.027, 0.033)
    ax.set_xlabel("One-to-one recall difference vs U-Net\n(95% image-bootstrap CI)")
    style_axis(ax, "x")

    ax = axs[1, 1]
    panel(ax, "d", "U-Net recovery varied by date")
    seed_root = project / "reproducibility_backups/H200_seed_stability_20260812_v3/results/p2_development"
    date_vals: dict[str, list[float]] = defaultdict(list)
    denoms: dict[str, tuple[int, int]] = {}
    date_source = []
    for f in sorted(seed_root.glob("P2_WEEDSGALORE_TARGET_SEMANTIC_H200_SEED_*/official_test_date_metrics.tsv")):
        seed = re.search(r"SEED_(\d+)_", str(f)).group(1)
        for r in read_tsv(f):
            value = float(r["one_to_one_role_qualified_recall"])
            date_vals[r["date"]].append(value)
            denoms[r["date"]] = (int(r["images"]), int(r["weed_instances"]))
            date_source.append({"seed": seed, **r})
    dates = sorted(date_vals)
    x = np.arange(len(dates))
    means = [np.mean(date_vals[d]) for d in dates]
    sd = [np.std(date_vals[d], ddof=1) for d in dates]
    ax.bar(x, means, color=[LIGHT_BLUE, LIGHT_TEAL, LIGHT_ORANGE, LIGHT_RED], edgecolor=[BLUE, TEAL, ORANGE, RED], linewidth=0.8, zorder=2)
    ax.errorbar(x, means, yerr=sd, fmt="none", ecolor=INK, capsize=3, linewidth=0.8, zorder=3)
    for i, d in enumerate(dates):
        vals = date_vals[d]
        ax.scatter(i + np.linspace(-0.10, 0.10, len(vals)), vals, s=14, color=INK, facecolor="white", linewidth=0.6, zorder=4)
        n, weeds = denoms[d]
        ax.text(i, 0.352, f"{n} tiles\n{weeds} weeds", ha="center", va="top", fontsize=6.0, color=MUTED)
    ax.set_xticks(x, ["25 May", "30 May", "6 Jun", "15 Jun"])
    ax.set_ylabel("One-to-one recall")
    ax.set_ylim(0, 0.36)
    style_axis(ax)

    source_rows = [{"panel": "a-b", **r} for r in per_run]
    source_rows += [{"panel": "a-b", "model": "Probability ranking", **r} for r in selected_rows]
    source_rows += [{"panel": "c", **r} for r in crows]
    source_rows += [{"panel": "d", **r} for r in date_source]
    write_tsv(out / "Figure_3_source_data.tsv", source_rows)
    text_bounds_audit(fig, out, "Figure_3")
    hashes = export(fig, out, "Figure_3")
    plt.close(fig)
    return hashes, source_rows


def figure4(project: Path, out: Path) -> dict[str, str]:
    src = project / "results/p2_development/P2_WEEDSGALORE_INSTANCE_FAILURE_ATLAS_20260812_v4"
    mapping = {
        "instance_support_failure_atlas.pdf": "Figure_4.pdf",
        "instance_support_failure_atlas.svg": "Figure_4.svg",
        "instance_support_failure_atlas.png": "Figure_4.png",
        "instance_support_failure_atlas.tiff": "Figure_4.tiff",
        "source_data.tsv": "Figure_4_source_data.tsv",
        "figure_source_data.json": "Figure_4_source_data.json",
    }
    hashes = {}
    for s, d in mapping.items():
        target = out / d
        shutil.copy2(src / s, target)
        hashes[d] = sha256(target)
    return hashes


def figure5(project: Path, remote: Path, out: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    seed_root = project / "reproducibility_backups/H200_seed_stability_20260812_v3/results/p2_development"
    internal = []
    for f in sorted(seed_root.glob("P2_WEEDSGALORE_TARGET_SEMANTIC_H200_SEED_*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        internal.append({
            "seed": s["training"]["seed"],
            "internal_one_to_one_recall": s["official_test_queue_metrics"]["K20_one_to_one_role_qualified_recall"],
        })
    ext = remote / "results/p2_external/P2_CWFID_EXTERNAL_LOCKED_ZERO_CANDIDATE_CONSOLIDATION_20260812_v1"
    es = json.loads((ext / "summary.json").read_text(encoding="utf-8"))
    per_seed = read_tsv(ext / "per_seed_metrics.tsv")
    fig, axs = plt.subplots(1, 4, figsize=(WIDTH, 2.55), gridspec_kw={"wspace": 0.48})
    fig.subplots_adjust(bottom=0.20, top=0.87)
    x = np.arange(len(internal))

    ax = axs[0]
    panel(ax, "a", "Internal recovery")
    vals = [float(r["internal_one_to_one_recall"]) for r in internal]
    ax.plot(x, vals, marker="o", color=BLUE, linewidth=1.4, markersize=4)
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.set_xlabel("Seed")
    ax.set_ylabel("1:1 recall")
    ax.set_ylim(0.18, 0.22)
    style_axis(ax)

    ax = axs[1]
    panel(ax, "b", "External candidates")
    cands = [int(r["candidates"]) for r in per_seed]
    ax.bar(x, cands, color=LIGHT_RED, edgecolor=RED)
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Candidates (60 images)")
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.50, "0 for all seeds", transform=ax.transAxes, ha="center", color=RED, fontweight="bold")
    style_axis(ax)

    ax = axs[2]
    panel(ax, "c", "External recovery")
    recalls = [float(r["one_to_one_role_qualified_recall"]) for r in per_seed]
    ax.bar(x, recalls, color=LIGHT_RED, edgecolor=RED)
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.set_xlabel("Seed")
    ax.set_ylabel("1:1 recall")
    ax.set_ylim(0, 0.1)
    ax.text(0.5, 0.55, "0 / 331 components", transform=ax.transAxes, ha="center", color=RED, fontweight="bold")
    style_axis(ax)

    ax = axs[3]
    panel(ax, "d", "External semantic IoU")
    names = ["Background", "Crop", "Weed"]
    vals = [es["semantic_iou"]["background"], es["semantic_iou"]["crop"], es["semantic_iou"]["weed"]]
    ax.bar(np.arange(3), vals, color=[MUTED, TEAL, ORANGE], width=0.66)
    ax.set_xticks(np.arange(3), ["Backgr.", "Crop", "Weed"], rotation=20)
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=6.5)
    style_axis(ax)

    rows = []
    for r, e in zip(internal, per_seed):
        rows.append({"seed": r["seed"], "internal_one_to_one_recall": r["internal_one_to_one_recall"], "external_candidates": e["candidates"], "external_one_to_one_recall": e["one_to_one_role_qualified_recall"], "external_weed_components": e["weed_components"]})
    for name, value in zip(names, vals):
        rows.append({"seed": "", "internal_one_to_one_recall": "", "external_candidates": "", "external_one_to_one_recall": "", "external_weed_components": "", "semantic_class": name, "external_iou": value})
    write_tsv(out / "Figure_5_source_data.tsv", rows)
    text_bounds_audit(fig, out, "Figure_5")
    hashes = export(fig, out, "Figure_5")
    plt.close(fig)
    return hashes, rows


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    remote = args.remote_tree.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest = {"backend": "Python/matplotlib", "figures": {}}
    hashes, _ = figure1(out)
    manifest["figures"]["Figure_1"] = hashes
    hashes, _ = figure2(project, out)
    manifest["figures"]["Figure_2"] = hashes
    hashes, _ = figure3(project, remote, out)
    manifest["figures"]["Figure_3"] = hashes
    manifest["figures"]["Figure_4"] = figure4(project, out)
    hashes, _ = figure5(project, remote, out)
    manifest["figures"]["Figure_5"] = hashes
    for p in sorted(out.glob("Figure_*_source_data*")):
        manifest.setdefault("source_data", {})[p.name] = sha256(p)
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
