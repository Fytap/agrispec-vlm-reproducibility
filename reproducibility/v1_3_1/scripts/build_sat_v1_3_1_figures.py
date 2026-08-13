#!/usr/bin/env python3
"""Build the five SAT v1.3.1 manuscript figures with Python/matplotlib."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


MM = 1 / 25.4
WIDTH = 183 * MM
INK = "#24323D"
MUTED = "#66737E"
GRID = "#D9E0E4"
BLUE = "#3B6FB6"
TEAL = "#1F9E89"
ORANGE = "#E6863B"
RED = "#C44E52"
PURPLE = "#756BB1"
YELLOW = "#E6AB02"
LIGHT_BLUE = "#DFEBF5"
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
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "agrispec-vlm-v1.3.1",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--figure",
        action="append",
        choices=[f"Figure_{index}" for index in range(1, 6)],
        help="Build only the selected figure; repeat to select multiple figures.",
    )
    return p.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def style_axis(ax: plt.Axes, grid: str = "y") -> None:
    ax.grid(axis=grid, color=GRID, linewidth=0.6, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)


def panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.12, 1.07, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.set_title(title, loc="left", fontweight="bold", pad=7)


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, face: str, title: str, detail: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.7,
            edgecolor=INK,
            facecolor=face,
        )
    )
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=7.0, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center", fontsize=6.0, color=MUTED)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = INK, dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            linestyle="--" if dashed else "-",
        )
    )


def text_bounds_audit(fig: plt.Figure, out: Path, stem: str) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside: list[str] = []
    for item in fig.findobj(match=matplotlib.text.Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        extent = item.get_window_extent(renderer)
        if extent.x0 < bounds.x0 - 2 or extent.y0 < bounds.y0 - 2 or extent.x1 > bounds.x1 + 2 or extent.y1 > bounds.y1 + 2:
            outside.append(item.get_text().replace("\n", " / "))
    result = {"figure": stem, "outside_canvas_count": len(outside), "outside_canvas_text": outside}
    (out / f"{stem}_text_bounds_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def export(fig: plt.Figure, out: Path, stem: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for ext, kwargs in [
        (
            "pdf",
            {
                "metadata": {
                    "Creator": "AgriSpec-VLM Python figure builder",
                    "Producer": "matplotlib",
                    "CreationDate": None,
                    "ModDate": None,
                }
            },
        ),
        ("svg", {"metadata": {"Creator": "AgriSpec-VLM Python figure builder", "Date": "2026-08-13"}}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        target = out / f"{stem}.{ext}"
        fig.savefig(target, **kwargs)
        if ext == "svg":
            # Matplotlib's multiline path formatting can retain trailing
            # spaces; remove them without changing the vector geometry.
            lines = target.read_text(encoding="utf-8").splitlines()
            target.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
        hashes[target.name] = sha256(target)
    return hashes


def figure1(out: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    fig = plt.figure(figsize=(WIDTH, 4.75))

    ax = fig.add_axes([0.035, 0.50, 0.93, 0.43])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.02, "a", fontsize=10, fontweight="bold")
    ax.text(0.045, 1.02, "Finite agricultural review queue", fontsize=9, fontweight="bold")
    stages = [
        (0.00, LIGHT_GREY, "RGB tile", "inference input"),
        (0.18, LIGHT_BLUE, "Candidate formation", "connected regions"),
        (0.38, LIGHT_TEAL, "Role qualification", "coverage + purity"),
        (0.58, LIGHT_ORANGE, "Ranking", "frozen score"),
        (0.76, LIGHT_RED, "Queue", "at most 20 regions"),
    ]
    for x, face, title, detail in stages:
        box(ax, x, 0.48, 0.145, 0.25, face, title, detail)
    for (x1, *_), (x2, *__) in zip(stages[:-1], stages[1:]):
        arrow(ax, (x1 + 0.147, 0.605), (x2 - 0.008, 0.605))
    box(ax, 0.91, 0.48, 0.085, 0.25, "#F8EDC7", "QA record", "retain / reject\n/ escalate")
    arrow(ax, (0.905, 0.605), (0.91, 0.605))
    for x, label, colour in [
        (0.25, "Spatial\nsupport", BLUE),
        (0.45, "Role-qualified\nsupport", TEAL),
        (0.83, "Queue\nsupport", RED),
    ]:
        arrow(ax, (x, 0.47), (x, 0.27), colour, dashed=True)
        ax.text(x, 0.13, label, ha="center", va="center", fontsize=6.5, color=colour, fontweight="bold")
    ax.text(0.0, 0.01, "Test masks enter only after queue construction and are used only for offline scoring.", fontsize=6.4, color=RED)

    ax2 = fig.add_axes([0.045, 0.08, 0.44, 0.31])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(-0.02, 1.02, "b", fontsize=10, fontweight="bold")
    ax2.text(0.07, 1.02, "Threshold-conditioned geometric endpoints", fontsize=8.5, fontweight="bold")
    box(ax2, 0.02, 0.44, 0.27, 0.30, LIGHT_RED, "One-to-one", "maximum matching")
    box(ax2, 0.36, 0.44, 0.27, 0.30, LIGHT_TEAL, "Set coverage", "truth-instance union")
    box(ax2, 0.70, 0.44, 0.27, 0.30, LIGHT_GREY, "Human recovery", "outside estimand")
    ax2.text(0.325, 0.59, r"$\leq$", ha="center", va="center", fontsize=10, color=TEAL, fontweight="bold")
    ax2.plot([0.665, 0.665], [0.39, 0.79], color=MUTED, linewidth=1.0, linestyle=(0, (3, 2)))
    ax2.text(0.665, 0.83, "not inferred", ha="center", va="bottom", fontsize=5.6, color=MUTED, fontweight="bold")
    ax2.text(0.155, 0.18, "fixed $K$ and $\\tau$", ha="center", fontsize=5.8, color=RED, fontweight="bold")
    ax2.text(0.495, 0.18, "same graph contract", ha="center", fontsize=5.8, color=TEAL, fontweight="bold")
    ax2.text(0.835, 0.18, "not bounded by geometry", ha="center", fontsize=5.8, color=MUTED, fontweight="bold")

    ax3 = fig.add_axes([0.56, 0.08, 0.40, 0.31])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.text(-0.04, 1.02, "c", fontsize=10, fontweight="bold")
    ax3.text(0.055, 1.02, "Evidence chronology", fontsize=8.5, fontweight="bold")
    timeline = [
        (0.03, LIGHT_BLUE, "Frozen\nprimary", "U-Net protocol"),
        (0.28, LIGHT_ORANGE, "Official\ntest", "one field"),
        (0.53, LIGHT_TEAL, "Post-test\naudits", "models / metrics"),
        (0.78, LIGHT_RED, "Locked\ntransfer", "CWFID once"),
    ]
    for x, face, title, detail in timeline:
        box(ax3, x, 0.42, 0.19, 0.31, face, title, detail)
    for (x1, *_), (x2, *__) in zip(timeline[:-1], timeline[1:]):
        arrow(ax3, (x1 + 0.195, 0.575), (x2 - 0.008, 0.575))
    ax3.text(0.50, 0.15, "Confirmatory scope narrows after the official test is observed.", ha="center", fontsize=6.2, color=MUTED)

    rows = [{"panel": "a", "stage": title, "detail": detail} for _, _, title, detail in stages]
    rows += [
        {"panel": "b", "endpoint": "one_to_one", "conditioning": "fixed_queue_capacity_and_matching_thresholds", "interpretation": "maximum_bipartite_matching_cardinality"},
        {"panel": "b", "endpoint": "set_coverage", "conditioning": "fixed_queue_capacity_and_matching_thresholds", "interpretation": "qualified_truth_instance_union"},
        {"panel": "b", "endpoint": "human_recovery", "conditioning": "not_measured", "interpretation": "not_bounded_by_geometric_endpoints"},
    ]
    rows += [{"panel": "c", "stage": title, "detail": detail} for _, _, title, detail in timeline]
    write_tsv(out / "Figure_1_source_data.tsv", rows)
    text_bounds_audit(fig, out, "Figure_1")
    hashes = export(fig, out, "Figure_1")
    plt.close(fig)
    return hashes, rows


def figure2(v13: Path, out: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    base = v13 / "evidence" / "commissioning_contract"
    primary = read_tsv(base / "primary_contract_ranker_schemes.tsv")
    stability = read_tsv(base / "effect_sign_stability.tsv")

    schemes = ["commissioned_only_shared", "original_only_shared", "union_shared", "reciprocal_cross_pool"]
    labels = ["Commissioned", "Original", "Union", "Reciprocal"]
    primary_map = {(r["ranker_scheme"], r["variant"]): r for r in primary}
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 5.65), gridspec_kw={"hspace": 0.50, "wspace": 0.42})

    ax = axs[0, 0]
    panel(ax, "a", "Raw proposal support before ranking")
    x = np.arange(2)
    spatial = [0.2973, 0.5713]
    qualified = [0.1928, 0.3026]
    burden = [30.85, 90.42]
    width = 0.32
    ax.bar(x - width / 2, spatial, width, color=BLUE, label="Spatial", zorder=3)
    ax.bar(x + width / 2, qualified, width, color=TEAL, label="Role-qualified", zorder=3)
    ax.set_xticks(x, ["Original", "Commissioned"])
    ax.set_ylabel("Proposal recall")
    ax.set_ylim(0, 0.66)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    for i, value in enumerate(burden):
        ax.text(i, max(spatial[i], qualified[i]) + 0.035, f"{value:.2f} candidates/tile", ha="center", fontsize=6.4, color=MUTED)

    ax = axs[0, 1]
    panel(ax, "b", "Queued effects by ranker training")
    metrics = ["candidate_precision", "spatial_recall", "set_coverage_recall", "one_to_one_recall"]
    metric_labels = ["Precision", "Spatial", "Set", "1:1"]
    matrix = np.zeros((len(schemes), len(metrics)))
    for i, scheme in enumerate(schemes):
        for j, metric in enumerate(metrics):
            matrix[i, j] = float(primary_map[(scheme, "commissioned")][metric]) - float(primary_map[(scheme, "original")][metric])
    norm = TwoSlopeNorm(vmin=-0.36, vcenter=0, vmax=0.15)
    image = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), metric_labels)
    ax.set_yticks(np.arange(len(schemes)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=6.2, color="white" if abs(matrix[i, j]) > 0.12 else INK)
    colourbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colourbar.set_label("Commissioned - original", fontsize=7)
    colourbar.set_ticks([-0.3, -0.2, -0.1, 0.0, 0.1])
    colourbar.ax.tick_params(labelsize=6)

    ax = axs[1, 0]
    panel(ax, "c", "Direction across 27 matching contracts")
    stab = {(r["ranker_scheme"], r["metric"]): r for r in stability}
    smetrics = ["spatial_recall", "set_coverage_recall", "one_to_one_recall"]
    slabels = ["Spatial", "Set", "1:1"]
    colours = [BLUE, TEAL, RED]
    count_matrix = np.asarray(
        [
            [int(stab[(scheme, metric)]["positive_contracts"]) for scheme in schemes]
            for metric in smetrics
        ]
    )
    for row, colour in enumerate(colours):
        for col, count in enumerate(count_matrix[row]):
            if count:
                alpha = min(1.0, 0.22 + 0.78 * count / 27)
                ax.scatter(
                    col,
                    row,
                    s=430,
                    marker="s",
                    color=colour,
                    alpha=alpha,
                    edgecolor=colour,
                    linewidth=0.8,
                    zorder=3,
                )
                text_colour = "white" if count >= 15 else INK
            else:
                ax.scatter(
                    col,
                    row,
                    s=430,
                    marker="s",
                    facecolor="white",
                    edgecolor=GRID,
                    linewidth=0.8,
                    zorder=3,
                )
                text_colour = MUTED
            ax.text(col, row, f"{count}/27", ha="center", va="center", fontsize=6.2, color=text_colour, zorder=4)
    ax.set_xticks(np.arange(len(schemes)), labels, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(smetrics)), slabels)
    ax.set_xlim(-0.5, len(schemes) - 0.5)
    ax.set_ylim(len(smetrics) - 0.5, -0.5)
    ax.set_ylabel("Endpoint")
    style_axis(ax, grid="both")
    ax.tick_params(length=0)

    ax = axs[1, 1]
    panel(ax, "d", "Ceiling, oracle, and achieved queue")
    scheme = "commissioned_only_shared"
    variants = ["original", "commissioned"]
    levels = ["proposal_set_coverage_ceiling", "per_tile_K20_oracle_one_to_one_recall", "one_to_one_recall"]
    level_labels = ["Set ceiling", "$K=20$ oracle", "Achieved 1:1"]
    colours = [LIGHT_BLUE, LIGHT_TEAL, LIGHT_RED]
    edges = [BLUE, TEAL, RED]
    x = np.arange(2)
    width = 0.23
    for j, (level, lab, face, edge) in enumerate(zip(levels, level_labels, colours, edges)):
        vals = [float(primary_map[(scheme, variant)][level]) for variant in variants]
        ax.bar(x + (j - 1) * width, vals, width, color=face, edgecolor=edge, linewidth=0.8, label=lab, zorder=3)
    ax.set_xticks(x, ["Original", "Commissioned"])
    ax.set_ylabel("Weed-instance recall")
    ax.set_ylim(0, 0.34)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")

    source = [{"panel": "b,d", **r} for r in primary] + [{"panel": "c", **r} for r in stability]
    source += [
        {"panel": "a", "variant": "original", "spatial_recall": spatial[0], "role_qualified_recall": qualified[0], "mean_candidates_per_tile": burden[0]},
        {"panel": "a", "variant": "commissioned", "spatial_recall": spatial[1], "role_qualified_recall": qualified[1], "mean_candidates_per_tile": burden[1]},
    ]
    write_tsv(out / "Figure_2_source_data.tsv", source)
    text_bounds_audit(fig, out, "Figure_2")
    hashes = export(fig, out, "Figure_2")
    plt.close(fig)
    return hashes, source


def figure3(repo: Path, v13: Path, out: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    per_seed = read_tsv(repo / "reproducibility" / "v1_2_0" / "evidence" / "strong_baselines" / "per_run_metrics.tsv")
    contrasts = read_tsv(v13 / "evidence" / "crossed_uncertainty" / "model_contrasts.tsv")
    standards = read_tsv(v13 / "evidence" / "model_standard_metrics" / "model_standard_instance_metrics.tsv")
    rankings = read_tsv(v13 / "evidence" / "model_standard_metrics" / "contract_ranking_stability.tsv")
    models = ["ResNet18-UNet", "DeepLabV3Plus-ResNet50", "Mask-RCNN-ResNet50-FPN-v2"]
    labels = ["U-Net", "DeepLabV3+", "Mask R-CNN"]
    colours = [BLUE, ORANGE, TEAL]
    groups: dict[str, list[float]] = defaultdict(list)
    for row in per_seed:
        if row["model"] in models:
            groups[row["model"]].append(float(row["one_to_one_role_qualified_recall"]))

    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 5.75), gridspec_kw={"hspace": 0.52, "wspace": 0.43})
    ax = axs[0, 0]
    panel(ax, "a", "Fixed-queue one-to-one endpoint")
    x = np.arange(3)
    means = [np.mean(groups[m]) for m in models]
    for i, (model, colour) in enumerate(zip(models, colours)):
        values = groups[model]
        ax.bar(i, np.mean(values), color=colour, alpha=0.82, width=0.62, zorder=2)
        ax.scatter(i + np.linspace(-0.09, 0.09, len(values)), values, s=20, facecolor="white", edgecolor=INK, linewidth=0.7, zorder=4)
    ax.set_xticks(x, labels)
    ax.set_ylabel("One-to-one recall")
    ax.set_ylim(0.18, 0.225)
    style_axis(ax)

    ax = axs[0, 1]
    panel(ax, "b", "Crossed seed-by-image uncertainty")
    selected = [r for r in contrasts if r["metric"] == "one_to_one_recall"]
    order = ["deeplabv3plus_minus_unet", "maskrcnn_minus_unet"]
    selected = sorted(selected, key=lambda r: order.index(r["contrast"]))
    y = np.array([1, 0])
    ax.axvline(0, color=INK, linewidth=0.8)
    for yi, row, colour in zip(y, selected, [ORANGE, TEAL]):
        value = float(row["point"])
        lo95 = float(row["crossed_seed_image_lower_95"])
        hi95 = float(row["crossed_seed_image_upper_95"])
        lof = float(row["bonferroni_family_lower_97_5"])
        hif = float(row["bonferroni_family_upper_97_5"])
        ax.plot([lof, hif], [yi, yi], color=colour, linewidth=1.2, alpha=0.55)
        ax.plot([lo95, hi95], [yi, yi], color=colour, linewidth=3.2)
        ax.scatter([value], [yi], s=28, facecolor="white", edgecolor=colour, linewidth=1.0, zorder=4)
    ax.set_yticks(y, ["DeepLabV3+ - U-Net", "Mask R-CNN - U-Net"])
    ax.set_xlabel("Recall difference\nthick: 95%; thin: familywise 97.5%")
    ax.set_xlim(-0.030, 0.038)
    style_axis(ax, "x")

    ax = axs[1, 0]
    panel(ax, "c", "Standard metrics on identical predictions")
    std_map = {r["model"]: r for r in standards}
    std_names = ["ResNet18--U-Net", "DeepLabV3+--ResNet50", "Mask R-CNN--ResNet50-FPN-v2"]
    metric_cols = ["mean_AP_IoU_0.50", "mean_AR100_IoU_0.50", "mean_queue_AR20_IoU_0.50", "mean_queue_PQ_style"]
    metric_labels = ["AP50", "AR100@50", "Queue AR20@50", "Queue PQ-style"]
    x = np.arange(len(metric_cols))
    width = 0.24
    for i, (name, label, colour) in enumerate(zip(std_names, labels, colours)):
        vals = [float(std_map[name][column]) for column in metric_cols]
        ax.bar(x + (i - 1) * width, vals, width, color=colour, label=label, zorder=3)
    ax.set_xticks(x, metric_labels, rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.43)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper right")

    ax = axs[1, 1]
    panel(ax, "d", "Model rank stability across contracts")
    label_to_code = {"DeepLabV3+--ResNet50": 0, "ResNet18--U-Net": 1, "Mask R-CNN--ResNet50-FPN-v2": 2}
    cmap = ListedColormap([ORANGE, BLUE, TEAL])
    purities = [0.5, 0.75, 0.9]
    pairs = [(i, l) for i in [0.25, 0.5, 0.75] for l in [0.25, 0.5, 0.75]]
    grid = np.full((len(pairs), len(purities)), np.nan)
    for row in rankings:
        iy = pairs.index((float(row["instance_coverage"]), float(row["candidate_labeled_coverage"])))
        ix = purities.index(float(row["same_role_purity"]))
        grid[iy, ix] = label_to_code[row["best_model"]]
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
    ax.set_xticks(np.arange(3), ["0.50", "0.75", "0.90"])
    ax.set_xlabel("Same-role purity")
    ax.set_yticks(np.arange(9), [f"{i:.2f} / {l:.2f}" for i, l in pairs], fontsize=5.8)
    ax.set_ylabel("Instance / labelled coverage")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(0.6)
    ax.scatter([], [], color=TEAL, label="Mask R-CNN: 21")
    ax.scatter([], [], color=BLUE, label="U-Net: 6")
    ax.scatter([], [], color=ORANGE, label="DeepLabV3+: 0")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=6.0)

    source = [{"panel": "a", **r} for r in per_seed if r["model"] in models]
    source += [{"panel": "b", **r} for r in selected]
    source += [{"panel": "c", **r} for r in standards]
    source += [{"panel": "d", **r} for r in rankings]
    write_tsv(out / "Figure_3_source_data.tsv", source)
    text_bounds_audit(fig, out, "Figure_3")
    hashes = export(fig, out, "Figure_3")
    plt.close(fig)
    return hashes, source


def figure4(repo: Path, out: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    legacy = repo / "figures" / "Figure_4.png"
    image = plt.imread(legacy)
    crops = {
        "a": (38, 223, 916, 1053, "Split: one weed, two fragments", "30 May, image 0649"),
        "b": (1499, 223, 2376, 1053, "Merge: one region, five weeds", "6 June, image 0176"),
        "c": (38, 1467, 916, 2297, "Background component", "25 May, image 0109"),
        "d": (1499, 1467, 2376, 2297, "Crop-weed mixture", "6 June, image 0770"),
    }
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 6.05), gridspec_kw={"hspace": 0.28, "wspace": 0.16})
    source: list[dict[str, object]] = []
    for ax, (label, values) in zip(axs.flat, crops.items()):
        x0, y0, x1, y1, title, subtitle = values
        panel_image = image[y0:y1, x0:x1]
        ax.imshow(panel_image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(INK)
            spine.set_linewidth(0.8)
        ax.text(-0.01, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
        ax.text(0.09, 1.08, title, transform=ax.transAxes, fontsize=8.4, fontweight="bold", va="bottom")
        ax.text(0.09, 1.015, subtitle, transform=ax.transAxes, fontsize=6.5, color=MUTED, va="bottom")
        source.append({"panel": label, "title": title, "subtitle": subtitle, "legacy_crop": f"{x0},{y0},{x1},{y1}"})
        if label == "b":
            inset = ax.inset_axes([0.52, 0.03, 0.45, 0.45])
            inset.imshow(panel_image[360:665, 250:625])
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("white")
                spine.set_linewidth(2.0)
            ax.add_patch(Rectangle((250, 360), 375, 305, fill=False, edgecolor="white", linewidth=1.2))
    fig.text(0.50, 0.015, "Cyan: candidate boundary   |   Magenta: weed truth   |   Green: crop truth   |   Yellow: background or mixed region", ha="center", fontsize=6.8)
    write_tsv(out / "Figure_4_source_data.tsv", source)
    text_bounds_audit(fig, out, "Figure_4")
    hashes = export(fig, out, "Figure_4")
    plt.close(fig)
    return hashes, source


def figure5(v13: Path, out: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    base = v13 / "evidence" / "cwfid_continuous"
    seeds = read_tsv(base / "per_seed_continuous_score_summary.tsv")
    sweep = read_tsv(base / "threshold_sweep.tsv")
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 5.5), gridspec_kw={"hspace": 0.50, "wspace": 0.40})
    x = np.arange(len(seeds))
    seed_labels = [str(i + 1) for i in range(len(seeds))]

    ax = axs[0, 0]
    panel(ax, "a", "Frozen external endpoint")
    ax.bar(x, [int(r["locked_images_with_any_candidate"]) for r in seeds], color=LIGHT_RED, edgecolor=RED, width=0.62)
    ax.set_xticks(x, seed_labels)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Images with candidates")
    ax.set_ylim(0, 5)
    ax.text(0.50, 0.62, "0 of 60 images\nfor every checkpoint", transform=ax.transAxes, ha="center", color=RED, fontweight="bold")
    ax.text(0.50, 0.34, "One-sided 95% upper bound\nfor image-level occurrence: 4.87%", transform=ax.transAxes, ha="center", fontsize=6.5, color=MUTED)
    style_axis(ax)

    ax = axs[0, 1]
    panel(ax, "b", "Post-lock continuous plant scores")
    auc = [float(r["image_macro_plant_pixel_auc"]) for r in seeds]
    ap = [float(r["image_macro_plant_pixel_ap"]) for r in seeds]
    ax.plot(x, auc, marker="o", color=BLUE, linewidth=1.5, label="Pixel AUROC")
    ax.plot(x, ap, marker="s", color=ORANGE, linewidth=1.5, label="Pixel AP")
    ax.axhline(0.5, color=GRID, linewidth=0.8, linestyle="--")
    ax.set_xticks(x, seed_labels)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Image-macro score")
    ax.set_ylim(0, 0.72)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left")

    ax = axs[1, 0]
    panel(ax, "c", "Foreground scores below frozen thresholds")
    q99 = [float(r["image_macro_foreground_probability_q99"]) for r in seeds]
    maximum = [float(r["image_macro_foreground_probability_max"]) for r in seeds]
    threshold = [float(r["locked_validation_threshold"]) for r in seeds]
    ax.scatter(x - 0.09, q99, color=BLUE, marker="o", label="Image-macro q99")
    ax.scatter(x, maximum, color=TEAL, marker="^", label="Image-macro maximum")
    ax.scatter(x + 0.09, threshold, color=RED, marker="_", s=90, linewidth=2.0, label="Frozen threshold")
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.1, 1.0])
    ax.set_xticks(x, seed_labels)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Foreground probability")
    ax.set_ylim(0.005, 1.0)
    style_axis(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=6.2)

    ax = axs[1, 1]
    panel(ax, "d", "Threshold 0.01: spatial and qualified support")
    low = [r for r in sweep if abs(float(r["diagnostic_threshold"]) - 0.01) < 1e-9]
    spatial = [float(r["spatial_recall"]) for r in low]
    qualified = [float(r["qualified_set_coverage_recall"]) for r in low]
    one = [float(r["one_to_one_recall"]) for r in low]
    width = 0.23
    ax.bar(x - width, spatial, width, color=BLUE, label="Spatial", zorder=3)
    ax.bar(x, qualified, width, color=TEAL, label="Qualified set", zorder=3)
    ax.bar(x + width, one, width, color=RED, label="Qualified 1:1", zorder=3)
    ax.set_xticks(x, seed_labels)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.08)
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=6.2)

    source = [{"panel": "a-c", **r} for r in seeds] + [{"panel": "d", **r} for r in low]
    write_tsv(out / "Figure_5_source_data.tsv", source)
    text_bounds_audit(fig, out, "Figure_5")
    hashes = export(fig, out, "Figure_5")
    plt.close(fig)
    return hashes, source


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    v13 = repo / "reproducibility" / "v1_3_1"
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest: dict[str, object] = {"backend": "Python/matplotlib", "figures": {}}
    selected = set(args.figure or [f"Figure_{index}" for index in range(1, 6)])
    for stem, builder in [
        ("Figure_1", lambda: figure1(out)),
        ("Figure_2", lambda: figure2(v13, out)),
        ("Figure_3", lambda: figure3(repo, v13, out)),
        ("Figure_4", lambda: figure4(repo, out)),
        ("Figure_5", lambda: figure5(v13, out)),
    ]:
        if stem not in selected:
            continue
        hashes, _ = builder()
        manifest["figures"][stem] = hashes
    manifest["source_data"] = {path.name: sha256(path) for path in sorted(out.glob("Figure_*_source_data.tsv"))}
    manifest["text_bounds"] = {
        path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(out.glob("Figure_*_text_bounds_audit.json"))
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
