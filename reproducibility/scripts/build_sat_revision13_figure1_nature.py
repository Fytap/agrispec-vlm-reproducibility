#!/usr/bin/env python3
"""Build the Nature-style Figure 1 information architecture for the SAT paper.

The figure is schematic and uses only declared dataset roles and manuscript facts.
It does not read images, model predictions, or test annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text


WIDTH_MM = 183.0
HEIGHT_MM = 124.0
INCH_PER_MM = 1.0 / 25.4

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


PALETTE = {
    "ink": "#27313B",
    "muted": "#66717C",
    "line": "#7A858E",
    "faint": "#EEF1F3",
    "aqua": "#DCEEEE",
    "teal": "#4E9297",
    "lilac": "#E7E4F2",
    "violet": "#6E6A9E",
    "peach": "#F1E6D7",
    "rose": "#F2DEE2",
    "rose_dark": "#A45E69",
    "train": "#CBDDED",
    "validation": "#E8DDC8",
    "test": "#EBCFD4",
    "white": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str = PALETTE["line"],
    linewidth: float = 0.75,
    linestyle: str = "-",
    radius: float = 0.018,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = PALETTE["line"],
    linewidth: float = 0.85,
    linestyle: str = "-",
    mutation_scale: float = 8.5,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.025,
        1.01,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=PALETTE["ink"],
        ha="left",
        va="bottom",
    )


def draw_rgb_tile(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    rounded_box(ax, x, y, width, height, PALETTE["white"], edgecolor=PALETTE["ink"], linewidth=0.65)
    inner_x = x + 0.012
    inner_y = y + 0.016
    inner_w = width - 0.024
    inner_h = height - 0.032
    ax.add_patch(Rectangle((inner_x, inner_y), inner_w, inner_h, facecolor="#D8C7A8", edgecolor="none", zorder=2.2))
    for offset in (0.20, 0.50, 0.80):
        ax.plot(
            [inner_x + 0.02, inner_x + inner_w - 0.02],
            [inner_y + inner_h * offset, inner_y + inner_h * offset],
            color="#B89F7A",
            linewidth=0.5,
            alpha=0.8,
            zorder=2.4,
        )
    plant_specs = [
        (0.22, 0.28, 0.026),
        (0.48, 0.63, 0.030),
        (0.72, 0.35, 0.024),
        (0.80, 0.76, 0.021),
    ]
    for px, py, radius in plant_specs:
        cx = inner_x + inner_w * px
        cy = inner_y + inner_h * py
        ax.add_patch(Circle((cx, cy), radius, facecolor="#6B9B61", edgecolor="#3E6E45", linewidth=0.35, zorder=2.8))
        ax.add_patch(Circle((cx + radius * 0.75, cy + radius * 0.15), radius * 0.65, facecolor="#88AE72", edgecolor="#3E6E45", linewidth=0.3, zorder=2.8))


def draw_queue(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    for index, dy in enumerate((0.035, 0.017, 0.0)):
        ax.add_patch(
            FancyBboxPatch(
                (x + index * 0.006, y + dy),
                width,
                height,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor=PALETTE["white"] if index < 2 else PALETTE["rose"],
                edgecolor=PALETTE["ink"],
                linewidth=0.55,
                zorder=4 + index,
            )
        )
        ax.plot(
            [x + 0.012 + index * 0.006, x + width * 0.70 + index * 0.006],
            [y + dy + height * 0.62, y + dy + height * 0.62],
            color=PALETTE["line"],
            linewidth=0.45,
            zorder=5 + index,
        )
        ax.plot(
            [x + 0.012 + index * 0.006, x + width * 0.52 + index * 0.006],
            [y + dy + height * 0.36, y + dy + height * 0.36],
            color=PALETTE["line"],
            linewidth=0.45,
            zorder=5 + index,
        )


def draw_lock(ax: plt.Axes, x: float, y: float, scale: float = 1.0) -> None:
    ax.add_patch(
        Arc(
            (x + 0.014 * scale, y + 0.026 * scale),
            0.022 * scale,
            0.029 * scale,
            theta1=0,
            theta2=180,
            linewidth=0.75,
            color=PALETTE["rose_dark"],
            zorder=7,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.028 * scale,
            0.023 * scale,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor=PALETTE["rose"],
            edgecolor=PALETTE["rose_dark"],
            linewidth=0.65,
            zorder=7,
        )
    )


def draw_panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a")
    ax.text(0.0, 0.985, "Support-aware commissioning", fontsize=8.2, fontweight="bold", color=PALETTE["ink"], va="top")
    ax.text(0.0, 0.915, "RGB observations are converted into a finite, auditable review queue", fontsize=6.0, color=PALETTE["muted"], va="top")
    rounded_box(ax, 0.565, 0.840, 0.195, 0.075, PALETTE["aqua"], edgecolor="none", linewidth=0, radius=0.025)
    ax.text(0.6625, 0.888, "CORE", fontsize=5.15, fontweight="bold", color=PALETTE["teal"], va="center", ha="center")
    ax.text(0.6625, 0.858, "proposal · rank · fixed K", fontsize=4.8, color=PALETTE["muted"], va="center", ha="center")
    rounded_box(ax, 0.785, 0.840, 0.200, 0.075, PALETTE["lilac"], edgecolor="none", linewidth=0, radius=0.025)
    ax.text(0.885, 0.888, "SECONDARY", fontsize=5.15, fontweight="bold", color=PALETTE["violet"], va="center", ha="center")
    ax.text(0.885, 0.858, "allocation · semantic · strata", fontsize=4.8, color=PALETTE["muted"], va="center", ha="center")

    ax.text(0.0, 0.790, "PRESPECIFIED CORE PATH", fontsize=5.7, fontweight="bold", color=PALETTE["teal"], va="center")
    ax.plot([0.165, 0.985], [0.790, 0.790], color=PALETTE["teal"], linewidth=0.75, alpha=0.65)

    draw_rgb_tile(ax, 0.005, 0.345, 0.105, 0.255)
    ax.text(0.0575, 0.315, "Target RGB", fontsize=6.1, fontweight="bold", color=PALETTE["ink"], ha="center", va="top")
    ax.text(0.0575, 0.270, "image only", fontsize=5.25, color=PALETTE["muted"], ha="center", va="top")

    node_specs = [
        (0.155, "Candidate\nformation", "connected components", PALETTE["aqua"]),
        (0.355, "Role\nqualification", "coverage + purity", PALETTE["lilac"]),
        (0.555, "Candidate\nranking", "masked DINOv2", PALETTE["peach"]),
        (0.755, "Review\nallocation", "fixed K / bounded", PALETTE["rose"]),
    ]
    node_w = 0.155
    node_y = 0.345
    node_h = 0.255
    for x, title, subtitle, color in node_specs:
        rounded_box(ax, x, node_y, node_w, node_h, color, edgecolor=PALETTE["ink"], linewidth=0.65)
        ax.text(x + node_w / 2, node_y + node_h * 0.61, title, fontsize=6.5, fontweight="bold", color=PALETTE["ink"], ha="center", va="center", linespacing=1.08)
        ax.text(x + node_w / 2, node_y + node_h * 0.20, subtitle, fontsize=5.15, color=PALETTE["muted"], ha="center", va="center")

    centers = [0.0575, 0.2325, 0.4325, 0.6325, 0.8325]
    widths = [0.105, node_w, node_w, node_w, node_w]
    for index in range(len(centers) - 1):
        start = (centers[index] + widths[index] / 2 + 0.006, node_y + node_h / 2)
        end = (centers[index + 1] - widths[index + 1] / 2 - 0.008, node_y + node_h / 2)
        arrow(ax, start, end, color=PALETTE["ink"], linewidth=0.85)

    draw_queue(ax, 0.928, 0.405, 0.030, 0.092)
    arrow(ax, (0.912, node_y + node_h / 2), (0.929, node_y + node_h / 2), color=PALETTE["ink"], linewidth=0.75, mutation_scale=6)
    draw_lock(ax, 0.939, 0.523, scale=0.72)
    ax.text(0.951, 0.315, "Locked queue", fontsize=5.0, color=PALETTE["rose_dark"], ha="center", va="top", fontweight="bold")

    rounded_box(ax, 0.166, 0.660, 0.133, 0.062, PALETTE["white"], edgecolor=PALETTE["teal"], linewidth=0.6)
    ax.text(0.2325, 0.691, "Frozen source specialists", fontsize=5.35, fontweight="bold", color=PALETTE["teal"], ha="center", va="center")
    arrow(ax, (0.2325, 0.658), (0.2325, 0.605), color=PALETTE["teal"], linewidth=0.75)

    metric_specs = [
        (0.2325, "Spatial proposal recall", PALETTE["aqua"]),
        (0.4325, "Role-qualified recall", PALETTE["lilac"]),
        (0.8325, "Queued recall", PALETTE["rose"]),
    ]
    metric_node_x = [0.2325, 0.4325, 0.8325]
    for (x, label, color), link_x in zip(metric_specs, metric_node_x):
        ax.plot([link_x, link_x], [0.34, 0.225], color=PALETTE["line"], linewidth=0.6, linestyle=(0, (2, 2)))
        rounded_box(ax, x - 0.078, 0.115, 0.156, 0.090, color, edgecolor=PALETTE["line"], linewidth=0.55, radius=0.025)
        ax.text(x, 0.160, label, fontsize=5.25, fontweight="bold", color=PALETTE["ink"], ha="center", va="center")

    ax.plot([0.0, 0.985], [0.055, 0.055], color=PALETTE["line"], linewidth=0.55, linestyle=(0, (3, 3)))
    ax.text(0.0, 0.010, "Test truth is excluded from this path and enters only after the queue is locked.", fontsize=5.45, color=PALETTE["rose_dark"], fontweight="bold", va="bottom")


def draw_panel_b(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "b")
    ax.text(0.0, 0.985, "Declared data roles", fontsize=8.0, fontweight="bold", color=PALETTE["ink"], va="top")
    ax.text(0.0, 0.875, "Source supervision", fontsize=5.6, fontweight="bold", color=PALETTE["muted"], va="center")
    rounded_box(ax, 0.0, 0.690, 0.98, 0.135, PALETTE["faint"], edgecolor="none", linewidth=0)
    ax.text(0.030, 0.770, "SugarBeets2016", fontsize=6.1, fontweight="bold", color=PALETTE["ink"], va="center")
    ax.text(0.030, 0.725, "RGB + masks", fontsize=5.45, color=PALETTE["muted"], va="center")
    arrow(ax, (0.37, 0.755), (0.56, 0.755), color=PALETTE["teal"], linewidth=0.7)
    ax.text(0.60, 0.770, "Frozen RGB specialists", fontsize=5.7, fontweight="bold", color=PALETTE["teal"], va="center")
    ax.text(0.60, 0.725, "foreground + crop/weed", fontsize=5.2, color=PALETTE["muted"], va="center")

    ax.text(0.0, 0.600, "Official target spatial split", fontsize=5.6, fontweight="bold", color=PALETTE["muted"], va="center")
    total = 156.0
    segments = [
        ("TRAIN", 104, PALETTE["train"]),
        ("VALID.", 26, PALETTE["validation"]),
        ("TEST", 26, PALETTE["test"]),
    ]
    x0 = 0.0
    y0 = 0.385
    bar_h = 0.16
    usable = 0.98
    for label, count, color in segments:
        width = usable * count / total
        ax.add_patch(Rectangle((x0, y0), width, bar_h, facecolor=color, edgecolor=PALETTE["white"], linewidth=1.2))
        ax.text(x0 + width / 2, y0 + bar_h * 0.64, label, fontsize=5.4, fontweight="bold", color=PALETTE["ink"], ha="center", va="center")
        ax.text(x0 + width / 2, y0 + bar_h * 0.30, f"{count} tiles", fontsize=5.15, color=PALETTE["muted"], ha="center", va="center")
        x0 += width
    ax.add_patch(Rectangle((0.0, y0), usable, bar_h, facecolor="none", edgecolor=PALETTE["line"], linewidth=0.55))
    ax.text(0.0, 0.285, "one public field · four acquisition dates", fontsize=5.5, color=PALETTE["muted"], va="center")
    ax.plot([0.0, usable], [0.230, 0.230], color=PALETTE["faint"], linewidth=1.0)
    ax.text(0.0, 0.165, "TEST SET", fontsize=5.3, fontweight="bold", color=PALETTE["rose_dark"], va="center")
    ax.text(0.0, 0.105, "two held-out spatial patches · 26 tiles", fontsize=5.45, color=PALETTE["ink"], va="center")
    ax.text(0.0, 0.045, "1,712 raster weed instances · offline scoring only", fontsize=5.25, color=PALETTE["muted"], va="center")


def draw_panel_c(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "c")
    ax.text(0.0, 0.985, "Truth-access contract", fontsize=8.0, fontweight="bold", color=PALETTE["ink"], va="top")
    ax.text(0.985, 0.900, "filled = accessed", fontsize=5.0, color=PALETTE["muted"], ha="right", va="center")

    columns = ["RGB / model\noutputs", "Train\ntruth", "Validation\ntruth", "Test\ntruth"]
    col_x = [0.50, 0.67, 0.82, 0.95]
    for x, label in zip(col_x, columns):
        ax.text(x, 0.790, label, fontsize=5.05, color=PALETTE["muted"], ha="center", va="center", linespacing=1.05)
    ax.add_patch(Rectangle((0.890, 0.055), 0.11, 0.66, facecolor=PALETTE["rose"], edgecolor="none", alpha=0.48, zorder=0))

    rows = ["Proposal search", "Model selection", "Queue formation", "Offline scoring"]
    row_y = [0.620, 0.455, 0.290, 0.125]
    access = [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
    colors = [PALETTE["teal"], PALETTE["violet"], "#A5824E", PALETTE["rose_dark"]]
    for idx, (label, y, row) in enumerate(zip(rows, row_y, access)):
        ax.text(0.0, y, label, fontsize=5.55, color=PALETTE["ink"], ha="left", va="center")
        ax.plot([0.0, 0.995], [y - 0.080, y - 0.080], color=PALETTE["faint"], linewidth=0.8, zorder=0)
        for x, used in zip(col_x, row):
            if used:
                ax.add_patch(Circle((x, y), 0.020, facecolor=colors[idx], edgecolor=PALETTE["white"], linewidth=0.8, zorder=3))
            else:
                ax.add_patch(Circle((x, y), 0.011, facecolor=PALETTE["white"], edgecolor="#C8CDD1", linewidth=0.55, zorder=2))

    ax.text(0.0, 0.015, "Test truth enters only after the queue is locked.", fontsize=5.35, fontweight="bold", color=PALETTE["rose_dark"], va="bottom")


def audit_text_layout(fig: plt.Figure, output_dir: Path) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    records: list[tuple[Text, object, str]] = []
    for artist in fig.findobj(match=lambda item: isinstance(item, Text)):
        label = artist.get_text().strip()
        if not artist.get_visible() or not label:
            continue
        box = artist.get_window_extent(renderer=renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        records.append((artist, box, label.replace("\n", " / ")))

    collisions: list[dict[str, object]] = []
    for index, (left_artist, left_box, left_label) in enumerate(records):
        for right_artist, right_box, right_label in records[index + 1 :]:
            if left_artist.axes is not right_artist.axes:
                continue
            x_overlap = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            y_overlap = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if x_overlap > 1.5 and y_overlap > 1.5:
                collisions.append(
                    {
                        "left": left_label,
                        "right": right_label,
                        "x_overlap_px": round(float(x_overlap), 2),
                        "y_overlap_px": round(float(y_overlap), 2),
                    }
                )

    outside: list[dict[str, object]] = []
    for _, box, label in records:
        if box.x0 < figure_box.x0 - 1 or box.y0 < figure_box.y0 - 1 or box.x1 > figure_box.x1 + 1 or box.y1 > figure_box.y1 + 1:
            outside.append({"label": label, "bounds_px": [round(float(v), 2) for v in box.bounds]})

    report = {
        "figure": "Figure_1",
        "backend": "Python/matplotlib",
        "visible_text_items": len(records),
        "collision_count": len(collisions),
        "outside_canvas_count": len(outside),
        "collisions": collisions,
        "outside_canvas": outside,
    }
    path = output_dir / "Figure_1_text_layout_audit.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def write_source_tables(output_dir: Path) -> dict[str, str]:
    workflow_rows = [
        ["a", "target_rgb", "Target RGB", "inference input", "core"],
        ["a", "candidate_formation", "Candidate formation", "connected components", "core"],
        ["a", "role_qualification", "Role qualification", "coverage and purity", "core"],
        ["a", "candidate_ranking", "Candidate ranking", "masked DINOv2", "core"],
        ["a", "review_allocation", "Review allocation", "fixed K and bounded allocation", "core and secondary"],
        ["a", "target_semantic", "Target-semantic baseline", "alternative component formation", "secondary"],
    ]
    data_rows = [
        ["source", "SugarBeets2016", "specialist training", "", "RGB and masks"],
        ["target_train", "WeedsGalore", "proposal and model development", "104", "official spatial split"],
        ["target_validation", "WeedsGalore", "selection", "26", "official spatial split"],
        ["target_test", "WeedsGalore", "offline scoring", "26", "two held-out patches; 1,712 raster weed instances"],
    ]
    access_rows = [
        ["proposal_search", 1, 1, 0, 0],
        ["model_selection", 1, 1, 1, 0],
        ["queue_formation", 1, 0, 0, 0],
        ["offline_scoring", 0, 0, 0, 1],
    ]
    tables = {
        "workflow": (
            ["panel", "node_id", "label", "function", "analysis_tier"],
            workflow_rows,
        ),
        "data_roles": (
            ["role", "dataset", "use", "tiles", "notes"],
            data_rows,
        ),
        "truth_access": (
            ["stage", "rgb_or_model_outputs", "train_truth", "validation_truth", "test_truth"],
            access_rows,
        ),
    }
    hashes: dict[str, str] = {}
    for name, (header, rows) in tables.items():
        path = output_dir / f"Figure_1_{name}_source_data.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        hashes[path.name] = sha256(path)
    return hashes


def export_figure(fig: plt.Figure, output_dir: Path) -> dict[str, str]:
    base = output_dir / "Figure_1"
    exports: dict[str, str] = {}
    export_specs = [
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ("png", {"dpi": 300}),
    ]
    for extension, kwargs in export_specs:
        path = base.with_suffix("." + extension)
        fig.savefig(path, facecolor="white", edgecolor="none", **kwargs)
        exports[path.name] = sha256(path)
    return exports


def build(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    fig = plt.figure(figsize=(WIDTH_MM * INCH_PER_MM, HEIGHT_MM * INCH_PER_MM), facecolor="white")
    ax_a = fig.add_axes([0.040, 0.505, 0.920, 0.450])
    ax_b = fig.add_axes([0.040, 0.065, 0.430, 0.360])
    ax_c = fig.add_axes([0.540, 0.065, 0.420, 0.360])
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)

    report = audit_text_layout(fig, output_dir)
    source_hashes = write_source_tables(output_dir)
    export_hashes = export_figure(fig, output_dir)
    plt.close(fig)

    manifest = {
        "figure": "Figure_1",
        "backend": "Python/matplotlib",
        "archetype": "asymmetric schematic-led composite",
        "target_width_mm": WIDTH_MM,
        "target_height_mm": HEIGHT_MM,
        "editable_svg_text": True,
        "core_conclusion": (
            "The framework separates candidate formation, role qualification, ranking and review-budget allocation "
            "while restricting truth access to declared development and offline-scoring roles."
        ),
        "exports": export_hashes,
        "source_data": source_hashes,
        "text_layout_audit": {
            "path": "Figure_1_text_layout_audit.json",
            "sha256": sha256(output_dir / "Figure_1_text_layout_audit.json"),
            "collision_count": report["collision_count"],
            "outside_canvas_count": report["outside_canvas_count"],
        },
    }
    manifest_path = output_dir / "Figure_1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    build(args.output_dir.resolve())


if __name__ == "__main__":
    main()
