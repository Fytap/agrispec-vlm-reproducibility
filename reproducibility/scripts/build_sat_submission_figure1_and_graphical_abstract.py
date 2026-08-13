#!/usr/bin/env python3
"""Build the SAT submission Figure 1 and graphical abstract.

The script reads only archived aggregate evidence. It does not access model
checkpoints, test images, masks, or GPUs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "manuscript_source" / "figures"
SOURCE_OUT = REPO / "reproducibility" / "v1_3_1" / "source_data"
FIG2_SOURCE = SOURCE_OUT / "Figure_2_source_data.tsv"
MERGE_SOURCE = (
    REPO
    / "reproducibility"
    / "v1_3_1"
    / "evidence"
    / "merge_aware"
    / "merge_aware_queue_summary.tsv"
)
OPERATOR_SOURCE = (
    REPO
    / "reproducibility"
    / "v1_3_1"
    / "evidence"
    / "operator_extended"
    / "operator_agreement_extended.tsv"
)

MM = 1.0 / 25.4
COLORS = {
    "ink": "#24313A",
    "muted": "#64727C",
    "grid": "#D9E0E3",
    "paper": "#FFFFFF",
    "green": "#3A7D6E",
    "green_light": "#DCECE6",
    "blue": "#3F6FA8",
    "blue_light": "#E0EAF4",
    "orange": "#CC7A3B",
    "orange_light": "#F5E7DA",
    "purple": "#735B92",
    "purple_light": "#ECE6F2",
    "red": "#B65B5B",
    "red_light": "#F3E2E2",
    "soil": "#C7AB80",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "agrispec-sat-figure1-v1",
        "savefig.facecolor": "white",
    }
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_evidence() -> dict[str, float]:
    proposal_rows = [r for r in read_tsv(FIG2_SOURCE) if r.get("panel") == "a"]
    proposal = {r["variant"]: r for r in proposal_rows}
    merge = next(r for r in read_tsv(MERGE_SOURCE) if r["group"] == "all_selected")
    operator = next(
        r
        for r in read_tsv(OPERATOR_SOURCE)
        if r["field"] == "weed_reviewable" and r["analysis"] == "all_categories_nominal"
    )
    return {
        "original_spatial": float(proposal["original"]["spatial_recall"]),
        "commissioned_spatial": float(proposal["commissioned"]["spatial_recall"]),
        "original_qualified": float(proposal["original"]["role_qualified_recall"]),
        "commissioned_qualified": float(proposal["commissioned"]["role_qualified_recall"]),
        "set_covered": float(merge["distinct_set_covered_instances"]),
        "one_to_one": float(merge["one_to_one_matched_instances"]),
        "merge_gap": float(merge["merge_capacity_gap_instances"]),
        "set_recall": float(merge["set_coverage_recall"]),
        "one_to_one_recall": float(merge["one_to_one_recall"]),
        "operator_agreement": float(operator["exact_agreement"]),
        "operator_kappa": float(operator["cohen_kappa"]),
        "operator_n": float(operator["n"]),
    }


def rounded(ax, xy, width, height, face, edge=None, lw=0.7, radius=0.025, z=2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge if edge is not None else face,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=COLORS["muted"], lw=1.0, z=4):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=z,
        )
    )


def panel_label(ax, label):
    ax.text(
        -0.025,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["ink"],
    )


def draw_field_tile(ax, x, y, w, h):
    rounded(ax, (x, y), w, h, COLORS["paper"], COLORS["ink"], lw=0.65, radius=0.018)
    ax.add_patch(Rectangle((x + 0.012, y + 0.018), w - 0.024, h - 0.036, color="#D8C6A8", zorder=2.2))
    for frac in (0.26, 0.52, 0.78):
        ax.plot(
            [x + 0.022, x + w - 0.022],
            [y + h * frac, y + h * frac],
            color="#B59468",
            lw=0.55,
            alpha=0.75,
            zorder=2.4,
        )
    for px, py, scale in ((0.27, 0.32, 1.0), (0.55, 0.69, 1.15), (0.76, 0.36, 0.9)):
        cx, cy = x + w * px, y + h * py
        ax.add_patch(Circle((cx, cy), 0.015 * scale, color="#4F8A55", zorder=3))
        ax.add_patch(Circle((cx + 0.014 * scale, cy + 0.006), 0.012 * scale, color="#79A968", zorder=3))


def pipeline_box(ax, x, title, subtitle, face, accent, icon):
    y, w, h = 0.25, 0.136, 0.49
    rounded(ax, (x, y), w, h, face, edge=accent, lw=0.8, radius=0.025)
    ax.add_patch(Circle((x + w / 2, y + 0.34), 0.055, facecolor=COLORS["paper"], edgecolor=accent, lw=0.8, zorder=3))
    if icon == "regions":
        ax.add_patch(Rectangle((x + 0.036, y + 0.305), 0.038, 0.045, fill=False, edgecolor=accent, lw=1.0, zorder=4))
        ax.add_patch(Rectangle((x + 0.067, y + 0.333), 0.043, 0.052, fill=False, edgecolor=accent, lw=1.0, zorder=4))
    elif icon == "role":
        ax.text(x + w / 2, y + 0.34, "W", ha="center", va="center", fontsize=11, fontweight="bold", color=accent, zorder=4)
    elif icon == "rank":
        for i, width in enumerate((0.065, 0.050, 0.035)):
            ax.plot([x + 0.041, x + 0.041 + width], [y + 0.372 - i * 0.032] * 2, color=accent, lw=2.0, zorder=4)
    elif icon == "queue":
        for i in range(3):
            ax.add_patch(Rectangle((x + 0.042 + i * 0.009, y + 0.305 + i * 0.013), 0.061, 0.052, facecolor=COLORS["paper"], edgecolor=accent, lw=0.75, zorder=4 + i))
    elif icon == "operator":
        ax.add_patch(Circle((x + w / 2, y + 0.36), 0.018, facecolor=accent, edgecolor="none", zorder=4))
        ax.plot([x + w / 2, x + w / 2], [y + 0.338, y + 0.305], color=accent, lw=1.3, zorder=4)
        ax.plot([x + 0.054, x + 0.091], [y + 0.325, y + 0.325], color=accent, lw=1.2, zorder=4)
    ax.text(x + w / 2, y + 0.205, title, ha="center", va="center", fontsize=7.1, fontweight="bold", color=COLORS["ink"])
    ax.text(x + w / 2, y + 0.115, subtitle, ha="center", va="center", fontsize=5.8, color=COLORS["muted"], linespacing=1.25)
    return x + w


def draw_panel_a(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel_label(ax, "a")
    ax.text(0.0, 0.98, "Finite crop--weed review queue", ha="left", va="top", fontsize=8.6, fontweight="bold", color=COLORS["ink"])
    ax.text(0.0, 0.875, "Distinct-plant support is tracked from image to operator record", ha="left", va="top", fontsize=6.3, color=COLORS["muted"])
    draw_field_tile(ax, 0.010, 0.29, 0.12, 0.39)
    ax.text(0.070, 0.205, "Target RGB", ha="center", va="center", fontsize=7.0, fontweight="bold", color=COLORS["ink"])
    ax.text(0.070, 0.135, "image-only input", ha="center", va="center", fontsize=5.8, color=COLORS["muted"])
    stages = [
        (0.168, "Candidates", "connected\nregions", COLORS["blue_light"], COLORS["blue"], "regions"),
        (0.332, "Role filter", "weed-qualified\nsupport", COLORS["green_light"], COLORS["green"], "role"),
        (0.496, "Ranker", "priority\nscore", COLORS["orange_light"], COLORS["orange"], "rank"),
        (0.660, "Top-K queue", "$K=20$\nrecords", COLORS["purple_light"], COLORS["purple"], "queue"),
        (0.824, "Operator", "retain, reject,\nor escalate", COLORS["red_light"], COLORS["red"], "operator"),
    ]
    arrow(ax, (0.135, 0.49), (0.173, 0.49))
    for i, stage in enumerate(stages):
        right = pipeline_box(ax, *stage)
        if i < len(stages) - 1:
            arrow(ax, (right + 0.004, 0.49), (stages[i + 1][0] - 0.004, 0.49))


def graph_nodes(ax, candidates, plants, edges, x0, x1, color):
    cy = [0.73 - i * 0.21 for i in range(candidates)]
    py = [0.78 - i * 0.15 for i in range(plants)]
    for a, b in edges:
        ax.plot([x0, x1], [cy[a], py[b]], color=color, lw=1.0, alpha=0.75, zorder=1)
    for i, y in enumerate(cy):
        ax.add_patch(Circle((x0, y), 0.030, facecolor=COLORS["blue_light"], edgecolor=COLORS["blue"], lw=0.8, zorder=2))
        ax.text(x0, y, f"c{i+1}", ha="center", va="center", fontsize=5.2, color=COLORS["ink"], zorder=3)
    for i, y in enumerate(py):
        ax.add_patch(Circle((x1, y), 0.030, facecolor=COLORS["green_light"], edgecolor=COLORS["green"], lw=0.8, zorder=2))
        ax.text(x1, y, f"p{i+1}", ha="center", va="center", fontsize=5.2, color=COLORS["ink"], zorder=3)


def draw_panel_b(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel_label(ax, "b")
    ax.text(0.0, 0.98, "Two geometric endpoints", ha="left", va="top", fontsize=8.6, fontweight="bold", color=COLORS["ink"])
    graph_nodes(ax, 2, 3, [(0, 0), (0, 1), (1, 2)], 0.12, 0.42, COLORS["grid"])
    ax.text(0.12, 0.835, "queued regions", ha="center", fontsize=5.8, color=COLORS["muted"])
    ax.text(0.42, 0.835, "weed instances", ha="center", fontsize=5.8, color=COLORS["muted"])
    rounded(ax, (0.54, 0.54), 0.43, 0.30, COLORS["green_light"], edge=COLORS["green"], lw=0.75, radius=0.025)
    ax.text(0.575, 0.77, "Set coverage", fontsize=7.1, fontweight="bold", color=COLORS["green"], va="center")
    ax.text(0.575, 0.68, "Incident truth union", fontsize=6.0, color=COLORS["ink"], va="center")
    ax.text(0.925, 0.68, "3", fontsize=16, fontweight="bold", color=COLORS["green"], ha="right", va="center")
    rounded(ax, (0.54, 0.16), 0.43, 0.30, COLORS["purple_light"], edge=COLORS["purple"], lw=0.75, radius=0.025)
    ax.text(0.575, 0.39, "One-to-one support", fontsize=7.1, fontweight="bold", color=COLORS["purple"], va="center")
    ax.text(0.575, 0.30, "Maximum matching", fontsize=6.0, color=COLORS["ink"], va="center")
    ax.text(0.925, 0.30, "2", fontsize=16, fontweight="bold", color=COLORS["purple"], ha="right", va="center")
    ax.text(0.50, 0.06, "A merged region can support multiple plants but creates one record", ha="center", va="center", fontsize=5.2, color=COLORS["muted"])


def result_card(ax, x, y, w, h, number, title, detail, face, accent, number_size=11.0, title_size=6.2):
    rounded(ax, (x, y), w, h, face, edge=accent, lw=0.75, radius=0.025)
    ax.text(x + 0.04 * w, y + 0.72 * h, number, ha="left", va="center", fontsize=number_size, fontweight="bold", color=accent)
    ax.text(x + 0.04 * w, y + 0.41 * h, title, ha="left", va="center", fontsize=title_size, fontweight="bold", color=COLORS["ink"], linespacing=1.05)
    ax.text(x + 0.04 * w, y + 0.14 * h, detail, ha="left", va="center", fontsize=5.3, color=COLORS["muted"])


def draw_panel_c(ax, e):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    panel_label(ax, "c")
    ax.text(0.0, 0.98, "Principal experimental findings", ha="left", va="top", fontsize=8.6, fontweight="bold", color=COLORS["ink"])
    result_card(
        ax,
        0.00,
        0.56,
        0.48,
        0.29,
        f"{e['original_spatial']:.4f}  to  {e['commissioned_spatial']:.4f}",
        "spatial support after commissioning",
        f"role-qualified: {e['original_qualified']:.4f} to {e['commissioned_qualified']:.4f}",
        COLORS["blue_light"],
        COLORS["blue"],
    )
    result_card(
        ax,
        0.52,
        0.56,
        0.48,
        0.29,
        f"{int(e['merge_gap'])} instances",
        "merge-capacity gap at K=20",
        f"set covered {int(e['set_covered'])}; one-to-one {int(e['one_to_one'])}",
        COLORS["purple_light"],
        COLORS["purple"],
    )
    result_card(
        ax,
        0.00,
        0.17,
        0.48,
        0.29,
        f"{e['operator_agreement']:.4f}",
        "operator agreement on\nweed reviewability",
        f"Cohen's kappa {e['operator_kappa']:.4f}; n={int(e['operator_n'])}",
        COLORS["green_light"],
        COLORS["green"],
        number_size=12.0,
        title_size=6.0,
    )
    rounded(ax, (0.52, 0.17), 0.48, 0.29, COLORS["orange_light"], edge=COLORS["orange"], lw=0.75, radius=0.025)
    ax.text(0.76, 0.365, "proposal + role", ha="center", va="center", fontsize=6.8, fontweight="bold", color=COLORS["blue"])
    ax.text(0.76, 0.295, "+ rank + capacity", ha="center", va="center", fontsize=6.8, fontweight="bold", color=COLORS["ink"])
    ax.text(0.76, 0.215, "jointly determine finite-queue support", ha="center", va="center", fontsize=5.3, color=COLORS["muted"])


def make_figure1(e):
    fig = plt.figure(figsize=(183 * MM, 108 * MM), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[0.92, 1.08], hspace=0.20, wspace=0.18, left=0.045, right=0.985, bottom=0.075, top=0.965)
    draw_panel_a(fig.add_subplot(gs[0, :]))
    draw_panel_b(fig.add_subplot(gs[1, 0]))
    draw_panel_c(fig.add_subplot(gs[1, 1]), e)
    return fig


def make_graphical_abstract(e):
    fig = plt.figure(figsize=(2200 / 300, 850 / 300), dpi=300)
    ax = fig.add_axes([0.025, 0.06, 0.95, 0.88])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.0, 0.97, "Bounded instance support for finite crop--weed review queues", fontsize=16, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(0.0, 0.875, "Track distinct plants through proposal formation, role qualification, ranking, and capacity", fontsize=8.3, color=COLORS["muted"], va="top")
    draw_field_tile(ax, 0.015, 0.44, 0.09, 0.26)
    ax.text(0.060, 0.385, "RGB", ha="center", fontsize=7.4, fontweight="bold", color=COLORS["ink"])
    compact = [
        (0.145, "Candidates", "regions", COLORS["blue_light"], COLORS["blue"], "regions"),
        (0.305, "Role", "weed support", COLORS["green_light"], COLORS["green"], "role"),
        (0.465, "Rank", "priority", COLORS["orange_light"], COLORS["orange"], "rank"),
        (0.625, "Queue", "$K=20$", COLORS["purple_light"], COLORS["purple"], "queue"),
        (0.785, "Operator", "one record", COLORS["red_light"], COLORS["red"], "operator"),
    ]
    arrow(ax, (0.108, 0.57), (0.142, 0.57))
    for i, s in enumerate(compact):
        x, title, subtitle, face, accent, icon = s
        rounded(ax, (x, 0.43), 0.12, 0.28, face, edge=accent, lw=0.8, radius=0.018)
        ax.text(x + 0.06, 0.62, title, ha="center", va="center", fontsize=7.4, fontweight="bold", color=COLORS["ink"])
        ax.text(x + 0.06, 0.52, subtitle, ha="center", va="center", fontsize=6.1, color=COLORS["muted"])
        if i < len(compact) - 1:
            arrow(ax, (x + 0.123, 0.57), (compact[i + 1][0] - 0.003, 0.57))
    cards = [
        (0.015, COLORS["blue_light"], COLORS["blue"], f"{e['original_spatial']:.4f} to {e['commissioned_spatial']:.4f}", "spatial support"),
        (0.345, COLORS["purple_light"], COLORS["purple"], f"{int(e['merge_gap'])} instances", "merge-capacity gap"),
        (0.675, COLORS["green_light"], COLORS["green"], f"{e['operator_agreement']:.4f}", "operator agreement"),
    ]
    for x, face, accent, number, label in cards:
        rounded(ax, (x, 0.07), 0.31, 0.22, face, edge=accent, lw=0.75, radius=0.018)
        ax.text(x + 0.155, 0.205, number, ha="center", va="center", fontsize=12, fontweight="bold", color=accent)
        ax.text(x + 0.155, 0.115, label, ha="center", va="center", fontsize=7.1, fontweight="bold", color=COLORS["ink"])
    return fig


def save_all(fig, stem: Path, *, tiff_dpi=600):
    fixed_time = datetime(2026, 8, 13, 0, 0, 0, tzinfo=timezone.utc)
    metadata = {
        "Creator": "Python/matplotlib",
        "Producer": "AgriSpec reproducibility script",
        "CreationDate": fixed_time,
        "ModDate": fixed_time,
    }
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, metadata=metadata)
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n", encoding="utf-8")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".tiff"), dpi=tiff_dpi, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})


def text_bounds_audit(fig, out_path):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    texts = []
    for text in fig.findobj(match=lambda a: hasattr(a, "get_text") and bool(a.get_text())):
        try:
            box = text.get_window_extent(renderer=renderer)
        except Exception:
            continue
        record = {"text": text.get_text(), "x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1}
        texts.append(record)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            outside.append(record)
    out_path.write_text(json.dumps({"canvas_px": [width, height], "outside_count": len(outside), "outside": outside, "texts": texts}, indent=2), encoding="utf-8")


def write_source_data(e):
    rows = [
        ("commissioning", "spatial_support", "original", e["original_spatial"], FIG2_SOURCE),
        ("commissioning", "spatial_support", "commissioned", e["commissioned_spatial"], FIG2_SOURCE),
        ("commissioning", "role_qualified_support", "original", e["original_qualified"], FIG2_SOURCE),
        ("commissioning", "role_qualified_support", "commissioned", e["commissioned_qualified"], FIG2_SOURCE),
        ("merge_aware", "distinct_set_covered_instances", "all_selected", e["set_covered"], MERGE_SOURCE),
        ("merge_aware", "one_to_one_matched_instances", "all_selected", e["one_to_one"], MERGE_SOURCE),
        ("merge_aware", "merge_capacity_gap_instances", "all_selected", e["merge_gap"], MERGE_SOURCE),
        ("operator", "exact_agreement", "weed_reviewable", e["operator_agreement"], OPERATOR_SOURCE),
        ("operator", "cohen_kappa", "weed_reviewable", e["operator_kappa"], OPERATOR_SOURCE),
        ("operator", "n", "weed_reviewable", e["operator_n"], OPERATOR_SOURCE),
    ]
    out = SOURCE_OUT / "Figure_1_source_data.tsv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["panel", "metric", "group", "value", "archived_source"])
        for panel, metric, group, value, source in rows:
            writer.writerow([panel, metric, group, f"{value:.12g}", source.relative_to(REPO).as_posix()])


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    evidence = load_evidence()
    fig1 = make_figure1(evidence)
    text_bounds_audit(fig1, SOURCE_OUT / "Figure_1_text_bounds_audit.json")
    save_all(fig1, OUT / "Figure_1")
    plt.close(fig1)
    ga = make_graphical_abstract(evidence)
    text_bounds_audit(ga, SOURCE_OUT / "Graphical_Abstract_text_bounds_audit.json")
    save_all(ga, OUT / "Graphical_Abstract", tiff_dpi=300)
    plt.close(ga)
    write_source_data(evidence)
    artifacts = [
        OUT / f"Figure_1.{ext}" for ext in ("pdf", "svg", "png", "tiff")
    ] + [OUT / f"Graphical_Abstract.{ext}" for ext in ("pdf", "svg", "png", "tiff")]
    manifest = {str(path.relative_to(REPO)).replace("\\", "/"): sha256(path) for path in artifacts}
    (SOURCE_OUT / "Figure_1_and_Graphical_Abstract_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": evidence, "artifacts": manifest}, indent=2))


if __name__ == "__main__":
    main()
