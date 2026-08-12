#!/usr/bin/env python3
"""Build SAT Figures 1--5 with collision-safe final-size typography.

The script preserves every plotted value and source-data table from the registered
analyses. It controls information architecture, line wrapping, legend placement,
and export QA.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)

COLORS = {
    "original": "#7884B4",
    "commissioned": "#E4CCD8",
    "source": "#A8A8A8",
    "transparent": "#7884B4",
    "geometry": "#B4C0E4",
    "dino": "#E4A3B7",
    "masked": "#B64342",
    "oracle": "#2E9E44",
    "random": "#D8D8D8",
    "weed": "#5B8F4E",
    "crop": "#D69B45",
    "ambiguous": "#B5B5B5",
}

POLICIES = [
    "fixed_K20_per_image",
    "global_top_N520",
    "bounded_global_min5_max50_N520",
    "validation_score_threshold_transferred",
]
POLICY_LABELS = ["Fixed K=20", "Global top-N", "Bounded global", "Val. threshold"]
POLICY_COLORS = ["#767676", "#0F4D92", "#42949E", "#E4CCD8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--allocation-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.045,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9.5,
        ha="left",
        va="bottom",
    )


def _is_axis_tick(text: Text) -> bool:
    return text in text.axes.get_xticklabels() + text.axes.get_yticklabels() if text.axes else False


def audit_text_layout(fig: plt.Figure, figure_name: str, output: Path) -> dict[str, object]:
    """Record text collisions and out-of-canvas labels after a real renderer pass."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
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
    for i, (left_artist, left, left_label) in enumerate(records):
        for right_artist, right, right_label in records[i + 1 :]:
            # Adjacent ticks on the same axis are assessed visually but ignored here;
            # their bounding boxes can touch by sub-pixel antialiasing.
            if left_artist.axes is right_artist.axes and _is_axis_tick(left_artist) and _is_axis_tick(right_artist):
                continue
            x_overlap = min(left.x1, right.x1) - max(left.x0, right.x0)
            y_overlap = min(left.y1, right.y1) - max(left.y0, right.y0)
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
        if box.x0 < fig_box.x0 - 1 or box.y0 < fig_box.y0 - 1 or box.x1 > fig_box.x1 + 1 or box.y1 > fig_box.y1 + 1:
            outside.append({"label": label, "bounds_px": [round(v, 2) for v in box.bounds]})

    report = {
        "figure": figure_name,
        "visible_text_items": len(records),
        "collision_count": len(collisions),
        "outside_canvas_count": len(outside),
        "collisions": collisions,
        "outside_canvas": outside,
    }
    (output / f"{figure_name}_text_layout_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def export(fig: plt.Figure, base: Path) -> tuple[dict[str, str], dict[str, object]]:
    report = audit_text_layout(fig, base.name, base.parent)
    outputs: dict[str, str] = {}
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("tiff", {"dpi": 600}),
        ("png", {"dpi": 300}),
    ):
        path = base.with_suffix("." + extension)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[extension] = sha256(path)
    plt.close(fig)
    return outputs, report


def workflow_figure(output: Path) -> tuple[dict[str, str], list[dict[str, object]], dict[str, object]]:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.64, 0.205, 0.22, "Public source\nSugarBeets RGB + masks", "#E0E0F0"),
        (0.265, 0.64, 0.205, 0.22, "Frozen RGB specialists\nforeground + crop/weed", "#E0F0F0"),
        (0.515, 0.67, 0.22, 0.19, "WeedsGalore RGB\ntruth-free proposals", "#F0E0D0"),
        (0.775, 0.67, 0.21, 0.19, "Review allocation\nfixed, global, bounded,\nvalidation threshold", "#E4CCD8"),
        (0.04, 0.24, 0.235, 0.21, "Official train (104 tiles)\nselect proposals; fit rankers and\ntarget semantic baseline", "#DCEBFA"),
        (0.38, 0.24, 0.235, 0.21, "Official validation (26)\nselect C, epoch, threshold,\narea, or queue threshold", "#E5E5EF"),
        (0.70, 0.24, 0.27, 0.21, "Official test (26)\noffline score declared outputs\n1,712 raster weed instances", "#F5E1E1"),
    ]
    for x, y, width, height, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.014",
                facecolor=color,
                edgecolor="#555555",
                linewidth=0.8,
            )
        )
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=7.25)
    arrows = [
        ((0.225, 0.75), (0.265, 0.75)),
        ((0.47, 0.75), (0.515, 0.75)),
        ((0.735, 0.765), (0.775, 0.765)),
        ((0.16, 0.45), (0.57, 0.67)),
        ((0.50, 0.45), (0.625, 0.67)),
        ((0.835, 0.45), (0.875, 0.67)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                color="#555555",
                linewidth=0.8,
                connectionstyle="arc3,rad=0.0",
            )
        )
    ax.text(0.015, 0.955, "a", fontweight="bold", fontsize=9)
    ax.text(0.035, 0.105, "Supervision boundary", fontweight="bold", fontsize=7.5)
    ax.text(
        0.035,
        0.065,
        "Source masks -> specialists   |   Target-train masks -> proposal/semantic fitting   |   "
        "Candidate labels -> ranker fitting   |   Test truth -> offline scoring",
        fontsize=6.35,
    )
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.02, top=0.98)
    hashes, report = export(fig, output / "Figure_1")
    source = [
        {"stage": text.replace("\n", " "), "x": x, "y": y, "width": width, "height": height}
        for x, y, width, height, text, _ in boxes
    ]
    return hashes, source, report


def proposal_figure(root: Path, output: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame], dict[str, object]]:
    proposal = pd.read_csv(
        root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_PROPOSAL_AUDIT_20260810_v1/official_split_proposal_layers.tsv",
        sep="\t",
    )
    burden = pd.read_csv(
        root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_MATCHED_PROPOSAL_QUEUE_20260810_v1/proposal_burden_and_composition.tsv",
        sep="\t",
    )
    queue = pd.read_csv(
        root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_MATCHED_PROPOSAL_QUEUE_20260810_v1/matched_weight_queue_curves.tsv",
        sep="\t",
    )
    p = proposal[
        (proposal.official_split == "test")
        & (proposal.acquisition_date == "ALL")
        & (proposal.truth_role == "weed")
    ].copy()
    p["short"] = p.variant.map({"original_T095_A032": "Original", "training_selected_commissioned": "Commissioned"})
    b = burden[burden.official_split == "test"].copy()
    b["short"] = b.variant.map({"original_T095_A032": "Original", "commissioned_T070_A032": "Commissioned"})
    q = queue[(queue.aggregation == "pooled_test") & (queue.k == 20)].copy()
    q["short"] = q.variant.map({"original_T095_A032": "Original", "commissioned_T070_A032": "Commissioned"})

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.05, 1.55, 1.05]})
    metric_names = ["Spatial", "Role-qualified"]
    original = p[p.short == "Original"].iloc[0]
    commissioned = p[p.short == "Commissioned"].iloc[0]
    values_by_policy = [
        [original.spatial_proposal_recall, original.role_qualified_proposal_recall],
        [commissioned.spatial_proposal_recall, commissioned.role_qualified_proposal_recall],
    ]
    x = np.arange(2)
    width = 0.34
    handles = []
    for offset, values, label, color in (
        (-width / 2, values_by_policy[0], "Original", COLORS["original"]),
        (width / 2, values_by_policy[1], "Commissioned", COLORS["commissioned"]),
    ):
        bars = axes[0].bar(x + offset, values, width, color=color, edgecolor="#303030", linewidth=0.45, label=label)
        handles.append(bars)
    axes[0].set_xticks(x, metric_names)
    axes[0].set_ylim(0, 0.68)
    axes[0].set_ylabel("Official-test weed recall")
    axes[0].text(0.02, 0.965, "Burden: 30.8 -> 90.4 candidates/tile", transform=axes[0].transAxes, ha="left", va="top", fontsize=6.2, color="#555555")
    panel(axes[0], "a")

    metrics = [
        ("all_candidate_queue_precision", "Queue\nprecision"),
        ("queued_spatial_weed_instance_recall", "Spatial\nrecall"),
        ("queued_role_qualified_weed_instance_recall", "Qualified\nrecall"),
        ("crop_overlap_scene_frequency_0.00", "Any crop\noverlap"),
    ]
    positions = np.arange(len(metrics))
    bw = 0.36
    for offset, label, color in ((-bw / 2, "Original", COLORS["original"]), (bw / 2, "Commissioned", COLORS["commissioned"])):
        row = q[q.short == label].iloc[0]
        axes[1].bar(positions + offset, [row[key] for key, _ in metrics], bw, color=color, edgecolor="#303030", linewidth=0.45)
    axes[1].set_xticks(positions, [label for _, label in metrics])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Empirical frequency")
    axes[1].set_title("Shared ranker weights; K=20", pad=6)
    panel(axes[1], "b")

    categories = ["Eligible weed", "Eligible crop", "Ambiguous/background"]
    for i, row in b.reset_index(drop=True).iterrows():
        values = [row.eligible_weed_fraction, row.eligible_crop_fraction, row.ambiguous_or_background_fraction]
        bottom = 0.0
        for value, color, label in zip(values, (COLORS["weed"], COLORS["crop"], COLORS["ambiguous"]), categories):
            axes[2].bar(i, value, bottom=bottom, color=color, width=0.58, label=label if i == 0 else None)
            bottom += value
    axes[2].set_xticks([0, 1], b.short)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Candidate-pool fraction")
    axes[2].legend(fontsize=6.0, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, borderaxespad=0.0, labelspacing=0.18)
    panel(axes[2], "c")

    fig.legend([item[0] for item in handles], ["Original", "Commissioned"], loc="upper center", bbox_to_anchor=(0.38, 0.99), ncol=2, fontsize=6.7)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.79, wspace=0.36)
    hashes, report = export(fig, output / "Figure_2")
    return hashes, {"proposal": p, "burden": b, "matched_queue": q}, report


def queue_figure(root: Path, output: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame], dict[str, object]]:
    base = root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_QUEUE_20260810_v2"
    queue = pd.read_csv(base / "official_test_queue_curves.tsv", sep="\t")
    oracle = pd.read_csv(base / "official_test_exact_oracles.tsv", sep="\t")
    proposal = pd.read_csv(
        root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_PROPOSAL_AUDIT_20260810_v1/official_split_proposal_layers.tsv",
        sep="\t",
    )
    pooled = queue[queue.aggregation == "pooled_test"].copy()
    models = [
        ("source_resnet18_role_score", "Source ResNet", COLORS["source"], "o", "-"),
        ("all_candidate_role_plus_local_logistic", "Role + local", COLORS["transparent"], "s", "--"),
        ("all_candidate_geometry_only_logistic", "Geometry only", COLORS["geometry"], "^", ":"),
        ("all_candidate_frozen_dinov2_linear", "Frozen DINOv2", COLORS["dino"], "D", "-."),
        ("all_candidate_component_masked_dinov2_linear", "Masked DINOv2", COLORS["masked"], "P", "-"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15))
    legend_handles = []
    for model, label, color, marker, linestyle in models:
        current = pooled[pooled.model == model].sort_values("k")
        line = axes[0, 0].plot(current.k, current.all_candidate_queue_precision, marker=marker, linestyle=linestyle, ms=3.8, lw=1.5, color=color, label=label)[0]
        axes[0, 1].plot(current.k, current.queued_role_qualified_weed_instance_recall, marker=marker, linestyle=linestyle, ms=3.8, lw=1.5, color=color)
        legend_handles.append(line)
    for ax, ylabel, label in ((axes[0, 0], "All-candidate queue precision", "a"), (axes[0, 1], "Role-qualified weed recall", "b")):
        ax.set_xlabel("Candidates per image, K")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 5, 10, 20, 50])
        ax.grid(axis="y", alpha=0.18)
        panel(ax, label)
    fig.legend(legend_handles, [label for _, label, _, _, _ in models], loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=6.4, handlelength=2.4)

    prop = proposal[
        (proposal.variant == "training_selected_commissioned")
        & (proposal.official_split == "test")
        & (proposal.acquisition_date == "ALL")
        & (proposal.truth_role == "weed")
    ].iloc[0]
    ora = oracle[(oracle.oracle_contract == "role_qualified") & (oracle.crop_constraint == "unconstrained") & (oracle.k == 20)].iloc[0]
    masked = pooled[(pooled.model == "all_candidate_component_masked_dinov2_linear") & (pooled.k == 20)].iloc[0]
    frozen = pooled[(pooled.model == "all_candidate_frozen_dinov2_linear") & (pooled.k == 20)].iloc[0]
    names = ["Proposal\nceiling", "Exact K=20\noracle", "Masked\nDINOv2", "Frozen\nDINOv2"]
    values = [prop.role_qualified_proposal_recall, ora.queued_role_qualified_weed_instance_recall, masked.queued_role_qualified_weed_instance_recall, frozen.queued_role_qualified_weed_instance_recall]
    axes[1, 0].bar(np.arange(4), values, color=[COLORS["commissioned"], COLORS["oracle"], COLORS["masked"], COLORS["dino"]], width=0.68)
    axes[1, 0].set_xticks(np.arange(4), names)
    axes[1, 0].set_ylim(0, 0.34)
    axes[1, 0].set_ylabel("Role-qualified weed recall")
    panel(axes[1, 0], "c")
    for i, value in enumerate(values):
        axes[1, 0].text(i, value + 0.009, f"{value:.3f}", ha="center", fontsize=6.8)

    thresholds = [0.0, 0.01, 0.10, 0.50]
    threshold_positions = np.arange(len(thresholds))
    for model, label, color, marker, linestyle in models[-2:]:
        row = pooled[(pooled.model == model) & (pooled.k == 20)].iloc[0]
        axes[1, 1].plot(threshold_positions, [row[f"crop_overlap_scene_frequency_{value:.2f}"] for value in thresholds], marker=marker, linestyle=linestyle, ms=4, lw=1.6, label=label, color=color)
    axes[1, 1].set_xticks(threshold_positions, ["0", "0.01", "0.10", "0.50"])
    axes[1, 1].set_xlabel("Minimum queued-region crop fraction")
    axes[1, 1].set_ylabel("Crop-overlap scene frequency")
    axes[1, 1].set_ylim(0, 0.5)
    axes[1, 1].legend(fontsize=6.4, loc="lower left")
    panel(axes[1, 1], "d")
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.11, top=0.88, wspace=0.30, hspace=0.38)
    hashes, report = export(fig, output / "Figure_3")
    return hashes, {"queue": pooled, "oracle": oracle, "proposal": pd.DataFrame([prop])}, report


def robustness_figure(root: Path, output: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame], dict[str, object]]:
    queue_root = root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_QUEUE_20260810_v2"
    labels = pd.read_csv(queue_root / "prospective_label_selection_results.tsv", sep="\t")
    random = pd.read_csv(queue_root / "official_test_random_ranking_draws.tsv", sep="\t")
    queues = pd.read_csv(queue_root / "official_test_queue_curves.tsv", sep="\t")
    paired = pd.read_csv(
        root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_DINOV2_FINETUNE_20260810_v1/paired_frozen_finetuned_differences.tsv",
        sep="\t",
    )
    sensitivity = pd.read_csv(
        root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_QUEUE_CONTRACT_SENSITIVITY_20260810_v1/queued_matching_contract_sensitivity.tsv",
        sep="\t",
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    strategies = [
        ("prospective_uniform", "Uniform"),
        ("prospective_dinov2_coreset", "DINOv2 coreset"),
        ("retrospective_class_stratified", "Retrospective\nstratified"),
    ]
    for x, (strategy, _) in enumerate(strategies):
        values = labels[labels.strategy == strategy].queued_role_qualified_weed_instance_recall.to_numpy()
        axes[0, 0].scatter(np.full(len(values), x) + np.linspace(-0.08, 0.08, len(values)), values, s=14, alpha=0.8, color=[COLORS["transparent"], COLORS["dino"], COLORS["source"]][x])
        axes[0, 0].plot([x - 0.18, x + 0.18], [values.mean(), values.mean()], color="black", lw=1.2)
    axes[0, 0].set_xticks(range(3), [label for _, label in strategies])
    axes[0, 0].set_ylabel("K=20 qualified recall")
    axes[0, 0].set_title("300 simulated mask-derived labels; 10 draws", pad=6)
    panel(axes[0, 0], "a")

    delta_metrics = [
        ("paired_auc_difference_finetuned_minus_frozen", "AUC"),
        ("paired_k20_precision_difference", "Queue\nprecision"),
        ("paired_k20_recall_difference", "Qualified\nrecall"),
    ]
    for x, (column, _) in enumerate(delta_metrics):
        values = paired[column].to_numpy()
        axes[0, 1].scatter(np.full(len(values), x) + np.linspace(-0.08, 0.08, len(values)), values, s=14, color=COLORS["masked"], alpha=0.8)
        axes[0, 1].plot([x - 0.18, x + 0.18], [values.mean(), values.mean()], color="black", lw=1.2)
    axes[0, 1].axhline(0, color="#666666", lw=0.8, ls="--")
    axes[0, 1].set_xticks(range(3), [label for _, label in delta_metrics])
    axes[0, 1].set_ylabel("Fine-tuned minus frozen")
    axes[0, 1].set_title("Identical DINOv2 coreset draws", pad=6)
    panel(axes[0, 1], "b")

    s = sensitivity[
        (sensitivity.model == "all_candidate_component_masked_dinov2_linear")
        & (sensitivity.k == 20)
        & (sensitivity.stratum == "ALL")
        & (sensitivity.minimum_instance_coverage == 0.50)
    ]
    matrix = s.pivot(index="minimum_same_role_purity", columns="minimum_candidate_labeled_coverage", values="many_to_one_queued_recall").sort_index(ascending=False)
    im = axes[1, 0].imshow(matrix.to_numpy(), vmin=0, vmax=max(0.25, matrix.to_numpy().max()), cmap="Blues", aspect="auto")
    axes[1, 0].set_xticks(range(len(matrix.columns)), [f"{value:.2f}" for value in matrix.columns])
    axes[1, 0].set_yticks(range(len(matrix.index)), [f"{value:.2f}" for value in matrix.index])
    axes[1, 0].set_xlabel("Minimum labeled coverage")
    axes[1, 0].set_ylabel("Minimum weed purity")
    panel(axes[1, 0], "c")
    for (row, column), value in np.ndenumerate(matrix.to_numpy()):
        axes[1, 0].text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=6.8, color="white" if value > 0.16 else "black")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Queued recall")

    random_k20 = random[random.k == 20].queued_role_qualified_weed_instance_recall.to_numpy()
    model_order = [
        ("Random", np.median(random_k20), COLORS["random"]),
        ("Source", float(queues[(queues.model == "source_resnet18_role_score") & (queues.k == 20) & (queues.aggregation == "pooled_test")].queued_role_qualified_weed_instance_recall.iloc[0]), COLORS["source"]),
        ("Role +\nlocal", float(queues[(queues.model == "all_candidate_role_plus_local_logistic") & (queues.k == 20) & (queues.aggregation == "pooled_test")].queued_role_qualified_weed_instance_recall.iloc[0]), COLORS["transparent"]),
        ("Frozen\nDINOv2", float(queues[(queues.model == "all_candidate_frozen_dinov2_linear") & (queues.k == 20) & (queues.aggregation == "pooled_test")].queued_role_qualified_weed_instance_recall.iloc[0]), COLORS["dino"]),
        ("Masked\nDINOv2", float(queues[(queues.model == "all_candidate_component_masked_dinov2_linear") & (queues.k == 20) & (queues.aggregation == "pooled_test")].queued_role_qualified_weed_instance_recall.iloc[0]), COLORS["masked"]),
    ]
    axes[1, 1].bar(np.arange(len(model_order)), [value for _, value, _ in model_order], color=[color for _, _, color in model_order], width=0.68)
    axes[1, 1].set_xticks(np.arange(len(model_order)), [label for label, _, _ in model_order])
    axes[1, 1].set_ylabel("K=20 qualified recall")
    axes[1, 1].set_ylim(0, 0.215)
    panel(axes[1, 1], "d")
    axes[1, 1].text(0, np.quantile(random_k20, 0.975) + 0.008, f"95% random\n{np.quantile(random_k20, 0.025):.3f}--{np.quantile(random_k20, 0.975):.3f}", ha="center", va="bottom", fontsize=6.5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4})
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.93, wspace=0.34, hspace=0.43)
    hashes, report = export(fig, output / "Figure_4")
    return hashes, {"label_selection": labels, "paired": paired, "contract_slice": s, "random": random[random.k == 20], "queues": queues[(queues.k == 20) & (queues.aggregation == "pooled_test")]}, report


def allocation_figure(audit: Path, output: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame], dict[str, object]]:
    pooled = pd.read_csv(audit / "official_test_queue_policy_comparison.tsv", sep="\t").set_index("policy").loc[POLICIES]
    tiles = pd.read_csv(audit / "official_test_queue_policy_per_tile.tsv", sep="\t")
    strata = pd.read_csv(audit / "official_test_queue_policy_date_patch.tsv", sep="\t")
    crop = pd.read_csv(audit / "official_test_crop_exposure_severity.tsv", sep="\t").set_index("policy").loc[POLICIES]
    fixed = tiles[tiles.policy == POLICIES[0]].set_index("file")
    global_n = tiles[tiles.policy == POLICIES[1]].set_index("file")
    common = fixed.index.intersection(global_n.index)
    delta = (global_n.loc[common, "one_to_one_role_qualified_recall"] - fixed.loc[common, "one_to_one_role_qualified_recall"]).sort_values()
    date_rows = strata[(strata.aggregation == "acquisition_date") & strata.policy.isin(POLICIES[:2])].copy()
    dates = sorted(date_rows.stratum.unique())

    sources = {
        "a": pooled.reset_index(),
        "b": delta.rename("global_minus_fixed_one_to_one_qualified_recall").reset_index(),
        "c": date_rows,
        "d": crop.reset_index(),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15))
    axa, axb, axc, axd = axes.flat
    metric_cols = ["all_candidate_queue_precision", "queued_role_qualified_weed_instance_recall", "one_to_one_role_qualified_recall"]
    metric_labels = ["Precision", "Qualified recall\n(many-to-one)", "Qualified recall\n(one-to-one)"]
    x = np.arange(len(metric_cols))
    width = 0.19
    for i, (policy, label, color) in enumerate(zip(POLICIES, POLICY_LABELS, POLICY_COLORS)):
        values = pooled.loc[policy, metric_cols].astype(float).to_numpy()
        bars = axa.bar(x + (i - 1.5) * width, values, width=width, color=color, edgecolor="#303030", linewidth=0.45, hatch="///" if policy == POLICIES[3] else None, label=label)
        for bar, value in zip(bars, values):
            axa.text(bar.get_x() + bar.get_width() / 2, value + 0.013, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2, rotation=90)
    axa.set_xticks(x, metric_labels)
    axa.set_ylim(0, 0.90)
    axa.set_ylabel("Proportion")
    axa.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=6.7, handlelength=1.2, labelspacing=0.2, borderaxespad=0.0)
    panel(axa, "a")

    values = delta.to_numpy()
    point_colors = np.where(values > 1e-12, "#2E9E44", np.where(values < -1e-12, "#B64342", "#A8A8A8"))
    axb.axhline(0, color="#606060", linewidth=0.8, linestyle="--", zorder=1)
    axb.vlines(np.arange(1, len(values) + 1), 0, values, color=point_colors, linewidth=0.8, alpha=0.75, zorder=2)
    axb.scatter(np.arange(1, len(values) + 1), values, c=point_colors, s=18, edgecolor="white", linewidth=0.35, zorder=3)
    axb.set_xlim(0.2, len(values) + 0.8)
    axb.set_xticks([5, 10, 15, 20, 25])
    axb.set_ylim(min(-0.15, values.min() - 0.03), max(0.43, values.max() + 0.03))
    axb.set_xlabel("Test tile (sorted by paired difference)")
    axb.set_ylabel("Global minus fixed\none-to-one qualified recall")
    improved = int((values > 1e-12).sum())
    unchanged = int((np.abs(values) <= 1e-12).sum())
    worsened = int((values < -1e-12).sum())
    axb.text(0.03, 0.96, f"Improved {improved} | unchanged {unchanged} | worsened {worsened}\nmedian 0.000; range {values.min():.3f} to {values.max():.3f}", transform=axb.transAxes, ha="left", va="top", fontsize=6.6)
    panel(axb, "b")

    xx = np.arange(len(dates))
    for offset, policy, label, color in ((-0.18, POLICIES[0], POLICY_LABELS[0], POLICY_COLORS[0]), (0.18, POLICIES[1], POLICY_LABELS[1], POLICY_COLORS[1])):
        rows = date_rows[date_rows.policy == policy].set_index("stratum").loc[dates]
        axc.bar(xx + offset, rows.queued_role_qualified_weed_instance_recall, width=0.34, color=color, edgecolor="#303030", linewidth=0.45, label=label)
    fixed_counts = date_rows[date_rows.policy == POLICIES[0]].set_index("stratum").loc[dates].accepted_candidates.astype(int)
    global_counts = date_rows[date_rows.policy == POLICIES[1]].set_index("stratum").loc[dates].accepted_candidates.astype(int)
    axc.set_xticks(xx, [f"{d[5:]}\nN={f}/{g}" for d, f, g in zip(dates, fixed_counts, global_counts)])
    axc.set_ylim(0, 0.54)
    axc.set_ylabel("Qualified recall (many-to-one)")
    axc.set_xlabel("Acquisition date (fixed/global accepted)")
    axc.legend(loc="upper left", fontsize=6.7)
    panel(axc, "c")

    scene = pooled.loc[POLICIES, "crop_overlap_scene_frequency_0.00"].astype(float).to_numpy()
    instance = crop.loc[POLICIES, "crop_instance_coverage_rate_at_0.50"].astype(float).to_numpy()
    xxx = np.arange(len(POLICIES))
    axd.bar(xxx - 0.17, scene, width=0.34, color="#E9A6A1", edgecolor="#303030", linewidth=0.45, label="Scenes with any crop overlap")
    axd.bar(xxx + 0.17, instance, width=0.34, color="#7884B4", edgecolor="#303030", linewidth=0.45, label="Crop instances covered >=50%")
    axd.set_xticks(xxx, ["Fixed", "Global", "Bounded", "Val.\nthresh."])
    axd.set_ylim(0, 0.62)
    axd.set_ylabel("Frequency / instance fraction")
    axd.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=6.7, ncol=1, borderaxespad=0.0, labelspacing=0.2)
    panel(axd, "d")
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.12, top=0.86, wspace=0.30, hspace=0.45)
    hashes, report = export(fig, output / "Figure_5")
    return hashes, sources, report


def save_sources(output: Path, figure_name: str, sources: object) -> dict[str, dict[str, str]]:
    if isinstance(sources, dict):
        source_files: dict[str, dict[str, str]] = {}
        for key, frame in sources.items():
            path = output / f"{figure_name}_{key}_source_data.tsv"
            frame.to_csv(path, sep="\t", index=False)
            source_files[str(key)] = {"path": path.name, "sha256": sha256(path)}
        return source_files
    path = output / f"{figure_name}_source_data.tsv"
    rows = list(sources)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"workflow": {"path": path.name, "sha256": sha256(path)}}


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    allocation = args.allocation_audit_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest: dict[str, object] = {
        "backend": "Python/matplotlib",
        "editable_svg_text": True,
        "figure_contract": {
            "target_width_mm": 183,
            "archetypes": {
                "Figure_1": "schematic-led composite",
                "Figure_2": "quantitative grid",
                "Figure_3": "quantitative grid",
                "Figure_4": "quantitative grid",
                "Figure_5": "quantitative grid",
            },
            "render_qa": "text bounding-box collision and canvas-boundary audit plus visual inspection",
        },
        "figures": {},
    }
    builders = [
        ("Figure_1", lambda: workflow_figure(output)),
        ("Figure_2", lambda: proposal_figure(root, output)),
        ("Figure_3", lambda: queue_figure(root, output)),
        ("Figure_4", lambda: robustness_figure(root, output)),
        ("Figure_5", lambda: allocation_figure(allocation, output)),
    ]
    for name, builder in builders:
        hashes, sources, report = builder()
        manifest["figures"][name] = {
            "exports": hashes,
            "source_data": save_sources(output, name, sources),
            "text_layout_audit": {
                "path": f"{name}_text_layout_audit.json",
                "sha256": sha256(output / f"{name}_text_layout_audit.json"),
                "collision_count": report["collision_count"],
                "outside_canvas_count": report["outside_canvas_count"],
            },
        }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: info["text_layout_audit"] for name, info in manifest["figures"].items()}, indent=2))


if __name__ == "__main__":
    main()
