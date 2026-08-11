#!/usr/bin/env python3
"""Build the evidence-traceable figures for the SAT revision-6 manuscript."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B7280",
    "light_blue": "#E8F1F8",
    "light_orange": "#FFF2D8",
    "light_green": "#E5F4EE",
    "light_gray": "#F3F4F6",
    "ink": "#17202A",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.65,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out: Path, stem: str) -> list[Path]:
    outputs: list[Path] = []
    for suffix, kwargs in (
        (".pdf", {"bbox_inches": "tight"}),
        (".svg", {"bbox_inches": "tight"}),
        (".png", {"dpi": 320, "bbox_inches": "tight"}),
        (".tif", {"dpi": 600, "bbox_inches": "tight", "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = out / f"{stem}{suffix}"
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def rounded_box(ax, xy, wh, text, face, edge, fontsize=8.7, weight="normal") -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.15,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color=COLORS["ink"],
        linespacing=1.25,
    )


def arrow(ax, start, end, color=COLORS["gray"]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=color,
        )
    )


def figure_workflow(out: Path) -> tuple[list[Path], dict]:
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.96,
        "Failure-aware commissioning with an explicit annotation boundary",
        fontsize=14,
        weight="bold",
        color=COLORS["ink"],
        va="top",
    )

    x_positions = [0.02, 0.205, 0.39, 0.575, 0.76]
    labels = [
        "Public RGB tiles\n156 images, four dates",
        "Label-free proposal\ngeneration on RGB",
        "Training-date\ndense masks select\nproposal setting",
        "Candidate roles\nfit and select\nranker",
        "Fixed-budget queue\nK = 1, 5, 10, 20\nper image",
    ]
    faces = [COLORS["light_blue"], COLORS["light_blue"], COLORS["light_orange"], COLORS["light_green"], "#F4E8F1"]
    edges = [COLORS["blue"], COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]
    for index, (x, label, face, edge) in enumerate(zip(x_positions, labels, faces, edges)):
        rounded_box(ax, (x, 0.68), (0.16, 0.17), label, face, edge, fontsize=7.5, weight="bold" if index >= 2 else "normal")
        if index < len(x_positions) - 1:
            arrow(ax, (x + 0.16, 0.765), (x_positions[index + 1] - 0.008, 0.765))

    ax.plot([0.02, 0.92], [0.585, 0.585], color=COLORS["red"], lw=1.4, ls=(0, (5, 3)))
    ax.text(0.02, 0.605, "Annotation boundary", color=COLORS["red"], weight="bold")
    ax.text(
        0.92,
        0.605,
        "Held-out masks and roles enter offline scoring only",
        ha="right",
        color=COLORS["red"],
        fontsize=8.7,
    )

    rounded_box(ax, (0.07, 0.35), (0.25, 0.13), "Outer leave-one-date-out\nthree training dates\none held-out date", "#FFF7E8", COLORS["orange"], 8.0)
    rounded_box(ax, (0.375, 0.35), (0.25, 0.13), "Three evaluation layers\nspatial / role-qualified\nqueued recall", "#EEF5FB", COLORS["blue"], 8.0)
    rounded_box(ax, (0.68, 0.35), (0.25, 0.13), "Repeat over all four dates\ndate-macro reporting", COLORS["light_gray"], COLORS["gray"], 8.2)
    arrow(ax, (0.32, 0.415), (0.375, 0.415), COLORS["orange"])
    arrow(ax, (0.625, 0.415), (0.68, 0.415), COLORS["gray"])

    ax.text(0.02, 0.285, "Supervision ledger", fontsize=10.2, weight="bold", color=COLORS["ink"])
    ledger = [
        ("Source masks", "7,050 source images", COLORS["light_gray"], COLORS["gray"]),
        ("Target dense masks", "108-144 training-date tiles/fold", COLORS["light_orange"], COLORS["orange"]),
        ("Candidate roles", "300 sampled labels/fold", COLORS["light_green"], COLORS["green"]),
        ("Held-out labels", "12-48 tiles; scoring only", "#FDECEC", COLORS["red"]),
    ]
    for index, (title, detail, face, edge) in enumerate(ledger):
        x = 0.04 + index * 0.235
        rounded_box(ax, (x, 0.13), (0.20, 0.10), f"{title}\n{detail}", face, edge, 7.9)

    ax.text(
        0.50,
        0.055,
        "Supported scope: single-source offline commissioning and fixed-budget ranking; no operator, independent-farm, or deployment claim.",
        ha="center",
        va="center",
        fontsize=8.7,
        color=COLORS["ink"],
    )
    files = save_figure(fig, out, "figure_1_workflow_and_annotation_boundary")
    source = {
        "figure": "Figure 1",
        "conclusion": "The workflow separates label-free proposal generation from supervised setting selection and held-out offline scoring.",
        "inputs": ["paper/revision_stage6_20260810/derived_tables_v1/annotation_budget.tsv"],
    }
    return files, source


def figure_proposals(root: Path, out: Path) -> tuple[list[Path], dict]:
    contract = root / "results/p2_development/P2_WEEDSGALORE_COMMISSIONING_CONTRACT_AUDIT_20260810_v2"
    proposal = read_tsv(contract / "proposal_recall_layers.tsv")
    split = read_tsv(contract / "split_merge_sensitivity.tsv")
    sensitivity = read_tsv(contract / "matching_contract_sensitivity.tsv")
    dates = ["2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15"]
    date_labels = ["25 May", "30 May", "6 Jun", "15 Jun"]
    variants = ["original", "nested_commissioned"]
    variant_labels = ["Original", "Commissioned"]
    colors = [COLORS["gray"], COLORS["blue"]]

    fig, axes = plt.subplots(2, 2, figsize=(10.9, 7.2), gridspec_kw={"wspace": 0.30, "hspace": 0.42})
    ax = axes[0, 0]
    x = np.arange(len(dates))
    width = 0.36
    for offset, variant, label, color in zip((-width / 2, width / 2), variants, variant_labels, colors):
        values = [float(next(r for r in proposal if r["held_out_date"] == date and r["variant"] == variant and r["truth_role"] == "weed")["spatial_proposal_recall"]) for date in dates]
        ax.bar(x + offset, values, width, color=color, label=label)
    ax.set_xticks(x, date_labels)
    ax.set_ylim(0, 0.70)
    ax.set_ylabel("Spatial proposal recall")
    ax.set_title("a  Spatial coverage", loc="left", weight="bold")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    for offset, variant, label, color in zip((-width / 2, width / 2), variants, variant_labels, colors):
        values = [float(next(r for r in proposal if r["held_out_date"] == date and r["variant"] == variant and r["truth_role"] == "weed")["role_qualified_proposal_recall"]) for date in dates]
        ax.bar(x + offset, values, width, color=color, label=label)
    ax.set_xticks(x, date_labels)
    ax.set_ylim(0, 0.52)
    ax.set_ylabel("Role-qualified proposal recall")
    ax.set_title("b  Role-qualified coverage", loc="left", weight="bold")

    ax = axes[1, 0]
    thresholds = [0.10, 0.25, 0.50]
    for variant, label, color, marker in zip(variants, variant_labels, colors, ("o", "s")):
        split_values = [sum(int(r["split_instances"]) for r in split if r["variant"] == variant and float(r["instance_coverage_threshold"]) == threshold) for threshold in thresholds]
        merge_values = [sum(int(r["merge_candidates"]) for r in split if r["variant"] == variant and float(r["instance_coverage_threshold"]) == threshold) for threshold in thresholds]
        ax.plot(thresholds, split_values, marker=marker, color=color, lw=1.8, label=f"{label}: split")
        ax.plot(thresholds, merge_values, marker=marker, color=color, lw=1.6, ls="--", label=f"{label}: merge")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(thresholds)
    ax.set_xlabel("Instance-coverage threshold")
    ax.set_ylabel("Count across four dates")
    ax.set_title("c  Split/merge sensitivity", loc="left", weight="bold")
    ax.legend(ncol=2, fontsize=6.8, loc="upper center", bbox_to_anchor=(0.5, 1.0))

    ax = axes[1, 1]
    filtered = [r for r in sensitivity if r["truth_role"] == "weed" and float(r["minimum_instance_coverage"]) == 0.50]
    labeled_thresholds = [0.25, 0.50, 0.75]
    purity_thresholds = [0.50, 0.75, 0.90]
    for variant, label, color, marker in zip(variants, variant_labels, colors, ("o", "s")):
        values = []
        for labeled in labeled_thresholds:
            subset = [r for r in filtered if r["variant"] == variant and float(r["minimum_candidate_labeled_coverage"]) == labeled and float(r["minimum_same_role_purity"]) == 0.90]
            values.append(mean(r["role_qualified_proposal_recall"] for r in subset))
        ax.plot(labeled_thresholds, values, marker=marker, color=color, lw=2, label=f"{label}; purity=0.90")
    for purity, ls in zip(purity_thresholds[:2], (":", "--")):
        values = []
        for labeled in labeled_thresholds:
            subset = [r for r in filtered if r["variant"] == "nested_commissioned" and float(r["minimum_candidate_labeled_coverage"]) == labeled and float(r["minimum_same_role_purity"]) == purity]
            values.append(mean(r["role_qualified_proposal_recall"] for r in subset))
        ax.plot(labeled_thresholds, values, color=COLORS["blue"], ls=ls, lw=1.5, label=f"Commissioned; purity={purity:.2f}")
    ax.set_xticks(labeled_thresholds)
    ax.set_ylim(0, 0.36)
    ax.set_xlabel("Candidate labeled-coverage threshold")
    ax.set_ylabel("Date-macro qualified recall")
    ax.set_title("d  Matching-contract sensitivity", loc="left", weight="bold")
    ax.legend(fontsize=7.1)

    fig.suptitle("Commissioning increases proposal coverage, but recall depends on the evaluation contract", x=0.04, ha="left", fontsize=13.2, weight="bold", color=COLORS["ink"])
    files = save_figure(fig, out, "figure_2_proposal_layers_and_sensitivity")
    source = {
        "figure": "Figure 2",
        "conclusion": "Spatial and role-qualified recall are distinct; split/merge counts and qualified recall are threshold-sensitive.",
        "inputs": [str(contract / name) for name in ("proposal_recall_layers.tsv", "split_merge_sensitivity.tsv", "matching_contract_sensitivity.tsv")],
    }
    return files, source


def queue_macro(rows: list[dict[str, str]], metric: str, *, model: str, variant: str, repeat: str | None = None) -> dict[int, float]:
    selected = [r for r in rows if r["model"] == model and r["variant"] == variant and (repeat is None or r.get("repeat") == repeat)]
    result = {}
    for k in (1, 5, 10, 20):
        group = [r for r in selected if int(r["k"]) == k]
        result[k] = mean(r[metric] for r in group)
    return result


def figure_queue(root: Path, out: Path) -> tuple[list[Path], dict]:
    contract_dir = root / "results/p2_development/P2_WEEDSGALORE_COMMISSIONING_CONTRACT_AUDIT_20260810_v2"
    dino_dir = root / "results/p2_development/P2_WEEDSGALORE_DINOV2_QUEUE_20260810_v2"
    contract = read_tsv(contract_dir / "commissioning_queue_comparison.tsv")
    dino = read_tsv(dino_dir / "dinov2_queue_by_date_repeat.tsv")
    series = [
        ("Original role+local", contract, "role_plus_local", "original", COLORS["gray"], "o"),
        ("Commissioned role+local", contract, "role_plus_local", "nested_commissioned", COLORS["blue"], "s"),
        ("Commissioned DINOv2 fine-tuned", dino, "dinov2_last_block_finetune", "nested_commissioned", COLORS["green"], "^"),
    ]
    metrics = [
        ("all_candidate_precision", "All-candidate precision", "a  Candidate precision"),
        ("queued_role_qualified_weed_instance_recall", "Queued weed-instance recall", "b  Role-qualified queue recall"),
        ("crop_overlap_scene_frequency_0.00", "Scenes with any crop overlap", "c  Crop-exposure frequency"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), gridspec_kw={"wspace": 0.34})
    k_values = np.array([1, 5, 10, 20])
    source_values = {}
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for label, rows, model, variant, color, marker in series:
            if rows is dino:
                by_repeat = [queue_macro(rows, metric, model=model, variant=variant, repeat=str(rep)) for rep in (0, 1, 2)]
                values = np.array([mean(item[k] for item in by_repeat) for k in k_values])
                low = np.array([min(item[k] for item in by_repeat) for k in k_values])
                high = np.array([max(item[k] for item in by_repeat) for k in k_values])
                ax.fill_between(k_values, low, high, color=color, alpha=0.16, linewidth=0)
            else:
                values_map = queue_macro(rows, metric, model=model, variant=variant)
                values = np.array([values_map[k] for k in k_values])
            ax.plot(k_values, values, marker=marker, color=color, lw=2, ms=5, label=label)
            source_values[f"{metric}:{label}"] = values.tolist()
        ax.set_xticks(k_values)
        ax.set_xlabel("Queue budget K per image")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7.5, loc="best")
    fig.suptitle("Modern target adaptation improves the queue, while proposal coverage remains limiting", x=0.03, ha="left", fontsize=13.2, weight="bold", color=COLORS["ink"])
    files = save_figure(fig, out, "figure_3_end_to_end_queue_comparison")
    source = {
        "figure": "Figure 3",
        "conclusion": "At fixed K, DINOv2 last-block adaptation raises qualified recall and precision while lowering crop exposure relative to the commissioned transparent ranker.",
        "inputs": [str(contract_dir / "commissioning_queue_comparison.tsv"), str(dino_dir / "dinov2_queue_by_date_repeat.tsv")],
        "date_macro_values": source_values,
    }
    return files, source


def figure_baselines(root: Path, out: Path) -> tuple[list[Path], dict]:
    modern_dir = root / "results/p2_development/P2_WEEDSGALORE_MODERN_FROZEN_BASELINES_20260810_v1"
    fine_dir = root / "results/p2_development/P2_WEEDSGALORE_DINOV2_TARGET_FINETUNE_20260810_v3"
    summary = read_json(modern_dir / "summary.json")
    paired = read_tsv(modern_dir / "paired_auc_differences.tsv")
    fine = read_tsv(fine_dir / "target_finetune_by_date_repeat.tsv")
    names = ["Source\nResNet-18", "Role +\nlocal", "Frozen\nDINOv2", "Fine-tuned\nDINOv2"]
    colors = [COLORS["gray"], COLORS["orange"], COLORS["blue"], COLORS["green"]]
    full_values = [
        summary["full_label_date_macro"]["source_semantic_frozen"]["roc_auc"],
        summary["full_label_date_macro"]["role_plus_local"]["roc_auc"],
        summary["full_label_date_macro"]["dinov2_frozen"]["roc_auc"],
        np.nan,
    ]
    budget_values = [
        summary["same_label_budget"]["source_semantic_frozen"]["mean"],
        summary["same_label_budget"]["role_plus_local"]["mean"],
        summary["same_label_budget"]["dinov2_frozen"]["mean"],
        mean(r["roc_auc"] for r in fine),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), gridspec_kw={"wspace": 0.36})
    ax = axes[0]
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, full_values, width, color=colors, alpha=0.50, label="All training candidates")
    ax.bar(x + width / 2, budget_values, width, color=colors, label="100 labels/training date")
    ax.set_xticks(x, names)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Date-macro ROC AUC")
    ax.set_title("a  Candidate-ranking baselines", loc="left", weight="bold")
    ax.legend(fontsize=7.3, loc="lower right")

    ax = axes[1]
    first = [float(r["role_plus_local_minus_dinov2"]) for r in paired]
    second = [float(r["dinov2_minus_source_semantic"]) for r in paired]
    jitter = np.linspace(-0.08, 0.08, len(first))
    ax.axhline(0, color=COLORS["gray"], ls="--", lw=1)
    ax.scatter(np.zeros(len(first)) + jitter, first, color=COLORS["orange"], s=25, alpha=0.85)
    ax.scatter(np.ones(len(second)) + jitter, second, color=COLORS["blue"], s=25, alpha=0.85)
    ax.plot([-0.16, 0.16], [mean(first), mean(first)], color=COLORS["ink"], lw=2.2)
    ax.plot([0.84, 1.16], [mean(second), mean(second)], color=COLORS["ink"], lw=2.2)
    ax.set_xticks([0, 1], ["Role+local -\nfrozen DINOv2", "Frozen DINOv2 -\nsource ResNet-18"])
    ax.set_ylabel("Paired date-macro AUC difference")
    ax.set_title("b  Ten paired label draws", loc="left", weight="bold")

    ax = axes[2]
    dates = ["2023-05-25", "2023-05-30", "2023-06-06", "2023-06-15"]
    labels = ["25 May", "30 May", "6 Jun", "15 Jun"]
    for rep, color in zip((0, 1, 2), ("#66C2A5", "#238B45", "#006D2C")):
        values = [float(next(r for r in fine if r["held_out_date"] == date and int(r["repeat"]) == rep)["roc_auc"]) for date in dates]
        ax.plot(range(4), values, marker="o", color=color, lw=1.6, label=f"Repeat {rep + 1}")
    ax.set_xticks(range(4), labels, rotation=22, ha="right")
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("Held-out-date ROC AUC")
    ax.set_title("c  Last-block fine-tuning", loc="left", weight="bold")
    ax.legend(fontsize=7.4, loc="lower left")

    fig.suptitle("A modern self-supervised encoder strengthens candidate ranking under the same label budget", x=0.03, ha="left", fontsize=13.2, weight="bold", color=COLORS["ink"])
    files = save_figure(fig, out, "figure_4_modern_baselines_and_paired_results")
    source = {
        "figure": "Figure 4",
        "conclusion": "Frozen DINOv2 clearly exceeds the source-trained ResNet-18 and last-block target fine-tuning provides the strongest mean AUC.",
        "inputs": [str(modern_dir / name) for name in ("summary.json", "paired_auc_differences.tsv")] + [str(fine_dir / "target_finetune_by_date_repeat.tsv")],
    }
    return files, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure directory: {out}")
    out.mkdir(parents=True)
    configure_style()
    sources = []
    files: list[Path] = []
    for builder in (
        lambda: figure_workflow(out),
        lambda: figure_proposals(root, out),
        lambda: figure_queue(root, out),
        lambda: figure_baselines(root, out),
    ):
        built, source = builder()
        files.extend(built)
        sources.append(source)
    manifest = {
        "figure_contract": {
            "conclusion": "Separating proposal and queue failures reveals that commissioning improves spatial coverage, while modern target adaptation repairs ranking without removing the proposal ceiling.",
            "backend": "Python/matplotlib",
            "journal_width": "double-column, 183 mm target",
            "exports": ["PDF", "SVG", "PNG", "TIFF 600 dpi"],
        },
        "figures": sources,
        "outputs": {path.name: sha256(path) for path in sorted(files)},
        "qa": [
            "No rasterized text in PDF/SVG exports.",
            "Colorblind-safe palette with markers and line styles as redundant encodings.",
            "All axes state metric, unit, and evaluation layer.",
            "Source tables are listed per figure and retained in the revision package.",
        ],
    }
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "files": len(files), "manifest": str(out / "figure_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
