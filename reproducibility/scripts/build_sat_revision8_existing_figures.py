#!/usr/bin/env python3
"""Rebuild SAT Revision 8 Figures 1--4 with readable final-size typography."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.75, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "xtick.major.width": 0.7, "ytick.major.width": 0.7})

COLORS = {
    "original": "#7884B4", "commissioned": "#E4CCD8", "source": "#A8A8A8",
    "transparent": "#7884B4", "geometry": "#B4C0E4", "dino": "#E4A3B7",
    "masked": "#B64342", "oracle": "#2E9E44", "random": "#D8D8D8",
    "weed": "#5B8F4E", "crop": "#D69B45", "ambiguous": "#B5B5B5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(fig: plt.Figure, base: Path) -> dict[str, str]:
    outputs = {}
    for extension, kwargs in (("svg", {}), ("pdf", {}), ("tif", {"dpi": 600}), ("png", {"dpi": 220})):
        path = base.with_suffix("." + extension)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[extension] = sha256(path)
    plt.close(fig)
    return outputs


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def workflow_figure(output: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (0.03, 0.64, 0.18, 0.22, "Public source\nSugarBeets RGB + masks", "#E0E0F0"),
        (0.27, 0.64, 0.18, 0.22, "Frozen RGB specialists\nforeground + crop/weed", "#E0F0F0"),
        (0.52, 0.69, 0.20, 0.18, "WeedsGalore RGB\ntruth-free proposals", "#F0E0D0"),
        (0.78, 0.69, 0.19, 0.18, "Review allocation\nfixed, global, bounded,\nvalidation threshold", "#E4CCD8"),
        (0.05, 0.25, 0.22, 0.20, "Official train (104 tiles)\nselect proposals; fit rankers\nand target semantic baseline", "#DCEBFA"),
        (0.38, 0.25, 0.22, 0.20, "Official validation (26)\nselect C, epoch, threshold,\narea, or queue threshold", "#E5E5EF"),
        (0.71, 0.25, 0.24, 0.20, "Official test (26)\noffline score declared outputs\n1,712 raster weed instances", "#F5E1E1"),
    ]
    for x, y, width, height, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.014,rounding_size=0.015", facecolor=color, edgecolor="#555555", linewidth=0.8))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=7.5)
    arrows = [((0.21, 0.75), (0.27, 0.75)), ((0.45, 0.75), (0.52, 0.75)), ((0.72, 0.78), (0.78, 0.78)), ((0.16, 0.45), (0.56, 0.69)), ((0.49, 0.45), (0.61, 0.69)), ((0.83, 0.45), (0.87, 0.69))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color="#555555", linewidth=0.8, connectionstyle="arc3,rad=0.0"))
    ax.text(0.02, 0.95, "a", fontweight="bold", fontsize=9)
    ax.text(0.03, 0.085, "Supervision boundary", fontweight="bold")
    ax.text(0.03, 0.052, "Source dense masks → specialist training   |   Target-train dense masks → proposal setting   |   Candidate role labels → ranker fitting   |   Test truth → offline scoring only", fontsize=6.35)
    ax.text(0.03, 0.014, "RGB-only design isolates transfer from the public RGB SugarBeets specialist; red-edge and NIR remain an untested sensing extension.", fontsize=6.35, color="#555555")
    for text in ax.texts:
        if text.get_text().startswith("Source dense masks"):
            text.set_visible(False)
    ax.add_patch(Rectangle((0.0, 0.037), 1.5, 0.040, facecolor="white", edgecolor="none", zorder=9, clip_on=False))
    ax.text(0.03, 0.052, "Source masks -> specialists   |   Target-train masks -> proposal/semantic fitting   |   Candidate labels -> ranker fitting   |   Test truth -> offline scoring", fontsize=6.55, zorder=10)
    source = [{"stage": text.replace("\n", " "), "x": x, "y": y, "width": width, "height": height} for x, y, width, height, text, _ in boxes]
    return export(fig, output / "Figure_1"), source


def proposal_figure(root: Path, output: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    proposal = pd.read_csv(root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_PROPOSAL_AUDIT_20260810_v1/official_split_proposal_layers.tsv", sep="\t")
    burden = pd.read_csv(root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_MATCHED_PROPOSAL_QUEUE_20260810_v1/proposal_burden_and_composition.tsv", sep="\t")
    queue = pd.read_csv(root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_MATCHED_PROPOSAL_QUEUE_20260810_v1/matched_weight_queue_curves.tsv", sep="\t")
    p = proposal[(proposal.official_split == "test") & (proposal.acquisition_date == "ALL") & (proposal.truth_role == "weed")].copy()
    p["short"] = p.variant.map({"original_T095_A032": "Original", "training_selected_commissioned": "Commissioned"})
    b = burden[burden.official_split == "test"].copy(); b["short"] = b.variant.map({"original_T095_A032": "Original", "commissioned_T070_A032": "Commissioned"})
    q = queue[(queue.aggregation == "pooled_test") & (queue.k == 20)].copy(); q["short"] = q.variant.map({"original_T095_A032": "Original", "commissioned_T070_A032": "Commissioned"})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), gridspec_kw={"width_ratios": [1.0, 1.45, 1.0]})
    x = np.arange(2); width = 0.34
    axes[0].bar(x - width / 2, p.spatial_proposal_recall, width, color=COLORS["original"], label="Spatial")
    axes[0].bar(x + width / 2, p.role_qualified_proposal_recall, width, color=COLORS["commissioned"], label="Role-qualified")
    axes[0].set_xticks(x, p.short); axes[0].set_ylim(0, 0.65); axes[0].set_ylabel("Official-test weed recall"); axes[0].legend(fontsize=6, loc="upper left"); panel(axes[0], "a")
    for i, value in enumerate(b.candidates_per_image_mean): axes[0].text(i, 0.615, f"{value:.1f}\nproposals/image", ha="center", va="top", fontsize=5.8, color="#555555")
    metrics = [("all_candidate_queue_precision", "Queue precision"), ("queued_spatial_weed_instance_recall", "Spatial recall"), ("queued_role_qualified_weed_instance_recall", "Qualified recall"), ("crop_overlap_scene_frequency_0.00", "Any crop overlap")]
    positions = np.arange(len(metrics)); bw = 0.36
    for offset, label, color in ((-bw/2, "Original", COLORS["original"]), (bw/2, "Commissioned", COLORS["commissioned"])):
        row = q[q.short == label].iloc[0]
        axes[1].bar(positions + offset, [row[key] for key, _ in metrics], bw, color=color, label=label)
    axes[1].set_xticks(positions, [label for _, label in metrics], rotation=22, ha="right"); axes[1].set_ylim(0, 1.0); axes[1].set_ylabel("Empirical frequency"); axes[1].legend(fontsize=6); axes[1].set_title("Identical ranker weights, K=20", fontsize=7); panel(axes[1], "b")
    categories = ["Eligible weed", "Eligible crop", "Ambiguous/background"]
    for i, row in b.reset_index(drop=True).iterrows():
        values = [row.eligible_weed_fraction, row.eligible_crop_fraction, row.ambiguous_or_background_fraction]
        bottom = 0
        for value, color, label in zip(values, (COLORS["weed"], COLORS["crop"], COLORS["ambiguous"]), categories):
            axes[2].bar(i, value, bottom=bottom, color=color, width=0.58, label=label if i == 0 else None); bottom += value
    axes[2].set_xticks([0, 1], b.short); axes[2].set_ylim(0, 1); axes[2].set_ylabel("Candidate-pool fraction"); axes[2].legend(fontsize=5.8, loc="upper left"); panel(axes[2], "c")
    fig.tight_layout(w_pad=1.3)
    return export(fig, output / "Figure_2"), {"proposal": p, "burden": b, "matched_queue": q}


def queue_figure(root: Path, output: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    base = root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_QUEUE_20260810_v2"
    queue = pd.read_csv(base / "official_test_queue_curves.tsv", sep="\t")
    oracle = pd.read_csv(base / "official_test_exact_oracles.tsv", sep="\t")
    proposal = pd.read_csv(root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_PROPOSAL_AUDIT_20260810_v1/official_split_proposal_layers.tsv", sep="\t")
    pooled = queue[queue.aggregation == "pooled_test"].copy()
    models = [
        ("source_resnet18_role_score", "Source ResNet", COLORS["source"]),
        ("all_candidate_role_plus_local_logistic", "Role + local", COLORS["transparent"]),
        ("all_candidate_geometry_only_logistic", "Geometry only", COLORS["geometry"]),
        ("all_candidate_frozen_dinov2_linear", "Frozen DINOv2", COLORS["dino"]),
        ("all_candidate_component_masked_dinov2_linear", "Masked DINOv2", COLORS["masked"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for model, label, color in models:
        current = pooled[pooled.model == model].sort_values("k")
        axes[0, 0].plot(current.k, current.all_candidate_queue_precision, marker="o", ms=3.5, lw=1.5, color=color, label=label)
        axes[0, 1].plot(current.k, current.queued_role_qualified_weed_instance_recall, marker="o", ms=3.5, lw=1.5, color=color, label=label)
    for ax, ylabel, label in ((axes[0,0], "All-candidate queue precision", "a"), (axes[0,1], "Role-qualified weed recall", "b")):
        ax.set_xlabel("Candidates per image, K"); ax.set_ylabel(ylabel); ax.set_xticks([1,5,10,20,50]); ax.grid(axis="y", alpha=0.18); panel(ax, label)
    axes[0,0].legend(fontsize=5.7, ncol=2)
    prop = proposal[(proposal.variant == "training_selected_commissioned") & (proposal.official_split == "test") & (proposal.acquisition_date == "ALL") & (proposal.truth_role == "weed")].iloc[0]
    ora = oracle[(oracle.oracle_contract == "role_qualified") & (oracle.crop_constraint == "unconstrained") & (oracle.k == 20)].iloc[0]
    masked = pooled[(pooled.model == "all_candidate_component_masked_dinov2_linear") & (pooled.k == 20)].iloc[0]
    frozen = pooled[(pooled.model == "all_candidate_frozen_dinov2_linear") & (pooled.k == 20)].iloc[0]
    names = ["Proposal\nceiling", "Exact K=20\noracle", "Masked\nDINOv2", "Frozen\nDINOv2"]
    values = [prop.role_qualified_proposal_recall, ora.queued_role_qualified_weed_instance_recall, masked.queued_role_qualified_weed_instance_recall, frozen.queued_role_qualified_weed_instance_recall]
    axes[1,0].bar(np.arange(4), values, color=[COLORS["commissioned"], COLORS["oracle"], COLORS["masked"], COLORS["dino"]], width=0.68)
    axes[1,0].set_xticks(np.arange(4), names); axes[1,0].set_ylim(0, 0.34); axes[1,0].set_ylabel("Role-qualified weed recall"); panel(axes[1,0], "c")
    for i, value in enumerate(values): axes[1,0].text(i, value + 0.009, f"{value:.3f}", ha="center", fontsize=6)
    thresholds = [0.0, 0.01, 0.10, 0.50]
    threshold_positions = np.arange(len(thresholds))
    for model, label, color in models[-2:]:
        row = pooled[(pooled.model == model) & (pooled.k == 20)].iloc[0]
        axes[1,1].plot(threshold_positions, [row[f"crop_overlap_scene_frequency_{value:.2f}"] for value in thresholds], marker="o", ms=4, lw=1.6, label=label, color=color)
    axes[1,1].set_xticks(threshold_positions, ["0", "0.01", "0.10", "0.50"]); axes[1,1].set_xlabel("Minimum queued-region crop fraction"); axes[1,1].set_ylabel("Crop-overlap scene frequency"); axes[1,1].set_ylim(0, 0.5); axes[1,1].legend(fontsize=6); panel(axes[1,1], "d")
    fig.tight_layout(h_pad=1.5, w_pad=1.3)
    return export(fig, output / "Figure_3"), {"queue": pooled, "oracle": oracle, "proposal": pd.DataFrame([prop])}


def robustness_figure(root: Path, output: Path) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    queue_root = root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_QUEUE_20260810_v2"
    labels = pd.read_csv(queue_root / "prospective_label_selection_results.tsv", sep="\t")
    random = pd.read_csv(queue_root / "official_test_random_ranking_draws.tsv", sep="\t")
    queues = pd.read_csv(queue_root / "official_test_queue_curves.tsv", sep="\t")
    paired = pd.read_csv(root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_SPATIAL_DINOV2_FINETUNE_20260810_v1/paired_frozen_finetuned_differences.tsv", sep="\t")
    sensitivity = pd.read_csv(root / "results/p2_development/P2_WEEDSGALORE_OFFICIAL_QUEUE_CONTRACT_SENSITIVITY_20260810_v1/queued_matching_contract_sensitivity.tsv", sep="\t")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    strategies = [("prospective_uniform", "Uniform"), ("prospective_dinov2_coreset", "DINOv2 coreset"), ("retrospective_class_stratified", "Retrospective\nstratified")]
    for x, (strategy, label) in enumerate(strategies):
        values = labels[labels.strategy == strategy].queued_role_qualified_weed_instance_recall.to_numpy()
        axes[0,0].scatter(np.full(len(values), x) + np.linspace(-0.08,0.08,len(values)), values, s=14, alpha=0.8, color=[COLORS["transparent"], COLORS["dino"], COLORS["source"]][x])
        axes[0,0].plot([x-0.18,x+0.18], [values.mean(),values.mean()], color="black", lw=1.2)
    axes[0,0].set_xticks(range(3), [label for _, label in strategies]); axes[0,0].set_ylabel("K=20 qualified recall"); axes[0,0].set_title("300 labels; 10 draws", fontsize=7); panel(axes[0,0], "a")
    delta_metrics = [("paired_auc_difference_finetuned_minus_frozen", "AUC"), ("paired_k20_precision_difference", "Queue precision"), ("paired_k20_recall_difference", "Qualified recall")]
    for x, (column, label) in enumerate(delta_metrics):
        values = paired[column].to_numpy(); axes[0,1].scatter(np.full(len(values), x) + np.linspace(-.08,.08,len(values)), values, s=14, color=COLORS["masked"], alpha=.8); axes[0,1].plot([x-.18,x+.18],[values.mean(),values.mean()],color="black",lw=1.2)
    axes[0,1].axhline(0, color="#666666", lw=.8, ls="--"); axes[0,1].set_xticks(range(3), [label for _, label in delta_metrics], rotation=15, ha="right"); axes[0,1].set_ylabel("Fine-tuned minus frozen"); axes[0,1].set_title("Identical DINOv2 coreset draws", fontsize=7); panel(axes[0,1], "b")
    s = sensitivity[(sensitivity.model == "all_candidate_component_masked_dinov2_linear") & (sensitivity.k == 20) & (sensitivity.stratum == "ALL") & (sensitivity.minimum_instance_coverage == 0.50)]
    matrix = s.pivot(index="minimum_same_role_purity", columns="minimum_candidate_labeled_coverage", values="many_to_one_queued_recall").sort_index(ascending=False)
    im = axes[1,0].imshow(matrix.to_numpy(), vmin=0, vmax=max(0.25, matrix.to_numpy().max()), cmap="Blues", aspect="auto")
    axes[1,0].set_xticks(range(len(matrix.columns)), [f"{value:.2f}" for value in matrix.columns]); axes[1,0].set_yticks(range(len(matrix.index)), [f"{value:.2f}" for value in matrix.index]); axes[1,0].set_xlabel("Minimum labeled coverage"); axes[1,0].set_ylabel("Minimum weed purity"); panel(axes[1,0], "c")
    for (row, column), value in np.ndenumerate(matrix.to_numpy()): axes[1,0].text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=6, color="white" if value > 0.16 else "black")
    fig.colorbar(im, ax=axes[1,0], fraction=.046, pad=.04, label="Queued recall")
    random_k20 = random[random.k == 20].queued_role_qualified_weed_instance_recall.to_numpy()
    model_order = [("Random", np.median(random_k20), COLORS["random"]), ("Source", float(queues[(queues.model == "source_resnet18_role_score") & (queues.k == 20) & (queues.aggregation == "pooled_test")].queued_role_qualified_weed_instance_recall.iloc[0]), COLORS["source"]), ("Role + local", float(queues[(queues.model == "all_candidate_role_plus_local_logistic") & (queues.k == 20) & (queues.aggregation == "pooled_test")].queued_role_qualified_weed_instance_recall.iloc[0]), COLORS["transparent"]), ("Frozen DINO", float(queues[(queues.model == "all_candidate_frozen_dinov2_linear") & (queues.k == 20) & (queues.aggregation == "pooled_test")].queued_role_qualified_weed_instance_recall.iloc[0]), COLORS["dino"]), ("Masked DINO", float(queues[(queues.model == "all_candidate_component_masked_dinov2_linear") & (queues.k == 20) & (queues.aggregation == "pooled_test")].queued_role_qualified_weed_instance_recall.iloc[0]), COLORS["masked"])]
    axes[1,1].bar(np.arange(len(model_order)), [value for _, value, _ in model_order], color=[color for _,_,color in model_order], width=.68)
    axes[1,1].set_xticks(np.arange(len(model_order)), [label for label,_,_ in model_order], rotation=22, ha="right"); axes[1,1].set_ylabel("K=20 qualified recall"); axes[1,1].set_ylim(0, .21); panel(axes[1,1], "d")
    axes[1,1].text(0, np.quantile(random_k20,.975)+.008, f"random 95% range\n{np.quantile(random_k20,.025):.3f}–{np.quantile(random_k20,.975):.3f}", ha="center", fontsize=5.7)
    for text in axes[1,1].texts:
        if text.get_text().startswith("random 95% range"):
            text.set_visible(False)
    axes[1,1].text(0, np.quantile(random_k20,.975)+.008, f"random 95% range\n{np.quantile(random_k20,.025):.3f}--{np.quantile(random_k20,.975):.3f}", ha="center", fontsize=6.0, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4}, zorder=10)
    fig.tight_layout(h_pad=1.5, w_pad=1.4)
    return export(fig, output / "Figure_4"), {"label_selection": labels, "paired": paired, "contract_slice": s, "random": random[random.k == 20], "queues": queues[(queues.k == 20) & (queues.aggregation == "pooled_test")]}


def main() -> None:
    args = parse_args(); root, output = args.project_root.resolve(), args.output_dir.resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = {"backend": "Python/matplotlib", "editable_svg_text": True, "figures": {}}
    for name, builder in (("Figure_1", lambda: workflow_figure(output)), ("Figure_2", lambda: proposal_figure(root, output)), ("Figure_3", lambda: queue_figure(root, output)), ("Figure_4", lambda: robustness_figure(root, output))):
        hashes, sources = builder(); manifest["figures"][name] = {"exports": hashes}
        if isinstance(sources, dict):
            source_files = {}
            for key, frame in sources.items():
                path = output / f"{name}_{key}_source_data.tsv"; frame.to_csv(path, sep="\t", index=False); source_files[key] = {"path": path.name, "sha256": sha256(path)}
        else:
            path = output / f"{name}_source_data.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(sources[0]), delimiter="\t", lineterminator="\n"); writer.writeheader(); writer.writerows(sources)
            source_files = {"workflow": {"path": path.name, "sha256": sha256(path)}}
        manifest["figures"][name]["source_data"] = source_files
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__": main()
