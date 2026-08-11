#!/usr/bin/env python3
"""Rebuild SAT revision tables from archived raw result tables.

This analysis-only replay uses Python's standard library and never imports the
training or proposal code. It is intended for clean-room verification of the
reported manuscript numbers after the raw TSV/JSON artifacts are available.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, platform
from collections import defaultdict
from pathlib import Path


def read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as h: return list(csv.DictReader(h, delimiter="\t"))


def write_tsv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]),delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(rows)


def sha(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def mean(values):
    values=[float(v) for v in values]; return sum(values)/len(values)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--project-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    root,out=a.project_root.resolve(),a.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    contract=root/"results/p2_development/P2_WEEDSGALORE_COMMISSIONING_CONTRACT_AUDIT_20260810_v2"
    modern=root/"results/p2_development/P2_WEEDSGALORE_MODERN_FROZEN_BASELINES_20260810_v1"
    finetune=root/"results/p2_development/P2_WEEDSGALORE_DINOV2_TARGET_FINETUNE_20260810_v3"
    queue=root/"results/p2_development/P2_WEEDSGALORE_DINOV2_QUEUE_20260810_v2"
    proposal=read_tsv(contract/"proposal_recall_layers.tsv"); qrows=read_tsv(contract/"commissioning_queue_comparison.tsv")
    ranking=read_tsv(contract/"candidate_ranking_by_date.tsv"); split=read_tsv(contract/"split_merge_sensitivity.tsv")
    full=read_tsv(modern/"full_label_baselines.tsv"); budget=read_tsv(modern/"same_label_budget.tsv"); paired=read_tsv(modern/"paired_auc_differences.tsv")
    tuned=read_tsv(finetune/"target_finetune_by_date_repeat.tsv"); dqueue=read_tsv(queue/"dinov2_queue_by_date_repeat.tsv")
    dates=("2023-05-25","2023-05-30","2023-06-06","2023-06-15"); image_n={"2023-05-25":48,"2023-05-30":48,"2023-06-06":48,"2023-06-15":12}
    t1=[]
    for date in dates:
        for variant in ("original","nested_commissioned"):
            row=next(r for r in proposal if r["held_out_date"]==date and r["variant"]==variant and r["truth_role"]=="weed")
            t1.append({k:row[k] for k in ("variant","held_out_date","truth_instances","spatial_proposal_recall","role_qualified_proposal_recall","proposal_ar_iou_0.25","proposal_ar_iou_0.50","proposal_ar_iou_0.75")})
    t2=[]
    for date in dates:
        for variant in ("original","nested_commissioned"):
            row=next(r for r in qrows if r["held_out_date"]==date and r["variant"]==variant and r["model"]=="role_plus_local" and r["k"]=="20")
            t2.append({k:row[k] for k in ("variant","held_out_date","all_candidate_precision","queued_spatial_weed_instance_recall","queued_role_qualified_weed_instance_recall","pure_crop_scene_frequency","crop_overlap_scene_frequency_0.00","crop_overlap_scene_frequency_0.10")})
    t3=[]
    for date in dates:
        fr=next(r for r in full if r["held_out_date"]==date and r["feature_kind"]=="dinov2_frozen")
        tr=[r for r in tuned if r["held_out_date"]==date]
        t3.append({"held_out_date":date,"eligible_crop_candidates":next(r["eligible_crop_candidates"] for r in ranking if r["variant"]=="nested_commissioned" and r["model"]=="role_plus_local" and r["held_out_date"]==date),
                   "eligible_weed_candidates":next(r["eligible_weed_candidates"] for r in ranking if r["variant"]=="nested_commissioned" and r["model"]=="role_plus_local" and r["held_out_date"]==date),
                   "dinov2_frozen_roc_auc":fr["roc_auc"],"dinov2_frozen_average_precision":fr["average_precision"],
                   "dinov2_finetune_mean_roc_auc":mean(r["roc_auc"] for r in tr),"dinov2_finetune_mean_average_precision":mean(r["average_precision"] for r in tr)})
    annotation=[]
    for date in dates:
        fr=next(r for r in full if r["held_out_date"]==date and r["feature_kind"]=="dinov2_frozen")
        b=[r for r in budget if r["held_out_date"]==date and r["feature_kind"]=="dinov2_frozen"]
        annotation.append({"held_out_date":date,"source_dense_mask_images_global":7050,"target_training_date_dense_masks_for_setting_selection":156-image_n[date],
                           "target_held_out_dense_masks_offline_scoring_only":image_n[date],"full_training_candidate_role_labels":fr["training_labels"],
                           "sampled_training_candidate_role_labels":int(mean(r["training_labels"] for r in b)),"held_out_candidate_role_labels_offline_scoring_only":fr["test_candidates"],
                           "class_stratified_sampling_requires_roles_before_sampling":"yes"})
    split_summary=[]
    for variant in ("original","nested_commissioned"):
        for threshold in ("0.1","0.25","0.5"):
            rr=[r for r in split if r["variant"]==variant and r["instance_coverage_threshold"]==threshold]
            split_summary.append({"variant":variant,"instance_coverage_threshold":threshold,"split_instances":sum(int(r["split_instances"]) for r in rr),"merge_candidates":sum(int(r["merge_candidates"]) for r in rr)})
    out.mkdir(parents=True)
    tables={"proposal_layers":t1,"matched_k20_queue":t2,"modern_baselines_by_date":t3,"annotation_budget":annotation,"split_merge_sensitivity":split_summary,"paired_auc_differences":paired}
    for name,rows in tables.items(): write_tsv(out/(name+".tsv"),rows)
    dino_k20=[r for r in dqueue if r["k"]=="20"]
    summary={"proposal_date_macro":{v:{m:mean(r[m] for r in t1 if r["variant"]==v) for m in ("spatial_proposal_recall","role_qualified_proposal_recall","proposal_ar_iou_0.25","proposal_ar_iou_0.50","proposal_ar_iou_0.75")} for v in ("original","nested_commissioned")},
             "matched_role_plus_local_k20":{v:{m:mean(r[m] for r in t2 if r["variant"]==v) for m in ("all_candidate_precision","queued_spatial_weed_instance_recall","queued_role_qualified_weed_instance_recall","crop_overlap_scene_frequency_0.00")} for v in ("original","nested_commissioned")},
             "dinov2_finetune_date_macro_roc_auc":mean(r["roc_auc"] for r in tuned),
             "dinov2_finetune_k20":{m:mean(r[m] for r in dino_k20) for m in ("all_candidate_precision","queued_spatial_weed_instance_recall","queued_role_qualified_weed_instance_recall","crop_overlap_scene_frequency_0.00")},
             "inputs":{},"outputs":{},"python":platform.python_version()}
    inputs=[contract/"proposal_recall_layers.tsv",contract/"commissioning_queue_comparison.tsv",contract/"candidate_ranking_by_date.tsv",contract/"split_merge_sensitivity.tsv",modern/"full_label_baselines.tsv",modern/"same_label_budget.tsv",modern/"paired_auc_differences.tsv",finetune/"target_finetune_by_date_repeat.tsv",queue/"dinov2_queue_by_date_repeat.tsv"]
    summary["inputs"]={str(p.relative_to(root)).replace("\\","/"):sha(p) for p in inputs}
    summary_path=out/"summary.json"; summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    summary["outputs"]={p.name:sha(p) for p in out.glob("*.tsv")}
    summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__": main()
