#!/usr/bin/env python3
"""Apply saved DINOv2 fine-tuning scores to the full fixed-budget queue."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_p2_weedsgalore_commissioning_contract_audit import DATES, KS, audit_setting, load_truth, queue, write_tsv  # noqa: E402
from evaluate_p2_weedsgalore_target_recalibration_nested_lodo import read_tsv, sha256  # noqa: E402


def main() -> None:
    p=argparse.ArgumentParser()
    for n in ("project-root","archive","manifest","grid-root","commissioning-summary","predictions","config","output-dir"):
        p.add_argument(f"--{n}",type=Path,required=True)
    p.add_argument("--run-id",required=True); p.add_argument("--expected-predictions-sha256",required=True)
    a=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES",""): raise RuntimeError("CPU-only")
    root,out=a.project_root.resolve(),a.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    if sha256(a.predictions.resolve()) != a.expected_predictions_sha256: raise ValueError("Prediction hash mismatch")
    started=time.perf_counter(); manifest=read_tsv(a.manifest.resolve()); truth=load_truth(a.archive.resolve(),manifest)
    commission=json.loads(a.commissioning_summary.read_text(encoding="utf-8")); selected={d:commission["per_outer_date"][d]["setting"] for d in DATES}
    pred=read_tsv(a.predictions.resolve()); cache={}; rows=[]
    for held in DATES:
        setting=selected[held]
        if setting not in cache: cache[setting]=audit_setting(a.grid_root.resolve()/setting,manifest,truth)
        audit=cache[setting]
        for repeat in range(3):
            score={r["candidate_id"]:float(r["score"]) for r in pred if r["held_out_date"]==held and int(r["repeat"])==repeat}
            expected={r["candidate_id"] for r in audit["rows"] if r["date"]==held}
            if set(score)!=expected: raise ValueError({"held":held,"repeat":repeat,"scores":len(score),"expected":len(expected)})
            for k in KS: rows.append({"repeat":repeat,**queue("nested_commissioned","dinov2_last_block_finetune",held,k,audit,score)})
    out.mkdir(parents=True); table=out/"dinov2_queue_by_date_repeat.tsv"; write_tsv(table,rows)
    k20=[r for r in rows if int(r["k"])==20]
    metrics=("all_candidate_precision","queued_spatial_weed_instance_recall","queued_role_qualified_weed_instance_recall","pure_crop_scene_frequency","crop_overlap_scene_frequency_0.00")
    by_repeat={str(rep):{m:float(np.mean([float(r[m]) for r in k20 if int(r["repeat"])==rep])) for m in metrics} for rep in range(3)}
    summary={"run_id":a.run_id,"status":"completed_dinov2_fixed_budget_queue","scope":{"public_data_only":True,"sealed_test_read":False,"formal_risk_certificate":False},
             "k20_date_macro_by_repeat":by_repeat,"k20_mean_across_repeats":{m:float(np.mean([v[m] for v in by_repeat.values()])) for m in metrics},
             "inputs":{"predictions_sha256":sha256(a.predictions.resolve()),"archive_sha256":sha256(a.archive.resolve()),"manifest_sha256":sha256(a.manifest.resolve())},
             "outputs":{"queue_tsv_sha256":sha256(table)},"runtime":{"wall_seconds":time.perf_counter()-started,"python":sys.version,"numpy":np.__version__},
             "script_sha256":sha256(Path(__file__).resolve()),"config_sha256":sha256(a.config.resolve()),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__": main()
