# Revision-10 reproducibility archive

This archive supports the official-spatial WeedsGalore analysis reported in revision 10. It does not redistribute dataset imagery, masks, DINOv2 weights, or training checkpoints.

## Fast evidence-table replay

The replay uses only Python's standard library:

```text
python scripts/replay_sat_revision8_tables.py --project-root . --output-dir replay_output_v8
```

The command first rebuilds the seven revision-7 core summaries. It then recomputes the revision-8 queue-policy ratios from their archived integer numerators and denominators, verifies crop-exposure rates, and extracts the completed equal-target-supervision semantic comparison and date rows from the registered run summary. `replay_check_v8/summary.json` records every input and output SHA-256. The replay uses only Python's standard library.

## Revision-10 robust allocation replay

The registered post-test robustness audit is reproduced with:

```text
python scripts/analyze_p2_weedsgalore_revision10_robust_queue.py --config configs/p2_weedsgalore_revision10_robust_queue_v1.yaml --queue-run results/p2_development/P2_WEEDSGALORE_REVISION8_QUEUE_POLICY_TILE_AUDIT_20260810_v1 --semantic-run results/p2_development/P2_WEEDSGALORE_TARGET_SEMANTIC_QUEUE_BASELINE_20260810_v3 --output replay_output_v10
```

This command consumes already-versioned candidate scores and matching summaries. It recomputes pooled/date/patch summaries, allocation starvation and concentration, candidate AUC/AP by date and tile, and single-run target-semantic date robustness. It does not open new test data or select a new model, threshold, proposal setting, or ranker. The archived reference outputs are in `derived_tables_v10/`.

## Collision-audited figure rebuild

From this `reproducibility/` directory, rebuild all five figures with:

```text
python scripts/build_sat_revision11_figures.py --project-root . --allocation-audit-dir results/p2_development/P2_WEEDSGALORE_REVISION8_QUEUE_POLICY_TILE_AUDIT_20260810_v1 --output-dir figure_rebuild
```

The command exports PDF, editable SVG, 300-dpi PNG and 600-dpi TIFF files, writes the plotted source tables and records renderer-level text-collision and canvas-boundary audits. It does not change any registered metric or model output.

## Full experiment material

- `configs/`: frozen YAML configurations used by the packaged replay.
- `scripts/`: proposal, queue, oracle/control, DINOv2, sensitivity, figure, and replay scripts.
- `results/p2_development/`: evidence-bearing TSV/JSON outputs and preserved historical diagnostics.
- `environment/requirements-lock.txt`: environment snapshot; run summaries additionally record exact remote Python/PyTorch/timm/NumPy versions.
- `MODEL_ACQUISITION.md`: public model identifier and acquisition boundary.
- `CODE_QA.md`: command-entry-point and full-test verification, including the preserved source-archive provenance failure and corrected replay behavior.

Every principal run summary records the git commit, configuration SHA-256, input/output hashes, seeds where applicable, software, and hardware. The failed target-semantic v2 run and corrected v3 run are both retained; failed versions are not deleted or overwritten. The public package contains a sanitized v2 failure record and the SHA-256 of the verbatim private log so machine identifiers are not redistributed.

## Scope

The replay verifies manuscript computations from archived result tables. Re-extraction and training from public imagery require downloading the cited datasets and public model weights under their original licenses. No credential, remote target, internal address, or personal data is included.
