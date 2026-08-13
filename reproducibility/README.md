# Reproducibility archive

The current manuscript-linked evidence layer is [`v1_2_0/`](v1_2_0/). It covers matched review burden, target-trained strong baselines, semantic ranking, a locked CWFID transfer test, and aggregate two-operator technical review. It does not redistribute dataset imagery, masks, public model weights, operator identities, item-level responses, or private infrastructure records.

## Current verification

From the repository root, run:

```text
python reproducibility/v1_2_0/scripts/replay_sat_v1_2_evidence.py --root reproducibility/v1_2_0 --output replay_v1_2
```

The command verifies the v1.2.x public manifest and registered numerical invariants. Text line endings are canonicalised for cross-platform source archives; binary artifacts remain byte-exact. See [`v1_2_0/README.md`](v1_2_0/README.md) for the current evidence map and end-to-end entry points.

## v1.1.0 compact replay

From the repository root, run:

```text
python reproducibility/scripts/replay_sat_v1_1_tables.py --project-root reproducibility --output-dir reproducibility/replay_check_v11
```

The command uses only Python's standard library. It verifies:

- all 12 proposal settings in the training-date selection grid;
- all 16 combinations of minimum and maximum per-tile quota;
- the date-balanced comparator and five seed-specific validation selections;
- the sign reversal of the date-balanced recall effect across the two test patch proxies;
- exact component-mask candidate identity, scene-metric identity, and match-edge counts.

The command writes compact TSV/JSON tables and a SHA-256 manifest. It does not retrain a model, alter a policy, or read source imagery.

## Earlier table replay

The previous standard-library replay is retained for continuity:

```text
cd reproducibility
python scripts/replay_sat_revision8_tables.py --project-root . --output-dir replay_output_v8
```

It reconstructs the earlier core summaries and queue-policy ratios from archived integer numerators and denominators.

## Robust-allocation analysis

The full candidate-level robust-allocation analysis can be rerun from archived candidate scores and matching summaries:

```text
python scripts/analyze_p2_weedsgalore_revision10_robust_queue.py --config configs/p2_weedsgalore_revision10_robust_queue_v1.yaml --queue-run results/p2_development/P2_WEEDSGALORE_REVISION8_QUEUE_POLICY_TILE_AUDIT_20260810_v1 --semantic-run results/p2_development/P2_WEEDSGALORE_TARGET_SEMANTIC_QUEUE_BASELINE_20260810_v3 --output replay_output_v10
```

The exact quota and patch analyses are implemented in `analyze_target_semantic_quota_robustness_exact_v3.py` and `analyze_quota_patch_effects.py`. Their immutable derived outputs are stored under `results/p2_development/` with per-directory manifests.

## Figure rebuild

Figures are Python-generated. The current generators are:

- `build_sat_revision14_figure1_nature.py` for the workflow diagram;
- `build_sat_revision14_figures.py` for the metric panels;
- `build_sat_revision17_quota_figure.py` for the quota and patch analysis;
- `build_sat_revision15_seed_stability_figure.py` for H200 seed stability;
- `build_sat_submission_figure5.py` for the three-panel CWFID transfer result.

Each generator accepts `--help`; its output is deterministic conditional on the archived inputs and installed plotting stack. Submission-ready exports are included in the repository-level `figures/` directory.

## Contents and boundaries

- `configs/`: versioned YAML configurations.
- `scripts/`: proposal, queue, baseline, sensitivity, figure, and replay code.
- `results/p2_development/`: evidence-bearing TSV/JSON outputs and preserved diagnostics.
- `environment/requirements-lock.txt`: environment versions recorded by completed runs.
- `DATASET_PROVENANCE_LICENSES.md`: dataset acquisition and licensing records.
- `MODEL_ACQUISITION.md`: public model identifiers and acquisition boundaries.
- `CODE_QA.md`: command-entry-point and replay verification.

The chronology table in the manuscript distinguishes the frozen core analysis from later exploratory sensitivity analyses. The archive contains no claim of immutable preregistration. Re-extraction and training require the cited public datasets and public model weights under their original licences.
