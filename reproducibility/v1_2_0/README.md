# v1.2.0 evidence layer

This directory supports the manuscript *From Candidate Scores to Plant Recovery in Finite Crop--Weed Review Queues*. It adds the matched-burden commissioning analysis, target-trained strong baselines, validation-selected semantic ranking, the locked CWFID transfer test, and the aggregate two-operator review analysis.

## Evidence map

- `configs/`: frozen data, training, matching, external-transfer, and operator-analysis contracts.
- `scripts/`: the training, evaluation, aggregation, bootstrap, and figure-generation programs used for the reported analyses.
- `evidence/matched_burden/`: paired common-capacity, global equal-count, recall--burden, and image-cluster bootstrap outputs.
- `evidence/strong_baselines/`: per-seed U-Net, DeepLabV3+, and Mask R-CNN metrics, checkpoint hashes, model summaries, and paired image-cluster contrasts.
- `evidence/semantic_ranking/`: validation policy selection and fixed-test queue outputs for conditional probability, mean weed probability, and predictive entropy.
- `evidence/cwfid_external/`: freeze/open receipts, 60-image input audits, checkpoint hashes, per-image results, the preserved empty-candidate writer failure, and zero-candidate consolidation.
- `evidence/operator_review/`: aggregate categorical agreement, scene-cluster intervals, and mask-contract correspondence. Item-level responses and operator identities are not distributed.
- `source_data/`: source tables and text-bound audits for every manuscript figure and the operator-agreement table.
- `manifests/PUBLIC_SHA256SUMS.tsv`: release-file integrity manifest.

Public imagery and public initialisation weights remain at their cited upstream repositories. Re-running model fitting requires the official SugarBeets2016, WeedsGalore, and CWFID releases and the public weights identified by the frozen configurations. Checkpoints are identified by SHA-256 in the evidence tables; they are not redistributed in this compact GitHub archive.

## Compact verification

The compact verifier uses only the Python standard library:

```bash
python reproducibility/v1_2_0/scripts/replay_sat_v1_2_evidence.py \
  --root reproducibility/v1_2_0 \
  --output replay_v1_2
```

It verifies the public manifest, reloads the five evidence summaries, checks the registered denominators and primary endpoints, and writes `verification.json` plus `key_results.tsv`. A non-zero exit status indicates a missing file, checksum mismatch, or violated numerical invariant.

## End-to-end entry points

After public data and weights are acquired, the principal commands are:

```bash
python reproducibility/v1_2_0/scripts/train_evaluate_p2_weedsgalore_target_semantic_queue_baseline.py --help
python reproducibility/v1_2_0/scripts/train_evaluate_p2_weedsgalore_deeplabv3plus_h200.py --help
python reproducibility/v1_2_0/scripts/train_evaluate_p2_weedsgalore_maskrcnn_h200.py --help
python reproducibility/v1_2_0/scripts/evaluate_p2_cwfid_external_locked_h200.py --help
```

The full command lines, random seeds, model-selection rules, role-qualified matching thresholds, fixed queue size, and hardware/software versions are encoded in `configs/`. The external CWFID configuration forbids external fitting, calibration, threshold selection, and allocation search.

## Interpretation boundary

Candidate AUC and AP are diagnostic. The primary system endpoint is maximum-cardinality one-to-one role-qualified weed-instance recovery at fixed review capacity. WeedsGalore results are internal to its official split; the separately locked CWFID result evaluates transfer and produced zero candidates at every frozen operating point.
