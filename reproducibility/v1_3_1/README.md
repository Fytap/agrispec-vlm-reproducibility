# v1.3.1 corrected evidence layer

This directory contains the manuscript-linked evidence added for **Bounded
Instance-Support Evaluation for Finite Crop--Weed Review Queues**. It is an
append-only correction layer over the earlier reproducibility releases. It
defines human--mask binary correspondence on the determinate two-operator
consensus subset while retaining all unresolved items in the complete
three-by-two contingency table. It also labels one-to-one support and set
coverage as threshold-conditioned geometric endpoints rather than human-
recovery bounds.

## What can be replayed here

The compact verifier checks the archived tables and registered invariants with
Python's standard library. It covers:

- four ranker-source schemes over 27 matching contracts;
- ranker-free proposal ceilings and per-tile queue oracles;
- merge-aware set coverage and one-to-one matching;
- seed-by-image and date-block uncertainty for three model families;
- COCO-style AP/AR and a clearly labelled queue PQ-style diagnostic;
- scene-clustered operator agreement, ordered confidence agreement, and the
  aggregate human--mask contingency;
- the locked CWFID zero-candidate endpoint and post-lock continuous-score
  diagnostics; and
- figure source data and text-bound checks.

Run from the repository root:

```bash
python reproducibility/v1_3_1/scripts/verify_v1_3_evidence.py
```

A successful run prints `status: verified` and writes no files. The verifier
also rejects machine-specific paths and infrastructure identifiers in this
release layer.

## Evidence map

| Directory | Contents |
|---|---|
| `configs/` | Frozen external and strong-baseline configurations |
| `evidence/commissioning_contract/` | Four ranker schemes, 27-contract grid, proposal ceilings, and queue oracles |
| `evidence/crossed_uncertainty/` | Seed-by-image estimates, model contrasts, and date-block sensitivity |
| `evidence/merge_aware/` | Set coverage, maximum matching, merge capacity, and multiplicity summaries |
| `evidence/model_standard_metrics/` | AP/AR, queue diagnostics, 27-contract model rankings, and exact primary replay checks |
| `evidence/operator_extended/` | Aggregate agreement, ordered confidence, contingencies, and human--mask correspondence |
| `evidence/cwfid_continuous/` | Locked-checkpoint audit and continuous post-lock diagnostics |
| `failures/` | Preserved logs from two superseded standard-metric attempts |
| `scripts/` | Analysis, plotting, and compact verification source |
| `source_data/` | Figure source tables, text-bound audits, and figure manifest |

## Reproduction boundary

The compact verifier replays the published numerical evidence without public
imagery, checkpoints, or retraining. Full re-extraction and model training use
the entry points in earlier release layers together with the public datasets
and public model weights recorded there. Dataset imagery, upstream weights,
item-level operator responses, operator identities, credentials, and private
infrastructure information are not redistributed.

All entries in this layer are post-test diagnostics unless an upstream frozen
record identifies them otherwise. The locked CWFID endpoint is unchanged by
the continuous-score analysis.
