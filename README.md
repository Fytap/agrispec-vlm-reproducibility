# Support-Aware Commissioning for Crop--Weed Vision

Versioned computational archive for the manuscript **“From Proposals to Review Queues: Support-Aware Commissioning for Crop--Weed Vision.”**

This repository records the official-spatial WeedsGalore proposal, ranking, allocation, and target-semantic analyses. It separates spatial proposal recall, role-qualified recall, and queued instance recall, and preserves both successful and unsuccessful registered runs.

## Repository contents

- `reproducibility/`: frozen configurations, analysis and replay scripts, environment records, derived tables, run summaries, checksums, and failure records.
- `manuscript_source/`: compilable LaTeX source for the manuscript and supplementary material.
- `manuscript_pdfs/`: final manuscript and supplementary PDF.
- `figures/`: vector PDF, editable SVG, PNG previews, source-data tables, and layout-audit records.
- `release_manifests/`: SHA-256 manifests for the complete submission and Overleaf archives.

The GitHub Release assets contain the complete submission archive, including the high-resolution TIFF figures omitted from Git history.

## Fast replay

The evidence-table replay uses only Python’s standard library:

```bash
cd reproducibility
python scripts/replay_sat_revision8_tables.py --project-root . --output-dir replay_output_v8
```

The robust allocation analysis is reproduced with:

```bash
python scripts/analyze_p2_weedsgalore_revision10_robust_queue.py \
  --config configs/p2_weedsgalore_revision10_robust_queue_v1.yaml \
  --queue-run results/p2_development/P2_WEEDSGALORE_REVISION8_QUEUE_POLICY_TILE_AUDIT_20260810_v1 \
  --semantic-run results/p2_development/P2_WEEDSGALORE_TARGET_SEMANTIC_QUEUE_BASELINE_20260810_v3 \
  --output replay_output_v10
```

See [`reproducibility/README.md`](reproducibility/README.md) for figure rebuilding, environment details, expected checksums, and model acquisition boundaries.

## Data and model acquisition

Source imagery is not redistributed. SugarBeets2016 and WeedsGalore must be obtained from their official public sources under their original licences. Public model identifiers, dataset versions, download boundaries, and licence records are documented in:

- [`reproducibility/DATASET_PROVENANCE_LICENSES.md`](reproducibility/DATASET_PROVENANCE_LICENSES.md)
- [`reproducibility/MODEL_ACQUISITION.md`](reproducibility/MODEL_ACQUISITION.md)

## Release

Version `v1.0.0` is the manuscript-linked immutable release:

<https://github.com/Fytap/agrispec-vlm-reproducibility/releases/tag/v1.0.0>

Release-asset checksums are listed in [`RELEASE_ASSETS_SHA256.txt`](RELEASE_ASSETS_SHA256.txt).

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). A repository DOI can be added to a later metadata-only release after archival deposition.

## Scope

The archive supports computational replay of the reported tables and figures. Re-extraction and model training require the cited public datasets and model weights. No credentials, internal addresses, private logs, or source imagery are included.

