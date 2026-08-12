# Support-Aware Review Queues for Crop--Weed Vision

Versioned computational archive for the manuscript **“Support-Aware Review Queues for Crop--Weed Vision: A Multitemporal Single-Field Case Study.”**

This repository records the official-spatial WeedsGalore proposal, ranking, allocation, target-semantic, and robustness analyses. The evaluation separates spatial proposal recall, role-qualified proposal recall, and queued one-to-one instance recall. Successful runs, negative results, frozen configurations, and provenance records are retained.

## Repository contents

- `reproducibility/`: configurations, analysis and replay scripts, environment records, derived evidence tables, run summaries, checksums, and preserved failure records.
- `manuscript_source/`: self-contained pdfLaTeX/BibTeX source for the manuscript and supplementary material.
- `manuscript_pdfs/`: submission PDFs.
- `figures/`: vector PDF, editable SVG, and PNG previews for Figures 1--7.
- `release_manifests/`: SHA-256 manifests for versioned release assets.

Source imagery, private records, and human-participant materials are not distributed.

## Fast replay

The compact v1.1.0 evidence replay uses only Python's standard library:

```bash
python reproducibility/scripts/replay_sat_v1_1_tables.py \
  --project-root reproducibility \
  --output-dir reproducibility/replay_check_v11
```

It verifies the 12-setting proposal grid, complete 4-by-4 quota grid, five H200 seed-specific validation selections, patch-proxy heterogeneity, and exact component-mask replay invariants. It then writes publication-facing TSV/JSON files and their SHA-256 manifest.

The earlier table replay remains available:

```bash
cd reproducibility
python scripts/replay_sat_revision8_tables.py --project-root . --output-dir replay_output_v8
```

See [`reproducibility/README.md`](reproducibility/README.md) for the full replay map, figure rebuilding, environment information, and model-acquisition boundaries.

## Data and model acquisition

SugarBeets2016 and WeedsGalore must be obtained from their official public sources under their original licences. Source imagery and model weights are not redistributed. Dataset versions, acquisition boundaries, licences, and model identifiers are recorded in:

- [`reproducibility/DATASET_PROVENANCE_LICENSES.md`](reproducibility/DATASET_PROVENANCE_LICENSES.md)
- [`reproducibility/MODEL_ACQUISITION.md`](reproducibility/MODEL_ACQUISITION.md)

## Release

Version `v1.1.0` is the manuscript-linked release:

<https://github.com/Fytap/agrispec-vlm-reproducibility/releases/tag/v1.1.0>

Release-asset checksums are provided with the release and in [`RELEASE_ASSETS_SHA256.txt`](RELEASE_ASSETS_SHA256.txt).

## Citation and scope

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The archive supports computational replay of the reported evidence tables and figures. Re-extraction and model training require the cited public datasets and public model weights. No credentials, internal addresses, private logs, source imagery, or personal data are included.
