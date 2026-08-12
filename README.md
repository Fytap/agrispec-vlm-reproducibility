# From Candidate Scores to Plant Recovery

Versioned computational archive for the manuscript **“From Candidate Scores to Plant Recovery in Finite Crop--Weed Review Queues.”**

The archive evaluates finite crop--weed review queues as a proposal-to-budget chain. It separates spatial support, role-qualified support, candidate discrimination, and deterministic one-to-one plant recovery. The current release adds review-burden matching, target-trained strong baselines, a two-operator technical review, and a separately locked CWFID transfer test.

## Repository contents

- `reproducibility/v1_2_0/`: frozen configurations, training and evaluation entry points, evidence tables, checkpoint and public-weight hashes, external-test receipts, aggregate operator results, source data, and a compact verifier.
- `reproducibility/`: earlier analysis layers retained for provenance and backward replay.
- `manuscript_source/`: self-contained pdfLaTeX/BibTeX source for the manuscript and supplementary material.
- `manuscript_pdfs/`: manuscript, supplement, and cover-letter PDFs.
- `figures/`: Figures 1--5 in vector PDF, editable SVG, high-resolution TIFF, and PNG formats, with source-data and text-bound audit files.
- `release_manifests/`: SHA-256 manifests for release assets.

Public dataset imagery and upstream model weights are not redistributed. Item-level operator responses and operator identities are not included.

## Fast verification

The v1.2.0 compact verifier uses only Python's standard library:

```bash
python reproducibility/v1_2_0/scripts/replay_sat_v1_2_evidence.py \
  --root reproducibility/v1_2_0 \
  --output replay_v1_2
```

It verifies all public checksums and registered numerical invariants for the matched-burden comparison, U-Net/DeepLabV3+/Mask R-CNN contrasts, semantic-probability ranking, locked CWFID transfer, and two-operator agreement. A successful run reports `status: verified`.

See [`reproducibility/v1_2_0/README.md`](reproducibility/v1_2_0/README.md) for the evidence map and end-to-end command entry points. Earlier replay layers remain documented in [`reproducibility/README.md`](reproducibility/README.md).

## Data and model acquisition

SugarBeets2016, WeedsGalore, and CWFID must be obtained from their official public sources under their original licences. Dataset versions, acquisition boundaries, licences, model identifiers, and public-weight hashes are recorded in the reproducibility materials.

## Release

Version `v1.2.0` is the manuscript-linked release:

<https://github.com/Fytap/agrispec-vlm-reproducibility/releases/tag/v1.2.0>

Release-asset checksums are provided with the release and in `RELEASE_ASSETS_SHA256.txt`.

## Citation and scope

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Re-extraction and model training require the cited public datasets and public model weights. The compact replay verifies archived evidence without retraining. No credentials, internal addresses, private infrastructure logs, source imagery, or personal data are included.
