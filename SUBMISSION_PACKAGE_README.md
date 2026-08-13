# SAT submission package v1.3.0

This package contains the submission-ready manuscript and the evidence needed
to inspect or reproduce its reported analyses.

## Contents

- `01_MANUSCRIPT/`: manuscript PDF, supplementary material, cover letter, and
  highlights.
- `02_OVERLEAF_SOURCE/`: clean LaTeX source and publication figures. The ZIP
  archive in the package root contains the same source tree.
- `03_FIGURES/`: publication figures in PDF, SVG, PNG, and TIFF formats,
  together with figure source data and text-bound audits.
- `04_REPRODUCIBILITY/`: the versioned reproducibility repository snapshot and
  the compact v1.3.0 verification layer.
- `MANIFEST_SHA256.tsv`: byte counts and SHA-256 digests for every packaged
  file except the manifest itself.

## Recommended checks

Compile `main.tex`, `supplement.tex`, and `cover_letter.tex` separately with
pdfLaTeX. The validated outputs contain 19, 14, and 1 pages, respectively.

From the reproducibility repository root, run:

```text
python reproducibility/v1_3_0/scripts/verify_v1_3_evidence.py
```

The verifier reconstructs archived result tables and checks numerical
invariants and file hashes. It is a table-level replay. Full retraining also
requires the cited public datasets and public initialisation weights.

## Scope

The study evaluates a finite offline crop--weed review queue. It does not claim
field deployment, production control, or a farm-population confidence bound.
Item-level operator responses, identities, credentials, internal hosts, and
private paths are excluded from the public package.
