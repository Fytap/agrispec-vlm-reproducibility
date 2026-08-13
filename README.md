# Bounded Instance-Support Evaluation

Versioned computational archive for the manuscript **Bounded Instance-Support
Evaluation for Finite Crop--Weed Review Queues**.

The study treats a finite crop--weed review queue as a proposal-to-budget
system. It separates spatial support, role qualification, candidate ranking,
merge capacity, one-to-one instance support, and review truncation. The v1.3.1
release retains the v1.3.0 analyses and corrects the human--mask binary
estimand to the determinate two-operator consensus subset. It also defines set
coverage and one-to-one support as threshold-conditioned graph endpoints,
rather than bounds on human recovery. The archive includes ranker-source
sensitivity, a 27-contract matching grid,
merge-aware accounting, crossed seed-by-image uncertainty, standard instance
metrics, extended aggregate operator agreement, and continuous diagnostics for
the unchanged locked CWFID result.

## Repository contents

- `reproducibility/v1_3_1/`: manuscript-linked v1.3.1 corrected evidence,
  analysis scripts, figure source data, and a compact verifier.
- `reproducibility/v1_3_0/`: immutable historical v1.3.0 evidence, analysis
  scripts, preserved failures, figure source data, and a compact verifier.
- `reproducibility/v1_2_0/`: frozen training/evaluation configurations,
  checkpoint and public-weight records, matched-burden evidence, and the
  previous compact replay layer.
- `reproducibility/`: earlier append-only analysis layers retained for
  provenance.
- `manuscript_source/`: self-contained pdfLaTeX/BibTeX source for the
  manuscript, supplement, and cover letter.
- `manuscript_pdfs/`: compiled manuscript, supplement, and cover-letter PDFs.
- `figures/`: Figures 1--5 in PDF, SVG, TIFF, and PNG formats.
- `release_manifests/`: release-asset SHA-256 records.

Public dataset imagery and upstream model weights are not redistributed.
Item-level operator responses and operator identities are not included.

## Fast verification

The v1.3.1 verifier uses only Python's standard library:

```bash
python reproducibility/v1_3_1/scripts/verify_v1_3_evidence.py
```

It checks 216 ranker/contract rows, 66 exact primary-endpoint replay checks,
the merge-aware and operator aggregates, the locked 60-image CWFID outcome,
and five figure text-bound audits. A successful run reports
`status: verified`.

See [`reproducibility/v1_3_1/README.md`](reproducibility/v1_3_1/README.md) for
the evidence map and replay boundary. Earlier layers remain documented in
[`reproducibility/README.md`](reproducibility/README.md).

## Reproduction boundary

The compact verifier replays archived numerical evidence without retraining.
Re-extraction and full model training require the cited public SugarBeets2016,
WeedsGalore, and CWFID data and the registered public model weights. Dataset
versions, access boundaries, licences, model identifiers, and hashes are
recorded in the archive.

## Release and citation

Version `v1.3.1` is the corrected manuscript-linked release:

<https://github.com/Fytap/agrispec-vlm-reproducibility/releases/tag/v1.3.1>

The corrected evidence layer used by the manuscript has an immutable commit
link:

<https://github.com/Fytap/agrispec-vlm-reproducibility/tree/6930f7b48995054095dbaaeebef4f155029c107a/reproducibility/v1_3_1>

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). No
credentials, internal addresses, private infrastructure logs, source imagery,
or personal data are included.
