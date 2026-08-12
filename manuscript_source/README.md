# SAT LaTeX submission package

This directory is the clean submission source for *From Candidate Scores to Plant Recovery in Finite Crop--Weed Review Queues*.

Compile with pdfLaTeX and BibTeX:

1. `pdflatex main && bibtex main && pdflatex main && pdflatex main`
2. `pdflatex supplement` twice
3. `pdflatex cover_letter` twice

The manuscript cites five vector PDF figures. Editable SVG, high-resolution TIFF, PNG previews, source-data tables, text-bound audits, and SHA-256 manifests are included in the full submission bundle. The compact Overleaf archive retains the LaTeX sources, vector figures, bibliography, author metadata, highlights, and cover letter.

The public reproducibility archive is maintained at:

<https://github.com/Fytap/agrispec-vlm-reproducibility/releases/tag/v1.2.0>

Release v1.2.0 contains the matched-burden analysis, strong-baseline runs, semantic-ranking diagnostics, locked CWFID protocol and outputs, aggregate two-operator analysis, frozen configurations, public-weight and checkpoint hashes, preserved failures, and clean-room replay checks. Public dataset imagery and upstream model weights remain at their original sources.

## Submission metadata

`author_metadata.tex` contains the author names, affiliations, equal-contribution note, corresponding-author details, and email addresses supplied by the author team. ORCID identifiers are intentionally omitted. The package is prepared for SAT's single-anonymized workflow with author information visible.

The operator study reports aggregate technical-review outcomes only. Operator identities and item-level responses are not included in the public submission package. Timing was not analysed. The ethics declaration states the scope of this annotation and software-quality activity.

Funding and author-specific CRediT statements remain disabled until factual role information is supplied by the author team; no empty declaration headings are rendered.
