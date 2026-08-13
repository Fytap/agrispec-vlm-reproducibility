# SAT LaTeX submission source

This directory is the clean submission source for *Bounded Instance-Support
Evaluation for Finite Crop--Weed Review Queues*.

Compile with pdfLaTeX and BibTeX:

1. `pdflatex main && bibtex main && pdflatex main && pdflatex main`
2. `pdflatex supplement` twice
3. `pdflatex cover_letter`

The manuscript uses five vector PDF figures. The full repository also provides
editable SVG, high-resolution TIFF, PNG previews, source-data tables, and
text-bound audits. The compact Overleaf archive retains the LaTeX sources,
vector figures, bibliography, author metadata, highlights, figure captions,
and cover letter.

The public reproducibility archive is maintained at:

<https://github.com/Fytap/agrispec-vlm-reproducibility/releases/tag/v1.3.0>

The manuscript-linked v1.3.1 evidence layer is pinned at the immutable commit
<https://github.com/Fytap/agrispec-vlm-reproducibility/tree/6930f7b48995054095dbaaeebef4f155029c107a/reproducibility/v1_3_1>.
It contains the ranker/contract grid, merge-aware accounting, crossed
uncertainty, standard metrics, aggregate operator analysis, CWFID evidence,
frozen configurations, failure records, and compact checks. Public imagery and
upstream model weights remain at their original sources.

## Submission metadata

`author_metadata.tex` contains the author names, affiliations,
equal-contribution note, corresponding-author details, and email addresses
supplied by the author team. ORCID identifiers are intentionally omitted. The
package is prepared for a single-anonymized workflow with author information
visible.

The operator analysis reports aggregate technical-review outcomes only.
Operator identities and item-level responses are excluded. Timing was not
analysed.
