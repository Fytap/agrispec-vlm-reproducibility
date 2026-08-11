# SAT Overleaf submission package

Compile with pdfLaTeX/BibTeX:

1. `pdflatex main && bibtex main && pdflatex main && pdflatex main`
2. `pdflatex supplement` twice (the supplement has no independent bibliography)
3. `pdflatex response_to_reviewer` twice
4. `pdflatex cover_letter` twice

The principal figures are in `figures/` as vector PDF. The complete submission bundle additionally provides editable SVG, 300 dpi PNG previews, 600 dpi TIFF, source-data tables, text-layout audits and SHA-256 hashes. Figure 1 is a Python-generated, Nature-style information architecture with editable SVG text. The full submission bundle also carries the separate reproducibility directory with scripts, original module paths, frozen configurations, evidence tables, environment information, failure records, and replay checks. The compact Overleaf source intentionally excludes large replay inputs.

The versioned public archive is GitHub Release v1.0.0: <https://github.com/Fytap/agrispec-vlm-reproducibility/releases/tag/v1.0.0>.

The manuscript presents a support-aware commissioning framework for crop--weed review queues on the official spatial split from one field and two test patches. It separates the prespecified core analysis from a secondary exploratory analysis and reports pooled, date-macro, worst-date, tile and patch-proxy results. Bounded allocation of 5--50 candidates/tile is the primary batch policy; unconstrained global top-520 is a diagnostic upper bound.

## Submission metadata

`author_metadata.tex` contains the author names, affiliation mapping, equal-contribution note, corresponding-author details and email addresses supplied on 2026-08-11. The manuscript and supplement compile as single-anonymized-review submission files with author information visible. Funding, competing-interest and author-specific CRediT statements remain disabled until they are separately confirmed by the author team; no empty declaration headings are rendered.
