# SAT LaTeX submission source

The directory is self-contained for pdfLaTeX/BibTeX and Overleaf.

Compile the manuscript with:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Compile the supplement and cover letter with:

```bash
pdflatex supplement
pdflatex supplement
pdflatex cover_letter
pdflatex cover_letter
```

Figures 1--7 are provided as vector PDFs in `figures/`. `author_metadata.tex` contains the author list, affiliations, equal-contribution statement, and corresponding-author details supplied by the authors. Funding and author-specific CRediT text are not enabled because factual role statements have not yet been supplied.

The manuscript-linked public archive is GitHub Release v1.1.0: <https://github.com/Fytap/agrispec-vlm-reproducibility/releases/tag/v1.1.0>.
