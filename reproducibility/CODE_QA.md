# Code and document QA

Verification completed on 2026-08-13:

- the v1.3.0 standard-library verifier completed with `status: verified`;
- 216 ranker/contract rows, 66 exact primary-endpoint replay checks, 518
  operator items, 60 locked CWFID images, and five figure text-bound audits
  passed their registered invariants;
- the H200 model audit registered 11 checkpoints and zero failures among the
  66 primary replay comparisons;
- both superseded standard-metric attempts remain preserved in the v1.3.0
  failure directory;
- the self-contained source compiled with pdfLaTeX/BibTeX to a 19-page
  manuscript, a 14-page supplement, and a one-page cover letter;
- final LaTeX logs contained no undefined citation, undefined reference, or
  overfull-box warning; and
- manuscript, supplement, and cover-letter PDFs were rendered and inspected
  page by page after the final compilation.

The compact verifier checks archived tables and numerical invariants. It does
not claim to retrain every model from raw public imagery. Full reproduction
uses the recorded public-data acquisition instructions, model identifiers,
frozen configurations, training and evaluation entry points, checkpoint
hashes, and environment records from the versioned release layers.

An earlier full local source test produced 16 passes and one provenance-only
failure because an exported source copy did not contain a resolvable Git
`HEAD`. That failure remains documented. A later source test completed with
17/17 tests passing. The provenance fallback does not alter model fitting,
candidate generation, policy selection, scoring, or any reported metric.
