# Revision 8 code QA

The revision-specific training, queue-audit, replay, figure, and bundle scripts all completed `--help` without module-resolution errors.

The first full local test run produced 16 passes and one provenance-only failure because the source copy did not contain a resolvable Git `HEAD`. The failed run was retained. The provenance helper was changed to write `UNAVAILABLE_SOURCE_ARCHIVE` when an exported source archive intentionally lacks `.git` metadata; it continues to record the exact commit whenever Git metadata is present. The subsequent full test run completed with 17/17 tests passing.

The fallback does not alter model fitting, candidate generation, policy selection, scoring, or any reported metric.

An initial document-build instruction incorrectly called BibTeX on `supplement.tex`, which has no independent bibliography. The PDF itself compiled, but BibTeX correctly returned a missing-bibliography error. The README was corrected to compile the supplement with two pdfLaTeX passes only.
