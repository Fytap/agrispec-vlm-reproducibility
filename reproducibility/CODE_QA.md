# Code and document QA

Verification completed on 2026-08-12:

- all 40 Python entry points under `reproducibility/scripts/` completed `--help` without import or module-resolution errors;
- `replay_sat_v1_1_tables.py` completed from the packaged evidence and verified every encoded numerical invariant;
- the earlier `replay_sat_revision8_tables.py` completed from the packaged evidence;
- the self-contained manuscript source compiled with pdfLaTeX/BibTeX to a 27-page manuscript, a 16-page supplement, and a one-page cover letter;
- LaTeX logs contained no undefined citation, undefined reference, or overfull-box warning;
- output directories carry per-directory SHA-256 manifests.

An earlier full local source test produced 16 passes and one provenance-only failure because an exported source copy did not contain a resolvable Git `HEAD`. That failure remains documented. The provenance helper records `UNAVAILABLE_SOURCE_ARCHIVE` when an exported archive intentionally lacks `.git` metadata and records the exact commit whenever Git metadata is present. The subsequent source test completed with 17/17 tests passing. This provenance fallback does not alter model fitting, candidate generation, policy selection, scoring, or any reported metric.
