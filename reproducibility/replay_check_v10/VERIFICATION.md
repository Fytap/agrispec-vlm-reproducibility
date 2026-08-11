# Revision-10 replay verification

The packaged command in `reproducibility/README.md` completed with exit code 0 on 2026-08-11.

- 10 TSV outputs were byte-identical to the archived `derived_tables_v10/` references.
- `summary.json` was not byte-compared because it records invocation-relative input paths; its run ID, governance class, primary policy and input file hashes were inspected separately.
- The replay uses only already-versioned prediction and matching summaries. It does not open imagery, select a model or threshold, or create a new test policy.
