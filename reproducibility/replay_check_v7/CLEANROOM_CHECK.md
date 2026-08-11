# Clean-room replay check

Date: 2026-08-10

The replay ran in a clean directory containing only `replay_sat_revision7_tables.py` and the seven declared TSV inputs under their documented relative paths. It regenerated seven output TSVs plus `summary.json`. Every output SHA-256 matched `derived_tables_v7`.

Result: **pass**.
