# Revision 8 clean replay check

Date: 2026-08-10

The revision-8 replay was run once into the archived `replay_check_v8` directory and independently again into a new empty temporary directory. The two runs produced identical SHA-256 values for all five revision-8 outputs and for the nested revision-7 core summary.

The replay recomputes queue-policy ratios from archived integer numerators and denominators, recomputes paired tile deltas from the 26 per-tile rows, recomputes target-semantic pooled and per-date queue metrics from per-tile integer counts, and verifies them against the completed run summary/date tables. It does not read images, masks, embeddings, checkpoints, credentials, or test-label files.

Result: pass.
