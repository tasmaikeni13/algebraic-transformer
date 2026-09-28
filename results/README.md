# Recorded measurements

Each directory groups measurements by the component or experiment it covers.
The pretraining directory contains six run records, one for each architecture
and seed, plus an aggregate ledger. The plotting script reads that ledger to
produce the figures in `paper/figures/`.
Incomplete pilot measurements are retained in `pilot/history/` and are
excluded from the manuscript's complete-run statistics.

[`record_origins.json`](record_origins.json) gives the Git commit and SHA-256
digest of each original record. The public directory and field labels are
descriptive aliases. Numerical measurements are unchanged; source hashes in
the records describe the code at execution time and do not certify the
current release layout. For exact original metadata, inspect the Git commit
and digest named in the record. Run
`python scripts/verify_recorded_results.py` to check the origin digest and
numerical contents of each retained JSON file.

CPU tests exercise the current source. Repeating the measured hardware runs
requires the datasets and a 16-chip TPU v4 slice. A local smoke run cannot
replace a hardware measurement.
