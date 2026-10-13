# Execution verification

These records verify the implementation using synthetic training tokens and limited benchmark examples. Full FineWeb-Edu pretraining and the budgeted pilots were not launched. Synthetic losses are not quality results.

| Evidence | Scope |
|---|---|
| `cpu_tests.txt` | 24 passing component, configuration, corpus-integrity, sweep-selection, and checkpoint/resume tests |
| `tpu_kernel_tests.txt` | 90 passing Pallas attention, head, and QK checks on a TPU v4 chip |
| `tpu_head_width_tests.txt` | Four passing dense-reference gradient checks for the production fused-head tiles at widths 1,024 and 1,536 |
| `smoke_350m.json`, `smoke_1b.json` | One synthetic AdamW step per full model, fused attention/head, held-out scoring, weight export, and checkpoint reload |
| `distributed_350m.json`, `distributed_1b.json` | One optimizer step per full model across four hosts / 16 chips, production global batch 512 and accumulation settings |
| `primitive_audit_350m.json`, `primitive_audit_1b.json` | Complete model-gradient traces at the production dimensions, context, and micro-batch; no forbidden primitive, including nested Pallas bodies |
| `benchmark_smoke.json` | Both architectures pass all eight public-task routes and four recall routes with present/absent/short controls and no evidence truncation |

The reports retain their measured source fingerprints. Primitive-audit files include the full configuration and kernel/model hashes. Benchmark smoke files are explicitly limited and cannot be reused as complete comparison outputs. The CPU suite verifies that resume reproduces final weights and rejects recipe changes; selector checks require all eight pilot records before generating hyperparameters.

Plan-only launchers were checked for both scales. Main execution refuses to start without a matching completed sweep selection. Single-seed tables were verified to report individual scores and paired differences with null seed SD. Package installation and wheel construction passed. Checksums confirm that every published 125M prediction file and all its numerical measurements are preserved.
