# 1B Results: Three 20B-Token Runs

All runs used seed 42, the FineWeb-Edu 20B-token array, a 32k LLaMA-style tokenizer, sequence length 2,048, and batch size 512. Each benchmark suite is complete and not limited. Training PPL is derived as exp(final training loss); validation PPL is reported by the evaluator. These are single-seed descriptive results.

## Training and validation

| Run | LR | Final train loss | Train PPL* | Validation NLL | Validation PPL | Mean tokens/sec |
|---|---:|---:|---:|---:|---:|---:|
| Standard 1B | 0.0012 | 2.2864 | 9.8398 | 2.2961 | 9.9352 | 219951 |
| Algebraic 1B (LR 0.0024) | 0.0024 | 2.2635 | 9.6164 | 2.2721 | 9.6994 | 208629 |
| Algebraic 1B (LR 0.0012) | 0.0012 | 2.2606 | 9.5890 | 2.2711 | 9.6897 | 208705 |

*Train PPL is calculated from the final logged training loss; validation PPL is the final held-out evaluation.

## Zero-shot accuracy

| Benchmark | Standard (LR 0.0012) | Algebraic (LR 0.0024) | Algebraic (LR 0.0012) |
|---|---:|---:|---:|
| Arc Easy | 54.71% (1300/2376) | 55.09% (1309/2376) | 54.34% (1291/2376) |
| Arc Challenge | 30.03% (352/1172) | 30.55% (358/1172) | 31.06% (364/1172) |
| Hellaswag | 44.27% (4446/10042) | 46.42% (4661/10042) | 45.66% (4585/10042) |
| Piqa | 69.97% (1286/1838) | 70.08% (1288/1838) | 70.67% (1299/1838) |
| Openbookqa | 31.60% (158/500) | 35.00% (175/500) | 33.00% (165/500) |
| Winogrande | 51.70% (655/1267) | 53.67% (680/1267) | 52.57% (666/1267) |
| Boolq | 57.37% (1876/3270) | 62.26% (2036/3270) | 54.92% (1796/3270) |
| Lambada | 29.30% (1510/5153) | 32.45% (1672/5153) | 32.35% (1667/5153) |

## Recall

Each task has 640 present-evidence examples, 640 absent-evidence examples, and 32 short controls. Exact retrieval requires the full requested code sequence. Recognition is four-choice (25% chance).

| Task | Present exact retrieval (Std / Alg 0.0024 / Alg 0.0012) | Absent unsupported hits (same order) | Short-control exact retrieval (same order) |
|---|---:|---:|---:|
| Needle Passkey | 72.03% / 77.34% / 49.38% | 0.00% / 0.00% / 0.00% | 100.00% / 100.00% / 100.00% |
| Associative Recall | 25.31% / 25.31% / 26.41% | 0.00% / 0.00% / 0.00% | 100.00% / 100.00% / 100.00% |
| Multi Needle | 0.47% / 0.47% / 0.78% | 0.00% / 0.00% / 0.00% | 9.38% / 0.00% / 0.00% |
| Latest Value | 17.03% / 17.81% / 15.78% | 0.00% / 0.00% / 0.00% | 78.12% / 50.00% / 90.62% |

Per-condition recognition and format-validity metrics, all per-depth/context cells, and prediction files are included in `all_runs_comparison.json` / `.csv` and the per-run benchmark reports.
