# Post-training comparison

Training seeds: 42, 43, 44.

Scores are percentages, mean ± sample SD across training seeds. Paired differences use the same seed and examples. No pooled item confidence interval is implied.

## Held-out language modeling and training telemetry

NLL is in nats/token; perplexity is exp(NLL). Training times and throughput are trainer telemetry.

| metric | standard | algebraic | hybrid |
|---|---|---|---|
| final_eval_nll | 3.17 ± 0.00 | 3.13 ± 0.00 | 3.13 ± 0.00 |
| final_eval_ppl | 23.86 ± 0.06 | 22.78 ± 0.06 | 22.93 ± 0.03 |
| train_seconds | 1555.39 ± 0.45 | 1813.70 ± 0.92 | 1761.28 ± 0.48 |
| mean_tok_per_s | 1622322.77 ± 346.58 | 1390541.99 ± 355.30 | 1431811.09 ± 155.47 |

## Public tasks

| task / condition / metric | standard | algebraic | hybrid |
|---|---|---|---|
| arc_challenge | 24.77 ± 0.26 | 25.11 ± 0.50 | 24.83 ± 0.39 |
| arc_easy | 43.71 ± 0.60 | 43.69 ± 0.43 | 44.74 ± 0.95 |
| boolq | 58.86 ± 3.70 | 58.80 ± 2.32 | 58.03 ± 3.81 |
| hellaswag | 28.60 ± 0.19 | 29.32 ± 0.18 | 29.12 ± 0.16 |
| lambada | 17.81 ± 0.24 | 17.41 ± 0.63 | 17.24 ± 0.60 |
| openbookqa | 27.93 ± 1.51 | 28.00 ± 0.35 | 27.87 ± 0.76 |
| piqa | 61.68 ± 0.31 | 62.44 ± 0.69 | 62.08 ± 0.86 |
| winogrande | 52.59 ± 0.20 | 51.64 ± 0.53 | 51.62 ± 0.65 |

## Recall with evidence

| task / condition / metric | standard | algebraic | hybrid |
|---|---|---|---|
| associative_recall / present / exact_retrieval | 1.82 ± 1.73 | 3.07 ± 1.35 | 0.47 ± 0.16 |
| latest_value / present / exact_retrieval | 2.03 ± 1.02 | 3.49 ± 2.42 | 2.19 ± 1.08 |
| multi_needle / present / exact_retrieval | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| needle_passkey / present / exact_retrieval | 6.67 ± 2.63 | 3.33 ± 2.50 | 3.85 ± 1.30 |

## Recall controls and diagnostics

| task / condition / metric | standard | algebraic | hybrid |
|---|---|---|---|
| associative_recall / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| associative_recall / absent / recognition_accuracy | 23.54 ± 4.15 | 23.75 ± 4.06 | 22.86 ± 2.11 |
| associative_recall / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| associative_recall / present / needle_recall | 1.82 ± 1.73 | 3.07 ± 1.35 | 0.47 ± 0.16 |
| associative_recall / present / recognition_accuracy | 81.04 ± 3.55 | 84.01 ± 1.66 | 82.71 ± 0.70 |
| associative_recall / short / exact_retrieval | 17.71 ± 23.03 | 0.00 ± 0.00 | 2.08 ± 1.80 |
| associative_recall / short / needle_recall | 17.71 ± 23.03 | 0.00 ± 0.00 | 2.08 ± 1.80 |
| associative_recall / short / recognition_accuracy | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| latest_value / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| latest_value / absent / recognition_accuracy | 25.68 ± 1.93 | 26.72 ± 6.10 | 27.55 ± 2.22 |
| latest_value / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| latest_value / present / needle_recall | 2.03 ± 1.02 | 3.49 ± 2.42 | 2.19 ± 1.08 |
| latest_value / present / recognition_accuracy | 48.12 ± 2.78 | 48.18 ± 6.41 | 47.66 ± 5.07 |
| latest_value / short / exact_retrieval | 7.29 ± 7.86 | 12.50 ± 10.83 | 8.33 ± 1.80 |
| latest_value / short / needle_recall | 7.29 ± 7.86 | 12.50 ± 10.83 | 8.33 ± 1.80 |
| latest_value / short / recognition_accuracy | 76.04 ± 10.05 | 69.79 ± 14.77 | 63.54 ± 19.09 |
| multi_needle / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| multi_needle / absent / recognition_accuracy | 21.04 ± 1.86 | 22.81 ± 4.84 | 13.18 ± 4.39 |
| multi_needle / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| multi_needle / present / needle_recall | 1.34 ± 1.01 | 2.47 ± 2.82 | 1.30 ± 1.08 |
| multi_needle / present / recognition_accuracy | 93.12 ± 4.71 | 99.84 ± 0.16 | 97.29 ± 2.98 |
| multi_needle / short / exact_retrieval | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| multi_needle / short / needle_recall | 4.17 ± 4.54 | 2.43 ± 2.62 | 9.03 ± 4.34 |
| multi_needle / short / recognition_accuracy | 98.96 ± 1.80 | 100.00 ± 0.00 | 97.92 ± 1.80 |
| needle_passkey / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| needle_passkey / absent / recognition_accuracy | 12.29 ± 3.06 | 15.00 ± 1.99 | 15.00 ± 1.51 |
| needle_passkey / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| needle_passkey / present / needle_recall | 6.67 ± 2.63 | 3.33 ± 2.50 | 3.85 ± 1.30 |
| needle_passkey / present / recognition_accuracy | 96.15 ± 0.09 | 96.93 ± 0.55 | 94.64 ± 2.80 |
| needle_passkey / short / exact_retrieval | 16.67 ± 23.45 | 1.04 ± 1.80 | 4.17 ± 3.61 |
| needle_passkey / short / needle_recall | 16.67 ± 23.45 | 1.04 ± 1.80 | 4.17 ± 3.61 |
| needle_passkey / short / recognition_accuracy | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

## Paired difference from standard

Percentage points, mean ± sample SD across matched training seeds.

| task | algebraic | hybrid |
|---|---|---|
| associative_recall | 1.25 ± 0.56 | -1.35 ± 1.67 |
| latest_value | 1.46 ± 3.17 | 0.16 ± 1.99 |
| multi_needle | 0.00 ± 0.00 | 0.00 ± 0.00 |
| needle_passkey | -3.33 ± 2.22 | -2.81 ± 1.34 |
| arc_challenge | 0.34 ± 0.76 | 0.06 ± 0.44 |
| arc_easy | -0.03 ± 0.26 | 1.02 ± 1.08 |
| boolq | -0.06 ± 3.73 | -0.83 ± 0.19 |
| hellaswag | 0.71 ± 0.25 | 0.51 ± 0.24 |
| lambada | -0.40 ± 0.63 | -0.58 ± 0.64 |
| openbookqa | 0.07 ± 1.17 | -0.07 ± 2.00 |
| piqa | 0.76 ± 1.00 | 0.40 ± 0.78 |
| winogrande | -0.95 ± 0.34 | -0.97 ± 0.55 |

The longest recall budget includes answer space: 2,032 prompt tokens + 16 generated tokens.

Absent-evidence target hits are unsupported guesses, not successful retrieval. Recognition has four choices (25% chance); short controls check whether the answer format is usable.
