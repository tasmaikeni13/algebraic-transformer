# Post-training comparison

Training seeds: 42, 43, 44.

Scores are percentages, mean ± sample SD across training seeds. Paired differences use the same seed and examples. No pooled item confidence interval is implied.

## Held-out language modeling and training telemetry

NLL is in nats/token; perplexity is exp(NLL). Training times and throughput are trainer telemetry.

| metric | standard | algebraic |
|---|---|---|
| final_eval_nll | 2.58 ± 0.00 | 2.55 ± 0.00 |
| final_eval_ppl | 13.26 ± 0.02 | 12.74 ± 0.02 |
| train_seconds | 11950.36 ± 2.57 | 13270.12 ± 0.98 |
| mean_tok_per_s | 586277.11 ± 124.03 | 528221.89 ± 32.10 |

## Public tasks

| task / condition / metric | standard | algebraic |
|---|---|---|
| arc_challenge | 25.43 ± 0.52 | 25.80 ± 1.03 |
| arc_easy | 44.40 ± 1.31 | 45.74 ± 0.42 |
| boolq | 58.36 ± 2.39 | 60.35 ± 1.40 |
| hellaswag | 34.11 ± 0.16 | 35.05 ± 0.12 |
| lambada | 21.89 ± 0.66 | 23.16 ± 0.89 |
| openbookqa | 30.33 ± 0.23 | 29.87 ± 1.01 |
| piqa | 65.18 ± 0.09 | 65.71 ± 0.28 |
| winogrande | 51.75 ± 0.48 | 51.17 ± 0.43 |

## Recall with evidence

| task / condition / metric | standard | algebraic |
|---|---|---|
| associative_recall / present / exact_retrieval | 2.45 ± 0.80 | 3.02 ± 0.72 |
| latest_value / present / exact_retrieval | 8.23 ± 2.19 | 10.57 ± 1.00 |
| multi_needle / present / exact_retrieval | 0.00 ± 0.00 | 0.10 ± 0.09 |
| needle_passkey / present / exact_retrieval | 41.30 ± 27.58 | 45.36 ± 4.77 |

## Recall controls and diagnostics

| task / condition / metric | standard | algebraic |
|---|---|---|
| associative_recall / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 |
| associative_recall / absent / recognition_accuracy | 24.06 ± 2.15 | 23.65 ± 1.10 |
| associative_recall / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 |
| associative_recall / present / needle_recall | 2.45 ± 0.80 | 3.02 ± 0.72 |
| associative_recall / present / recognition_accuracy | 97.03 ± 0.41 | 98.18 ± 0.92 |
| associative_recall / short / exact_retrieval | 60.42 ± 47.01 | 78.12 ± 37.89 |
| associative_recall / short / needle_recall | 60.42 ± 47.01 | 78.12 ± 37.89 |
| associative_recall / short / recognition_accuracy | 100.00 ± 0.00 | 100.00 ± 0.00 |
| latest_value / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 |
| latest_value / absent / recognition_accuracy | 26.35 ± 8.41 | 23.49 ± 4.37 |
| latest_value / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 |
| latest_value / present / needle_recall | 8.23 ± 2.19 | 10.57 ± 1.00 |
| latest_value / present / recognition_accuracy | 60.16 ± 3.79 | 57.08 ± 4.52 |
| latest_value / short / exact_retrieval | 28.12 ± 29.81 | 38.54 ± 25.26 |
| latest_value / short / needle_recall | 28.12 ± 29.81 | 38.54 ± 25.26 |
| latest_value / short / recognition_accuracy | 86.46 ± 9.55 | 81.25 ± 16.54 |
| multi_needle / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 |
| multi_needle / absent / recognition_accuracy | 26.41 ± 0.94 | 24.32 ± 2.51 |
| multi_needle / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 |
| multi_needle / present / needle_recall | 12.71 ± 11.72 | 25.83 ± 4.08 |
| multi_needle / present / recognition_accuracy | 100.00 ± 0.00 | 100.00 ± 0.00 |
| multi_needle / short / exact_retrieval | 0.00 ± 0.00 | 0.00 ± 0.00 |
| multi_needle / short / needle_recall | 36.98 ± 11.52 | 42.19 ± 4.69 |
| multi_needle / short / recognition_accuracy | 100.00 ± 0.00 | 100.00 ± 0.00 |
| needle_passkey / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 |
| needle_passkey / absent / recognition_accuracy | 26.98 ± 2.47 | 26.72 ± 2.07 |
| needle_passkey / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 |
| needle_passkey / present / needle_recall | 41.30 ± 27.58 | 45.36 ± 4.77 |
| needle_passkey / present / recognition_accuracy | 98.85 ± 0.63 | 99.90 ± 0.18 |
| needle_passkey / short / exact_retrieval | 61.46 ± 47.84 | 78.12 ± 37.89 |
| needle_passkey / short / needle_recall | 61.46 ± 47.84 | 78.12 ± 37.89 |
| needle_passkey / short / recognition_accuracy | 100.00 ± 0.00 | 100.00 ± 0.00 |

## Paired difference from standard

Percentage points, mean ± sample SD across matched training seeds.

| task | algebraic |
|---|---|
| associative_recall | 0.57 ± 1.42 |
| latest_value | 2.34 ± 1.51 |
| multi_needle | 0.10 ± 0.09 |
| needle_passkey | 4.06 ± 25.37 |
| arc_challenge | 0.37 ± 0.57 |
| arc_easy | 1.33 ± 1.35 |
| boolq | 1.99 ± 3.73 |
| hellaswag | 0.94 ± 0.11 |
| lambada | 1.27 ± 0.38 |
| openbookqa | -0.47 ± 0.83 |
| piqa | 0.53 ± 0.27 |
| winogrande | -0.58 ± 0.78 |

The longest recall budget includes answer space: 2,032 prompt tokens + 16 generated tokens.

Absent-evidence target hits are unsupported guesses, not successful retrieval. Recognition has four choices (25% chance); short controls check whether the answer format is usable.
