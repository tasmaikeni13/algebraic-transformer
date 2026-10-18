# Post-training comparison

Training seeds: 42, 43, 44, 45, 46.

Scores are percentages, mean ± sample SD across training seeds. Paired differences use the same seed and examples. No pooled item confidence interval is implied.

## Held-out language modeling and training telemetry

NLL is in nats/token; perplexity is exp(NLL). Training times and throughput are trainer telemetry.

| metric | standard | algebraic |
|---|---|---|
| final_eval_nll | 3.20 ± 0.00 | 3.16 ± 0.00 |
| final_eval_ppl | 24.53 ± 0.09 | 23.53 ± 0.10 |
| train_seconds | 1605.80 ± 0.55 | 1735.65 ± 0.49 |
| mean_tok_per_s | 1570799.01 ± 423.49 | 1455788.68 ± 481.94 |

## Public tasks

| task / condition / metric | standard | algebraic |
|---|---|---|
| arc_challenge | 24.10 ± 0.92 | 24.83 ± 0.74 |
| arc_easy | 43.16 ± 0.59 | 43.48 ± 0.97 |
| boolq | 60.33 ± 1.60 | 60.60 ± 1.83 |
| hellaswag | 28.43 ± 0.09 | 28.87 ± 0.45 |
| lambada | 17.15 ± 0.76 | 17.45 ± 0.65 |
| openbookqa | 27.12 ± 1.71 | 27.08 ± 1.68 |
| piqa | 61.75 ± 0.56 | 61.94 ± 0.38 |
| winogrande | 51.32 ± 0.68 | 52.19 ± 1.29 |

## Recall with evidence

| task / condition / metric | standard | algebraic |
|---|---|---|
| associative_recall / present / exact_retrieval | 1.88 ± 1.60 | 2.66 ± 1.62 |
| latest_value / present / exact_retrieval | 3.00 ± 2.67 | 3.31 ± 0.98 |
| multi_needle / present / exact_retrieval | 0.00 ± 0.00 | 0.00 ± 0.00 |
| needle_passkey / present / exact_retrieval | 3.03 ± 1.74 | 2.12 ± 2.07 |

## Recall controls and diagnostics

| task / condition / metric | standard | algebraic |
|---|---|---|
| associative_recall / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 |
| associative_recall / absent / recognition_accuracy | 25.72 ± 2.81 | 26.72 ± 4.60 |
| associative_recall / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 |
| associative_recall / present / needle_recall | 1.88 ± 1.60 | 2.66 ± 1.62 |
| associative_recall / present / recognition_accuracy | 77.97 ± 4.16 | 85.72 ± 2.24 |
| associative_recall / short / exact_retrieval | 25.62 ± 26.00 | 10.62 ± 20.32 |
| associative_recall / short / needle_recall | 25.62 ± 26.00 | 10.62 ± 20.32 |
| associative_recall / short / recognition_accuracy | 100.00 ± 0.00 | 100.00 ± 0.00 |
| latest_value / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 |
| latest_value / absent / recognition_accuracy | 25.41 ± 4.00 | 25.56 ± 3.29 |
| latest_value / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 |
| latest_value / present / needle_recall | 3.00 ± 2.67 | 3.31 ± 0.98 |
| latest_value / present / recognition_accuracy | 46.88 ± 4.74 | 51.12 ± 7.07 |
| latest_value / short / exact_retrieval | 13.12 ± 8.95 | 16.25 ± 19.69 |
| latest_value / short / needle_recall | 13.12 ± 8.95 | 16.25 ± 19.69 |
| latest_value / short / recognition_accuracy | 65.62 ± 10.60 | 65.62 ± 7.65 |
| multi_needle / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 |
| multi_needle / absent / recognition_accuracy | 21.47 ± 3.48 | 23.00 ± 4.21 |
| multi_needle / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 |
| multi_needle / present / needle_recall | 1.00 ± 0.81 | 0.99 ± 1.40 |
| multi_needle / present / recognition_accuracy | 96.69 ± 2.40 | 97.62 ± 3.18 |
| multi_needle / short / exact_retrieval | 0.00 ± 0.00 | 0.00 ± 0.00 |
| multi_needle / short / needle_recall | 0.83 ± 1.36 | 1.67 ± 1.58 |
| multi_needle / short / recognition_accuracy | 99.38 ± 1.40 | 100.00 ± 0.00 |
| needle_passkey / absent / needle_recall | 0.00 ± 0.00 | 0.00 ± 0.00 |
| needle_passkey / absent / recognition_accuracy | 16.88 ± 4.59 | 19.06 ± 3.68 |
| needle_passkey / absent / unsupported_target_hit | 0.00 ± 0.00 | 0.00 ± 0.00 |
| needle_passkey / present / needle_recall | 3.03 ± 1.74 | 2.12 ± 2.07 |
| needle_passkey / present / recognition_accuracy | 95.06 ± 0.99 | 94.09 ± 2.32 |
| needle_passkey / short / exact_retrieval | 23.12 ± 22.81 | 12.50 ± 22.86 |
| needle_passkey / short / needle_recall | 23.12 ± 22.81 | 12.50 ± 22.86 |
| needle_passkey / short / recognition_accuracy | 100.00 ± 0.00 | 100.00 ± 0.00 |

## Paired difference from standard

Percentage points, mean ± sample SD across matched training seeds.

| task | algebraic |
|---|---|
| associative_recall | 0.78 ± 2.31 |
| latest_value | 0.31 ± 2.97 |
| multi_needle | 0.00 ± 0.00 |
| needle_passkey | -0.91 ± 2.82 |
| arc_challenge | 0.73 ± 0.64 |
| arc_easy | 0.32 ± 0.79 |
| boolq | 0.27 ± 1.16 |
| hellaswag | 0.44 ± 0.43 |
| lambada | 0.30 ± 1.17 |
| openbookqa | -0.04 ± 2.15 |
| piqa | 0.18 ± 0.81 |
| winogrande | 0.87 ± 1.68 |

The longest recall budget includes answer space: 2,032 prompt tokens + 16 generated tokens.

Absent-evidence target hits are unsupported guesses, not successful retrieval. Recognition has four choices (25% chance); short controls check whether the answer format is usable.
