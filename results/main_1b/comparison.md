# Post-training comparison

Training seeds: 42.

Scores are percentages from one training seed; seed variance is not estimated. Paired differences use the same seed and examples. No pooled item confidence interval is implied.

## Held-out language modeling and training telemetry

NLL is in nats/token; perplexity is exp(NLL). Training times and throughput are trainer telemetry.

| metric | standard | algebraic |
|---|---|---|
| final_eval_nll | 2.30 | 2.27 |
| final_eval_ppl | 9.94 | 9.70 |
| train_seconds | 90952.56 | 95886.57 |
| mean_tok_per_s | 219951.50 | 208629.06 |

## Public tasks

| task / condition / metric | standard | algebraic |
|---|---|---|
| arc_challenge | 30.03 | 30.55 |
| arc_easy | 54.71 | 55.09 |
| boolq | 57.37 | 62.26 |
| hellaswag | 44.27 | 46.42 |
| lambada | 29.30 | 32.45 |
| openbookqa | 31.60 | 35.00 |
| piqa | 69.97 | 70.08 |
| winogrande | 51.70 | 53.67 |

## Recall with evidence

| task / condition / metric | standard | algebraic |
|---|---|---|
| associative_recall / present / exact_retrieval | 25.31 | 25.31 |
| latest_value / present / exact_retrieval | 17.03 | 17.81 |
| multi_needle / present / exact_retrieval | 0.47 | 0.47 |
| needle_passkey / present / exact_retrieval | 72.03 | 77.34 |

## Recall controls and diagnostics

| task / condition / metric | standard | algebraic |
|---|---|---|
| associative_recall / absent / needle_recall | 0.00 | 0.00 |
| associative_recall / absent / recognition_accuracy | 28.59 | 26.72 |
| associative_recall / absent / unsupported_target_hit | 0.00 | 0.00 |
| associative_recall / present / needle_recall | 25.31 | 25.31 |
| associative_recall / present / recognition_accuracy | 98.75 | 99.06 |
| associative_recall / short / exact_retrieval | 100.00 | 100.00 |
| associative_recall / short / needle_recall | 100.00 | 100.00 |
| associative_recall / short / recognition_accuracy | 100.00 | 100.00 |
| latest_value / absent / needle_recall | 0.00 | 0.00 |
| latest_value / absent / recognition_accuracy | 23.91 | 22.66 |
| latest_value / absent / unsupported_target_hit | 0.00 | 0.00 |
| latest_value / present / needle_recall | 17.03 | 17.81 |
| latest_value / present / recognition_accuracy | 50.62 | 49.84 |
| latest_value / short / exact_retrieval | 78.12 | 50.00 |
| latest_value / short / needle_recall | 78.12 | 50.00 |
| latest_value / short / recognition_accuracy | 93.75 | 90.62 |
| multi_needle / absent / needle_recall | 0.00 | 0.00 |
| multi_needle / absent / recognition_accuracy | 25.62 | 27.66 |
| multi_needle / absent / unsupported_target_hit | 0.00 | 0.00 |
| multi_needle / present / needle_recall | 19.38 | 33.20 |
| multi_needle / present / recognition_accuracy | 100.00 | 100.00 |
| multi_needle / short / exact_retrieval | 9.38 | 0.00 |
| multi_needle / short / needle_recall | 45.31 | 50.00 |
| multi_needle / short / recognition_accuracy | 100.00 | 100.00 |
| needle_passkey / absent / needle_recall | 0.00 | 0.00 |
| needle_passkey / absent / recognition_accuracy | 27.34 | 29.38 |
| needle_passkey / absent / unsupported_target_hit | 0.00 | 0.00 |
| needle_passkey / present / needle_recall | 72.03 | 77.34 |
| needle_passkey / present / recognition_accuracy | 100.00 | 100.00 |
| needle_passkey / short / exact_retrieval | 100.00 | 100.00 |
| needle_passkey / short / needle_recall | 100.00 | 100.00 |
| needle_passkey / short / recognition_accuracy | 100.00 | 100.00 |

## Paired difference from standard

Percentage points for the matched training seed.

| task | algebraic |
|---|---|
| associative_recall | 0.00 |
| latest_value | 0.78 |
| multi_needle | 0.00 |
| needle_passkey | 5.31 |
| arc_challenge | 0.51 |
| arc_easy | 0.38 |
| boolq | 4.89 |
| hellaswag | 2.14 |
| lambada | 3.14 |
| openbookqa | 3.40 |
| piqa | 0.11 |
| winogrande | 1.97 |

The longest recall budget includes answer space: 2,032 prompt tokens + 16 generated tokens.

Absent-evidence target hits are unsupported guesses, not successful retrieval. Recognition has four choices (25% chance); short controls check whether the answer format is usable.
