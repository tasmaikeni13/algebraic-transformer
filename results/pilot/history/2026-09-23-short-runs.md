# Short throughput diagnostics, 23 September 2026

Each paired diagnostic ran for 1,000 steps on 16 TPU v4 chips using the same
WikiText-103 token cache and model and batch sizes. These runs are shorter
than the 100,000-step pilot and are excluded from its reported outcome.

| Source configuration | Algebraic tokens/s | Standard tokens/s | Ratio |
| --- | ---: | ---: | ---: |
| `af0047c`, fused vocabulary loss with three passes | 2,078,007 | 2,454,100 | 0.84675 |
| Causal attention backward tile pruning | 2,209,690 | 2,441,579 | 0.90503 |
| Causal pruning with BF16 inputs and FP32 dot accumulation | 2,210,538 | 2,456,788 | 0.89977 |
| `b3e1f5f`, Gram normalization and 16,384-token vocabulary tiles | 2,360,774 | 2,490,032 | 0.94809 |

The `b3e1f5f` short run recorded a perplexity ratio of 1.00097, zero
non-finite updates, zero loss spikes, and peak clipped gradient norm
1.00000024. At the measured per-chip attention shape, algebraic attention
took 0.644 ms for forward and backward; maintained TPU softmax attention took
0.645 ms. These component timings do not determine full-step throughput.
