# FlashAttention choice for the hybrid

The active speed target covers both the pure algebraic model and the hybrid,
at the attention-kernel level and for complete optimizer steps. The reference
is the fastest measured standard implementation on the same hardware and
shape. Neither model has achieved parity yet.

## Which attention implementation

The hybrid is the standard backbone with learned QK RMS normalization and
the algebraic vocabulary head `rho(z/2)^8`. Its attention weights remain
ordinary softmax, with no sink. It therefore uses the **same resident-memory
Pallas softmax FlashAttention as the standard model**. The pure model retains
the algebraic `rho(s/8)^16` FlashAttention with its learned zero-value sink.

On the installed TPU v4/JAX 0.6.2 stack, at B=8,H=12,T=2048,D=64:

| Standard attention implementation | Forward (ms) | Forward + q/k/v backward (ms) |
|---|---:|---:|
| Repository resident-memory Pallas, 512-square tiles | 1.564 | **3.364** |
| Maintained Splash, 512-square tiles, fused backward | **1.518** | 4.178 |
| JAX TPU flash attention, default settings | — | 22.124 |

The current training dispatch uses the resident kernel. Splash is a measured
candidate for a forward-only path, but it is not selected by current model
dispatch. These comparisons concern this installed stack, not a benchmark
of every contemporary GPU/TPU attention package. Values have dimension 64;
the Splash comparison pads only V to 128 and slices the output.

## Optimize the extra QK normalization work

For a row, `tau=(mean(x^2)+eps)^(-1/2)` and `y=x*tau*gain`. The gain VJP is

```
d_gain[d] = sum_rows d_y[row,d] * x[row,d] * tau[row].
```

The new kernel reduces this sum within 1024-row Pallas tiles and sums the
small partials in XLA. Its forward and input VJP retain the native JAX
calculation. Lean checks the gain derivative, disjoint partition identity,
and real radial-subtraction identity; it does not assert equal IEEE rounding.

An earlier all-Pallas RMS prototype gave faster timing but failed complete
model checks with large QK gains (up to 7.1% relative gradient error). Retaining
native forward and a matching radial formula reduced the error but still
failed the preselected 1% gate in one case. These prototypes are not used.
The accepted gain-only kernel passes all five complete-model checks,
including a 12-layer 125M model, a 10-layer 60M model, and gains of four. The
largest parameter-gradient relative L2 difference is recorded in
`results/analysis/qk_model_production_full_validation.json`.

Automatic dispatch is restricted to BF16 softmax attention at context 2048
and D=64 with compatible row counts and FP32 gains. Native RMS remains the
pure model's path: the same gain-reduction substitution increased its
exploratory optimizer-step time from 181.2 to 187.4 ms. This result illustrates
why isolated normalization timings do not determine complete model speed.

## Complete training-step confirmation

Randomized architecture order, three fresh seeds 191/193/197, 30 real AdamW
steps after the first compilation step, one TPU v4 chip, micro-batch 8,
context 2048, original model presets and dense output heads:

| 125M model/path | Median optimizer step (ms) | Ratio to standard |
|---|---:|---:|
| Standard | 156.587 | 1.0000 |
| Hybrid, native QK gain reduction | 178.414 | 1.1394 |
| **Hybrid, tiled QK gain reduction** | **171.627** | **1.0961** |
| Pure algebraic, original link | 181.203 | 1.1572 |

The accepted change lowers hybrid latency by **3.80%**, leaving it **9.61%**
slower than standard. The pure model remains **15.72%** slower for these
complete steps. Raw paired jobs, arguments, source hashes, and metadata are
in `results/kernels/model_step_confirm_125m.json`. The tiny bootstrap
intervals reflect these three same-chip profiles; they do not establish
variation across trained checkpoints, devices, or a multi-host slice.
These short profiles are speed measurements, not language-model quality runs.

The same three-seed protocol at 60M gives **86.985 ms standard**, **97.052 ms
native hybrid**, **95.010 ms tiled-gain hybrid**, and **100.592 ms pure
algebraic**. The hybrid change is a **2.11%** latency reduction; remaining
hybrid/pure gaps are **9.22%/15.64%**. The raw record is
`results/kernels/model_step_confirm_60m.json`. Each profile includes input
loading and AdamW work; component timings are not added to estimate these
complete-step numbers.

The component benchmark uses the deployed n=8,c=2 head, rather than an older
n=16,c=16 head configuration. Its forward/backward medians are 21.519 ms
for the algebraic head and 19.092 ms for cross-entropy. Saving the complete
logit gradient loses: 29.735 ms with float32 caching and 24.588 ms with BF16
caching. The deployed head is unchanged. Raw measurements are
`model_component_performance_125m.json` and `head_gradient_cache_probe.json`
under `results/kernels/`.
