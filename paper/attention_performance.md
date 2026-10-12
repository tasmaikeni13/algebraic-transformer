# Attention kernel engineering: measured status

**Speed parity is not achieved.** The mathematical model remains
`rho(s/8)^16` with a learned zero-value sink. The new implementation uses
an inverse denominator, compact row statistics, and a checked range guard.
The fully algebraic training path passes the primitive audit.

## Latest confirmed resident-memory backend

The backward now keeps an entire head's K/V in VMEM, iterates over 512-key
compute chunks, and accumulates each query block's dQ before its only store.
It retains FP32 accumulation and the original weight/VJP formulas. The
automatic dispatch is restricted to compact-statistic bfloat16 inputs,
T=2048,D=64, 512-square forward tiles, and softmax or exact asinh n=16.
Other shapes/dtypes retain the previous backend. An explicit
`resident_bwd=False` allows comparison with per-key-block dQ partials.

Fresh data seed 71, fifteen randomized paired rounds of 30 queued calls:

| Forward + q/k/v backward | 125M shape (ms) | 60M shape (ms) |
|---|---:|---:|
| Guarded algebraic, dQ partials | 5.943 | 3.948 |
| **Guarded algebraic, resident dQ** | **5.419** | **3.531** |
| Standard compact, dQ partials | 3.971 | 2.604 |
| **Standard compact, resident dQ** | **3.364** | **2.214** |
| Maintained Splash 512, fused backward | 4.178 | 2.710 |

This lowers deployed algebraic latency by about 8.8% at the 125M shape and
10.5% at the 60M shape. The same optimization benefits standard attention
more, so the fair algebraic/standard ratios are **1.6107** [1.6105,1.6110]
and **1.5950** [1.5948,1.5951]. The speed goal remains open. Forward is
unchanged. The tiny intervals describe within-run repeatability only.
Raw records are `results/kernels/attention_resident_confirm_{125m,60m}.json`.

At the 125M shape, the old F32 dQ partial array occupies 192 MiB. Removing
its store and subsequent reduction read avoids about 384 MiB of device-memory
traffic per attention backward; this byte count is calculated from tensor
shapes, not measured bandwidth. Finite key-partition/slope identities are
checked in Lean. Reassociation can still change floating-point rounding.

The tile sweep tests query/key compute blocks 256/512/1024. 512x512 wins
among successful candidates. Larger float32 resident asinh shapes exceed
16 MiB VMEM; large bfloat16 query tiles also fail. Failure diagnostics are
retained in `attention_resident_tiles.json` and
`attention_resident_bf16_tiles.json`. Changing the order of independent
score/do@V matrix products changes latency very little; doing do@V first
is slower (`attention_overlap_probe.json`). This simple reorder did not
resolve the nonlinear cost.

**71 TPU tests pass**, including independent dense-reference resident VJPs,
context 2048, bfloat16, both saved-statistic formulas, sinks, automatic
dispatch, and extreme-score fallback. The context-2048 algebraic model
primitive audit passes. **52 Lean theorems** build successfully, covering
the original operator, accumulation identities, and experimental links.

The new-link probes are documented in [attention_pade.md](attention_pade.md),
[attention_binomial.md](attention_binomial.md), and
[attention_quadratic.md](attention_quadratic.md). With the same resident
backend, the fastest affine candidate takes 4.414 ms; the quadratic candidate
takes 4.651 ms with an exact product derivative or 4.349 ms with an approximate
reciprocal derivative. These are isolated prototypes, with formal/Monte Carlo
analysis and numerical checks; no language-model quality result or model
adoption is claimed.

## Confirmed measurements

The table below records the preceding compact-statistic/guard milestone,
before resident dQ was introduced.

Single TPU v4 chip, installed JAX 0.6.2/libtpu 0.0.17, bfloat16 inputs,
causal context 2,048, head dimension 64, batch 8. The model labels specify
head count (12 for 125M, 8 for 60M); these are attention microbenchmarks,
not new full-model training-throughput measurements.

| Forward + q/k/v backward | 125M shape (ms) | 60M shape (ms) |
|---|---:|---:|
| Original row layout/statistic, corrected clipping VJP, algebraic | 6.601 | 4.280 |
| Inverse denominator + compact statistics, algebraic | 6.042 | 3.986 |
| **Guarded production algebraic kernel** | **5.944** | **3.947** |
| **Compact-statistic standard kernel** | **3.971** | **2.605** |
| Tuned maintained Splash, fused backward, values padded to 128 | 4.178 | 2.710 |
| JAX flash attention, defaults | 22.131 | 14.545 |

Guarded algebraic latency / fastest measured standard latency:

| Shape | Paired geometric ratio | Bootstrap 95% interval |
|---|---:|---:|
| 125M | 1.4968 | [1.4965, 1.4971] |
| 60M | 1.5151 | [1.5148, 1.5153] |

Fifteen randomized paired rounds, 30 queued calls per measurement, fresh
data seed 37 after configuration selection. Compilation and three warmup
calls are excluded. A synchronization before each measurement drains the
previous call; a final synchronization includes all queued work. Raw samples,
source hashes, devices, and arguments are in
`results/kernels/attention_confirm_{125m,60m}.json`. Seed-level numerical
statistics and timing ratios are in `results/analysis/attention_statistics.json`.

These narrow intervals measure repeatability in one process on one chip.
They do not cover different chips, learned-score distributions, shapes,
software versions, or end-to-end model variability. The reference kernel
was selected in the earlier sweep and confirmed with the new seed. Sink
gradients are checked for accuracy separately; the timing loss differentiates
q, k, and v, with a constant sink matching the old benchmark convention.

Forward-only latency is 2.231 ms for guarded algebraic at the 125M shape;
the fastest measured standard forward is Splash at 1.518 ms. At the 60M
shape, guarded algebraic is 1.433 ms and compact standard is 0.931 ms.
Forward parity is also open.

## What changed and what failed

1. **Save `1/Z` for additive accumulation.** This removes the n=16 root
   chain and reconstructs probabilities with `rho(x)^16/Z`. The formula is
   checked in Lean, and the Monte Carlo error analysis favors this statistic.
2. **Store one statistic per row.** The old output wrote 128 replicated
   lanes, then sliced in XLA. At the 125M shape this output alone was about
   96 MiB; the new output is 0.75 MiB. A transpose before storing avoids a
   Mosaic layout conversion unsupported by the first implementation.
   The standard kernel receives the same improvement.
3. **Remove clipping when a range guard proves it unnecessary.** A scalar
   Cauchy-Schwarz test covers scores of either sign, and a value bound covers
   numerator overflow. Scores outside that range use shifted stable rho.
   This addresses whole-row underflow and avoids a per-score clamp gradient
   in the fast branch. The guard includes additional reductions, so its
   latency gain is modest. Its main benefit is preserving the original link.
4. **Correct the explicit clipped variant's VJP.** The old straight-through
   derivative changed the mathematical gradient above the cap. The derivative
   is now zero there. Old 6.32 ms records predate this correction; they are
   historical measurements, not the current comparison row.
   The cap is now derived from `rho((t-1/t)/2)=t`, with
   `t=(1e37/(T+1))^(1/n)`, rather than the unsafe large-n heuristic. The
   first shifted tile also establishes its actual maximum, so the finite
   sentinel cannot corrupt very negative rows. Tail tests reach kernel-unit
   scores of magnitude 10 million.
5. **Test bounded polynomial radical evaluation.** A degree-four radical and
   slope approximation on `|x|<=1` has an exact rational Bernstein certificate
   with real weight error below 0.094% and TV bound below 0.094% before
   arithmetic rounding. TPU output/VJP checks pass, but the measured compact
   variant takes **9.70 ms** at the 125M shape. It stays opt-in and is not
   enabled by the model. Arithmetic count alone did not predict speed.
6. **Sweep tile shapes rather than assume a winner.** Smaller blocks lost to
   512x512 at these shapes. A 1024x1024 algebraic backward exceeded the
   physical 16 MiB VMEM budget (20.86 MiB reported by the compiler). The
   nominal compiler arena limit does not enlarge device memory. Archived
   exploratory files retain the failed candidates; their source hashes
   distinguish those prototypes from the final confirmation.

## Mathematical and statistical validation

`paper/attention_exact.md` connects the exact operator, Jacobian, sensitivity
bounds, range guard, and numerical counterexamples. The original four Lean modules prove
radical identities, normalization/statistic formulas, derivatives, and range
bounds. `lake build` succeeds with no project `sorry`/additional axioms.

The preceding milestone's **55 TPU tests pass**: forward outputs and q/k/v/sink gradients, float32 and
bfloat16, square/rectangular tiles, both backward statistics, compact output,
guarded fallback, and the opt-in polynomial candidate. The NumPy analysis
uses 9,216 random rows across eight independent seeds plus adversarial
negative/positive rows. This is evidence about the specified domains, not
a claim that random rows reproduce every learned attention pattern.

## Modern FlashAttention and the next iteration

FlashAttention-3/4 emphasize overlap of matrix and nonlinear work, reducing
rescaling and intermediate memory traffic. FA4 also evaluates exponentials
with polynomials to relieve a hardware bottleneck. These design principles
motivated the candidates above; their GPU speedups do not transfer directly
to this TPU. Primary references:
[FA3](https://arxiv.org/abs/2407.08608),
[FA4](https://arxiv.org/abs/2603.05451), and
[Pallas TPU design](https://docs.jax.dev/en/latest/pallas/tpu/details.html).

The installed software is older than current JAX releases. A claim of parity
with all contemporary attention kernels would require a separate comparison
on an updated isolated environment and a supported maintained kernel.
[JAX releases](https://github.com/jax-ml/jax/releases) and
[Tokamax](https://github.com/openxla/tokamax) are relevant primary sources;
this milestone makes a measured claim about the installed stack only.

Next work must resolve the remaining gap: inspect the compiled nonlinear
and backward pipeline costs, test radical evaluation alternatives with
bounded error, and restructure overlap/recomputation without exceeding VMEM.
Candidates that optimize the existing link must retain its n=16,c=8
semantics and pass VJP checks. A new algebraic link needs formal and Monte
Carlo analysis plus retrieval/training-quality checks before adoption. Every
candidate must be compared with the fastest standard implementation on the
same software and hardware. A faster result on another head dimension cannot
establish parity for the original 64-dimensional training shapes.

## Reproduce

```bash
JAX_PLATFORMS=cpu python analysis/attention_numerics.py
JAX_PLATFORMS=cpu python analysis/attention_polynomial.py
cd formal && lake exe cache get && lake build
```

On a TPU worker, use its single-chip wrapper (or an equivalent TPU setup):

```bash
~/bin/chiprun 0 -m pytest -q tests/test_flash_kernel_tpu.py
~/bin/chiprun 0 analysis/attention_performance.py --blocks 512 \
  --families softmax,softmax_compact,asinh,asinh_compact,asinh_guarded \
  --baselines --baseline-blocks 512 --seed 37 --rounds 15 --iters 30 \
  --out results/kernels/attention_confirm_125m.json
# Add --heads 8 for the 60M shape.
JAX_PLATFORMS=cpu python scripts/audit_algebraic.py --set attn_impl=flash
python analysis/attention_statistics.py
```
