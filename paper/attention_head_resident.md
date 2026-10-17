# Pure algebraic attention: whole-head residency

Speed parity remains open. This continuation optimizes the original
`rho(s/8)^16` attention and its learned zero-value sink. It introduces no new
attention link.

## Exact kernel schedule

For BF16 inputs at context 2048 and head dimension 64, the guarded algebraic
fast path now retains Q, K, V, and the output buffers for one head in VMEM.
Static 128-query by 512-key score tiles bound the nonlinear working set.
The last key tile ends at the end of the current query tile; later keys are
causally invisible. dQ is accumulated before its only store, and dK/dV remain
FP32 until the head is finished. No quadratic-sized attention array is written
to HBM. The real weight and VJP formulas are unchanged.

`head_resident=False` selects the preceding query-streamed resident backend.
Automatic selection requires compact statistics, the measured shape, a range
guard, and no explicit diagonal-schedule override. Other shapes use the
existing backends. The shifted tail uses the existing query-streamed schedule.
`head_resident=True` also permits a separately tuned softmax control: its
512-query tiles are faster than the algebraic winner's 128-query tiles.
The standard model defaults continue to use their prior dispatch.

The first whole-head sweep reduced algebraic training latency from 4.976 to
4.491 ms. Further score-tile experiments selected 128x512: 128x128 and
128x256 were slower for the complete VJP, and 128x1024 took 4.371 ms.
The first smaller-tile draft failed the softmax scratch-shape checks; those
errors remain in the exploratory records. Corrected controls and the deployed
backend pass independent dense-reference checks. Historical probes are
distinguished by their source fingerprints.

## Confirmed timings

The following records include the subsequent FP32 tail-centering correction.
Measurements use one TPU v4 chip, BF16 inputs, causal context 2048, D=64,
fresh data seeds, randomized paired rounds, and 30 queued calls per round.
Compilation and warmup are excluded from the latency samples. Training means
forward plus q/k/v VJPs; sink VJPs are checked separately and exercised in the
complete-model smoke.

| Shape / software | Algebraic forward | Standard forward | Algebraic forward + backward | Fastest measured standard forward + backward |
|---|---:|---:|---:|---:|
| B=8,H=12; JAX 0.6.2 | 1.617 ms | 1.150 ms | 4.301 ms | 2.675 ms |
| B=8,H=12; isolated JAX 0.11.2 | 1.758 ms | 1.165 ms | 4.259 ms | 2.700 ms |
| 350M micro-batch, B=2,H=16; JAX 0.6.2 | 0.518 ms | 0.346 ms | 1.450 ms | 0.847 ms |
| 1B micro-batch, B=1,H=24; JAX 0.6.2 | 0.392 ms | 0.263 ms | 1.106 ms | 0.641 ms |

The strongest measured standard control is the whole-head softmax kernel.
On the isolated newer stack, maintained Tokamax Splash takes 1.256 ms forward
with 512 tiles and 3.644 ms forward plus backward with 1024 tiles. The Tokamax
attention API rejects TPU v4; that failure is retained in the raw record.

At the 125M attention shape, whole-head algebraic accumulation reduces
training latency by about 15.2% against the preceding resident causal kernel
on JAX 0.6.2. The fair algebraic/standard ratio is 1.6076, with a within-run
bootstrap interval [1.6073,1.6079]. On JAX 0.11.2 it is 1.5773. The smaller
production micro-batches have ratios 1.7129 and 1.7252. These ratios do not
prove parity; the remaining gap is substantial. Intervals cover measurement
rounds in one process, not software versions, chips, learned distributions,
or complete-model variability.

Raw confirmations are `results/kernels/attention_head_precise_confirm_125m.json`,
`attention_head_precise_modern_confirm_125m.json`,
`attention_head_confirm_350m_shape.json`, and
`attention_head_confirm_1b_shape.json`. The 1B shape matches the current
1536-wide, 24-head configuration; the recovered 32-head probe describes an
earlier shape. `analysis/attention_head_statistics.py` validates sample counts
and medians and recomputes paired bootstrap ratios.

## Complete-model smoke and arithmetic checks

The initial five-step 125M smoke reduced the pure model from 174.056 to
159.453 ms per AdamW update. The strongest standard control took 143.348 ms.
The final smoke, with corrected tail centering, takes 162.285 ms for pure,
176.502 ms for the preceding pure backend, and 143.287 ms for the strongest
standard control. The remaining full-step gap is 13.3%. This fresh-seed smoke
is retained separately in `attention_head_precise_model_paired_smoke_125m.json`.
It uses synthetic tokens,
one compilation step and five timed updates per case, no corpus, and no
saved checkpoints. These are short speed checks, not training-quality results
or statistically established full-model parity.

Complete-model output and every parameter VJP are compared with the preceding
resident causal backend at context 2048. The initial check's maximum output
difference is 0.0000611 and maximum parameter relative L2 difference is
0.000492. The existing component checks include the independently implemented
dense operator, learned sink, negative rows, both statistic formulas,
automatic dispatch, and tail fallback.

The positive constant-score tail exposed a centering error in the old VJP:
its `do . output` used the BF16-rounded output. The shifted fallback now saves
the unrounded FP32 output for that dot while returning BF16 to the caller.
BF16 rounding of the score VJP before dQ's matrix contraction still produces
a small residual when the mathematical dQ is zero. The new tail check uses an
independent FP32 dense oracle and bounds that residual from float64 value and
cotangent products. It also checks bitwise equality with the same fallback
when whole-head residency is disabled. This is a case-specific arithmetic
bound, not a universal floating-point certificate. The extra residual affects
compilation of the guarded backward, increasing measured latency slightly;
the final confirmation includes that cost.

The full algebraic model's forward and backward primitive audit passes at
context 2048, including the new kernel. Its training path contains no
exponential or logarithm. The Lean build checks 76 project theorems, including
causal suffix omission, finite VJP partitioning, and movement of normalization
factors through value and score contractions. Real-arithmetic identities do
not certify compiler lowering, machine rounding, or speed.
The final suite passes 113 TPU output/gradient/dispatch checks and 24 CPU
component/training/workflow checks. Its TPU output is retained in
`results/kernels/attention_head_tpu_test_output.txt`.
The centering-bias and constant-key rounding identities are also checked in
Lean with explicit premises; the actual machine-rounding bounds are not
imported as axioms or claimed to follow from those real identities.
The 350M and 1B pure configurations each complete one synthetic AdamW step
on all 16 TPU v4 chips, with the production global batch of 512 and
accumulation counts 16 and 32. Their fused attention and vocabulary head
checks are retained in `results/verification/distributed_attention_head_350m.json`
and `distributed_attention_head_1b.json`. Neither check is a training run or
throughput comparison.

## Theory candidates and rejected rearrangements

Moving inverse denominators into row cotangents passes output and q/k/v/sink
VJP checks but barely changes query-streamed latency: 4.973 versus 4.977 ms.
Appending a value coordinate of one computes the denominator on the MXU, but
takes 5.010 ms. Combining the two takes 5.006 ms. They remain isolated
experiments in `attention_projective_probe.py` and are not production dispatch.

The recovered anchored cubic radical has an exact rational Bernstein
certificate on |x| <= 1.125. The new Lean projective-error proof sharpens the
normalized TV bound to 4.617%, compared with the earlier coarse bound of
10.683%. Random-row errors are smaller, but the uniform bound and adversarial
VJPs do not justify substituting it for the original operator. It remains an
opt-in research candidate, with no language-model quality claim.

## Reproduction and remaining work

```bash
bash scripts/tpu_python.sh 0 -m pytest -q tests/test_flash_kernel_tpu.py tests/test_qk_norm_tpu.py
bash scripts/tpu_python.sh 0 analysis/attention_performance.py \
  --blocks 512 --families softmax_head,asinh_guarded,asinh_guarded_resident \
  --rounds 11 --iters 30 --seed 587 --out /tmp/head-confirm.json
bash scripts/tpu_python.sh 0 analysis/attention_checkpoint_probe.py \
  --head-comparison --steps 5 --seed 593 --out /tmp/head-model-smoke.json
JAX_PLATFORMS=cpu python analysis/attention_head_statistics.py
(cd formal && lake build)
```

The target remains pure algebraic attention and complete-model speed parity
against the strongest supported standard implementation on this TPU slice.
Next work must reduce the nonlinear backward cost and the remaining QK,
rotation, activation, and head overhead. A faster isolated attention operator
alone cannot close the complete-model goal, and a changed algebraic link
requires numerical, formal, and quality evidence before model adoption.
No 125M, 350M, or 1B long training run is authorized or launched here.

The engineering references are the primary
[Pallas TPU hardware/layout guide](https://docs.jax.dev/en/latest/pallas/tpu/details.html),
[TPU pipelining guide](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html),
[FlashAttention-3 paper](https://arxiv.org/abs/2407.08608),
[FlashAttention-4 paper](https://arxiv.org/abs/2603.05451), and
[Tokamax implementation](https://github.com/openxla/tokamax).
They motivate overlap, operand reuse, and reduced nonlinear work; GPU speedups
are not measurements of TPU v4 behavior.
