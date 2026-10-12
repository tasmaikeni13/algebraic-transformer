# Rational attention candidate: analysis before adoption

This is a candidate, not the deployed attention link. The deployed
asinh kernel remains in commit `25c0831`. A hardware speedup alone does not
establish that this new link preserves language-model quality.

## Definition and operating range

Let `x=clip(s/c,-a,a)` and `r(x)=(1+x)/(1-x)`, with `0<a<1`.
The positive weight is `kappa(s)=r(x)^n`, including the zero-value sink.
Normalizing these weights gives softmax of the conceptual logit
`2n atanh(x)`; the implementation uses division and integer powers.

The leading candidate is **n=8,c=16,a=3/4**. Its slope at the origin is
`beta=2n/c=1`, matching standard softmax. Taylor expansion gives
`2n atanh(s/(2n))=s+s^3/(12n^2)+O(s^5/n^4)`; this is a repeated rational
Padé approximation to the exponential. Positive scores sharpen beyond the
linear temperature. The existing asinh link compresses them instead.

Clipping is part of this candidate's mathematical definition, with zero
VJP outside the interval and a zero subgradient at a boundary. The cap
is not silently claimed to reproduce unbounded asinh attention.

For `a=3/4`, `1/7<=r<=7`, so n=8 weights satisfy
`7^(-8)<=kappa<=7^8`. At context 2,048, every nonempty denominator stays
between a positive normal float32 value and about `1.18e10`. This avoids
the whole-negative-row underflow and running-maximum work of an unbounded
kernel. Numerator accumulation still assumes ordinary finite values.

## Gradient and the proposed optimization

Inside the cap,

```
r'(x) = 2/(1-x)^2
kappa'(x) = 2n r(x)^(n-1)/(1-x)^2
d log kappa / ds = (2n/c)/(1-x^2).
```

The existing normalized-output/Jacobian proofs apply to this positive link
once its derivative is established. At `a=3/4`, n=8,c=16, the logarithmic
slope is at most `16/7`, giving Jacobian operator norm at most `8/7` by the
written argument in `attention_exact.md`. That is close to the deployed
asinh bound of one; it does not ensure that training behaves identically.

Saving `u=1/Z`, the backward can reuse `v=1/(1-x)` and the power chain:

```
r=(1+x)v
r2=r*r; r4=r2*r2; r8=r4*r4
p=r8*u
ds/n = 2*v*v*(r4*r2*r)*u*(dp-di)
```

This uses one division for the score tile, and no per-score square root.
It also skips a second division for the logarithmic slope. The slope
formula and this product formula will be compared on identical arrays;
operation counts alone are not a hardware result.

## Quality risks to measure

At unit QK-normalization in dimension 64, all raw scores satisfy `|s|<=8`.
The candidate's clipping threshold is 12, so unit-gain rows do not clip.
Gains can grow: with distractor score SD four, the threshold is only three
standard deviations away, and multiple distractors can tie a clipped cue.
Gaussian tails alone cannot establish the actual learned-score range.

The Monte Carlo investigation must report temperature-matched distortion,
clip frequency, cue probability, and ties across normalized-sphere rows
and grown gains. Cue probabilities are static routing diagnostics, not
trained associative-recall accuracy. Adoption additionally requires a
training comparison using the same algebraic head, QK-normalization,
data budget, and optimization budget. The earlier hybrid results motivate
trying a softmax-like link; they do not prove this candidate's quality.

The hardware probe uses process-local replacements of the existing Pallas
weight and tile-gradient functions. Baseline executables are compiled before
replacement, then retained directly; JAX trace caches are cleared before
candidate compilation. This isolates the experiment from the deployed
model and prevents stale traces from masquerading as a new kernel.

## Measured decision

The updated adapter skips the redundant asinh overflow cap because the
rational link applies its own intrinsic cap. On the same TPU v4/JAX 0.6.2
125M attention shape, nine randomized paired rounds of 30 calls give:

| Kernel | Forward (ms) | Forward + q/k/v backward (ms) |
|---|---:|---:|
| Standard compact | 1.564 | 3.972 |
| Deployed guarded asinh | 2.231 | 5.942 |
| Rational n=8,c=16, slope derivative | 2.308 | 7.013 |
| Rational n=8,c=16, product derivative | 2.308 | 6.522 |

Float32 output and q/k/v/sink VJPs pass comparison with an independent dense
reference. Source hashes and samples are in
`results/kernels/attention_link_probe.json`; the earlier adapter is archived
in `attention_pade_probe.json`. These are exploratory microbenchmarks, not
new model-throughput measurements.

The static routing study uses eight seeds, 128 normalized-sphere rows per
seed, dimension 64, and 2,048 distractors. At QK gain product 4, a cue with
raw score 32 has mean probability 0.239; **908 of 1,024 rows** have distractors
tied with the clipped cue. Deployed asinh has mean probability 0.999992 and
no cue ties in this setup. At gain product 8, every rational row has a cue
tie and mean cue probability falls to 0.00641. The study is a routing
diagnostic, not a trained-recall test. Its raw artifact is
`results/analysis/attention_link_analysis.json`.

This candidate is rejected for the current speed goal: it is slower than
asinh and has a clipping failure in the tested range. Lean certifies its
real derivative and bounded-domain properties, not its training quality.
