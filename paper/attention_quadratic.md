# Quadratic-power attention candidate

This link is an experiment. The deployed model retains exact asinh attention.
It improves the static routing behavior of the earlier clipped rational and
affine candidates, but neither speed parity nor language-model quality has
been established.

## Definition and relation to asinh

Let `x=clip(s/8,-1,8)`, `b(x)=1+x+x^2/2`, and `kappa(s)=b(x)^16`.
The learned sink uses the same link. The implementation evaluates a quadratic
and four squarings; it contains no per-score radical or transcendental.

Before clipping, the exact difference from the radical is

```
b(x) - rho(x) = x^4 / [4(1+x^2/2+sqrt(1+x^2))] >= 0.
```

Thus the bases agree through cubic order at zero. Their conceptual logits
have expansions

```
16 log b(x) = 16[x - x^3/6 + x^4/8 + O(x^5)]
16 asinh(x) = 16[x - x^3/6 + O(x^5)].
```

The original attention and this candidate have origin slope two in raw
score units. This is closer locally to deployed asinh than a temperature-one
affine link, but the fourth-order surplus sharpens positive weights. The
candidate is not an exact replacement for the original operator.

For `x>=-1`, `b` is increasing, `b>=1/2`, and
`0 <= b'(x)/b(x)=(1+x)/b(x) <= 1`. The raw logarithmic slope is therefore
at most two, matching the deployed asinh bound; the normalized Jacobian's
operator norm is at most one by the argument in `attention_exact.md`.
The lower cap joins with derivative zero; outside either cap the VJP is zero.
The upper cap at raw score 64 can still create ties beyond the tested range.

On `[-1,8]`, the base is in `[1/2,41]`. Weights stay between `2^-16` and
`41^16`; for 2,048 keys plus a sink the denominator is below `1.4e29`, and
multiplying by values bounded by `1e8` stays below `1.4e37`. This is a real bound on the
specified domain, not a certification of every float32 intermediate.

Lean checks positivity, monotonicity, the exact surplus identity, actual
`HasDerivAt` derivatives, base/slope bounds, and the power-pair chain.

## Backward formulas and reciprocal experiment

The exact kernel-unit derivative is

```
ds/16 = b(x)^15 * (1+x) * (dp-di) / Z.
```

Six shared multiplications reconstruct `b^15,b^16` using
`1,2,3,6,12,15,16`. The alternative logarithmic-slope formula uses
`p*(dp-di)*(1+x)/b`. TPU probes compare full reciprocal, the Pallas
approximate reciprocal, and one Newton refinement `y <- y*(2-b*y)`.
In real arithmetic the residual after refinement is exactly
`-(b*y-1)^2`; its absolute value is at most the square of the input
residual. Lean proves this conditional statement. Hardware approximation
and float32 rounding still require measurement.

The approximate slope is a numerical approximation to the VJP. It does
not change the forward weight, and is not described as an exact gradient.

## Measurements

TPU v4, installed JAX 0.6.2, B=8,H=12,T=2048,D=64, bfloat16; nine randomized
paired rounds of 30 calls. The same resident backend is applied to both
standard and algebraic attention. Compilation and warmup are excluded.

| Kernel / VJP | Forward (ms) | Forward + q/k/v backward (ms) |
|---|---:|---:|
| Standard compact + resident | 1.565 | 3.364 |
| Deployed guarded asinh + resident | 2.232 | 5.419 |
| Quadratic, full reciprocal | 1.675 | 5.143 |
| Quadratic, shared exact power chain | 1.675 | 4.651 |
| Quadratic, approximate reciprocal | 1.675 | 4.349 |
| Quadratic, reciprocal + Newton refinement | 1.675 | 4.527 |

Float32 and bfloat16 output/q/k/v/sink checks against the independent dense
reference pass for the recorded variants. These are finite numerical checks,
not a global guarantee on the approximate derivative. Raw errors, timings,
and source hashes are in `results/kernels/attention_quadratic_reciprocal.json`.
The initial slope/product/chain experiment is archived in
`attention_quadratic_probe.json`.

## Static routing results and quality gate

Eight seeds, 128 normalized-sphere rows per seed, dimension 64, 2,048
distractors, and a cue with raw score `8*gain_product`:

| Gain product | Mean cue probability | Mean TV to deployed asinh | Cue-tie rows / 1,024 |
|---|---:|---:|---:|
| 1 | 0.99417 | 0.00427 | 0 |
| 1.5 | 0.99988 | 0.02108 | 0 |
| 2 | 0.999992 | 0.04555 | 0 |
| 4 | 0.99999996 | 0.12304 | 0 |
| 8 | 0.999999999 | 0.18852 | 0 |

Most clipping at high gain is on negative distractors; the positive cap
does not create observed cue ties in this finite study. The growing TV
shows that positive-tail sharpening changes the distribution even when
the cue probability is high. Static cue probability is not trained recall
accuracy. Learned-score diagnostics, trained retrieval, and a matched
language-model comparison remain necessary before adoption.
