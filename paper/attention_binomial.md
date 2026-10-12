# Affine-power attention candidate

The rational Padé probe passed output/VJP checks but was slower than the
deployed asinh kernel. Reusing one reciprocal improved its backward time,
but still left the updated adapter at 6.52 ms against 5.94 ms for asinh and 3.97 ms for
standard. The next candidate removes both per-score division and radicals.
It is a different link and is not adopted by the model.

## Definition

`x=clip(s/c,l,u)`, `kappa(s)=(1+x)^n`, with `-1<l<u` and n a power of two.
Near zero its conceptual logit is
`n log(1+s/c)=(n/c)s - n s^2/(2c^2)+O(s^3/c^3)`.
For c=n this approaches standard softmax as n grows, with a quadratic
compression term. It damps positive scores more than standard softmax and
can amplify slopes on negative scores. It is not translation invariant.

The candidates are n=c=32, `[l,u]=[-3/4,3/4]`, and n=c=64,
`[l,u]=[-1/2,3/4]`. Both have origin slope one. A third candidate n=64,c=32
matches the deployed asinh link's origin slope two. Its positive-score slope
matches asinh's at raw score 64/15, in the range that receives appreciable
attention; this motivates measuring it separately. The n=64 negative cap is
chosen so the entire row's smallest raw weight is `2^-64`, a normal float32
value; a symmetric -3/4 cap would produce `2^-128` and risk whole-row
underflow. Its largest raw weight is `(7/4)^64`, about `3.57e15`, so the
denominator at 2,048 keys remains far below float32 overflow. Ordinary
value magnitudes still need to fit the unnormalized numerator range.

## Exact derivative and multiplication-only score evaluation

Inside the cap,

```
kappa'(s) = (n/c)(1+s/c)^(n-1)
d log kappa / ds = (n/c)/(1+s/c)
p = (1+x)^n / Z
ds/n = (1+x)^(n-1) (dp-di) / Z.
```

The last formula uses kernel-unit scores; the frontend applies 1/c and
1/sqrt(d). It reuses the power chain rather than dividing every probability
by `1+x`. With n=32 the weight takes five squarings and the derivative uses
four additional product stages. n=64 adds one stage to each. Division
remains only in row normalization. This is a speed hypothesis to measure.

For n=c=64 and x>=-1/2, the logarithmic slope is at most two and the
normalized Jacobian operator norm is at most one. The clipping VJP is zero
outside the interval, with zero chosen at each boundary. The generic
normalization/derivative proofs apply to this positive polynomial link.

For the n=64 backward, a shared addition chain computes both powers in nine
multiplications instead of eleven and keeps fewer power arrays live:
`1,2,3,6,7,14,21,42,63,64`. The n=32 pair uses eight instead of nine.
Lean checks both reconstruction identities. Fewer operations alone do not
prove a speedup: the chain also delays the weight used for dV. The probe
compares it with the separate power chains and the reciprocal-slope formula.
An exact `rsqrt(base)^2` slope variant tests whether avoiding a native division
is faster; it uses a radical in backward and does not change the link.

## Quality gate

The cap can create ties, and sufficiently large common offsets can flatten
an entire head. The affine offset also changes the effective temperature.
The wider raw thresholds (negative -32, positive 48 for n=64) reduce these
risks relative to the tested rational link but do not prove model quality.
Monte Carlo uses temperature-matched standard weights, normalized-sphere
distractors, and grown gains; learned-score ranges and language-model NLL
must be checked before adoption. Static cue probability is not trained
retrieval accuracy.

## Hardware and routing results

Nine randomized paired rounds, 30 calls each, on the same TPU v4/JAX 0.6.2
125M attention shape give the following exploratory measurements. All
listed links passed float32 output/q/k/v/sink checks against an independent
dense reference. They have not received language-model training checks.

| Link | Forward (ms) | Slope VJP (ms) | Product VJP (ms) | Shared-chain VJP (ms) |
|---|---:|---:|---:|---:|
| Standard compact | 1.564 | 3.972 | — | — |
| Guarded asinh n=16,c=8 | 2.231 | 5.942 | — | — |
| Affine n=32,c=32 | 1.597 | 5.388 | 5.089 | **5.033** |
| Affine n=64,c=64 | 1.695 | 5.567 | 5.413 | **5.232** |
| Affine n=64,c=32 | 1.695 | 5.566 | 5.415 | **5.236** |

Forward is close to standard for these candidates, but the fastest combined
forward/backward remains 27% slower. The shared chain improves measured
backward time; reciprocal-square-root slopes lose to it. Samples and source
hashes are in `results/kernels/attention_link_probe.json`.

The eight-seed static normalized-sphere study gives:

| Link | Cue probability, gain 1 | Gain 4 | Gain 8 | Cue-tie rows at gain 8 / 1,024 |
|---|---:|---:|---:|---:|
| Deployed asinh | 0.98994 | 0.999992 | 0.999996 | 0 |
| Affine n=32,c=32 | 0.27867 | 0.992901 | 0.15346 | 908 |
| Affine n=64,c=64 | 0.36185 | 0.999995 | 0.999992 | 0 |
| Affine n=64,c=32 | 0.99215 | 0.999992 | 0.26445 | 908 |

These cue probabilities are not retrieval accuracies. The n=64,c=32 link
matches the original link's origin temperature and is plausible at moderate
gain, but its positive cap produces ties at gain 8. The n=64,c=64 link has
a wider raw operating range and a different temperature. These tradeoffs
are explicit reasons to retain the current model while improving the exact
backward schedule. Raw rows are in
`results/analysis/attention_link_analysis.json`; the study reports entropy,
temperature-matched KL, clipping frequency, and ties for all five gains.

Applying the resident backward to every baseline and candidate gives 3.364 ms
for standard, 5.419 ms for deployed asinh, 4.414 ms for affine n=32,c=32,
and 4.616 ms for either n=64 affine variant with shared powers. These paired
results are in `results/kernels/attention_link_resident.json`. The relative
gap to the newly improved standard remains 31% for the fastest affine case.
The language-model quality gate still applies.
