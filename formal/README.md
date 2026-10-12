# Formal coverage of algebraic attention

The project pins Lean **4.34.0-rc2** and mathlib commit
`7974e751bece493b6ff508039423ca9fa2452fa8`. These are real-arithmetic
theorems about the attention definition and its derivatives, checked by
Lean. They do not certify the Pallas compiler, IEEE rounding, or latency.

```bash
cd formal
lake exe cache get
lake build
```

| Module | Checked claims |
|---|---|
| `Radical.lean` | positivity, stable negative branch, reciprocal identity, exp/asinh equivalence, monotonicity, group law, shifted weights in (0,1], and the corrected global bound `rho(x) <= 2|x|+1` |
| `Normalization.lean` | common-factor invariance, inverse-denominator and root statistics, tile decomposition including a sink, and normalized weight bounds given relative weight/denominator error |
| `Derivatives.lean` | actual `HasDerivAt` proofs for the radical and integer-power kernel; diagonal/off-diagonal normalized Jacobian; normalized output derivative underlying the attention VJP |
| `Bounds.lean` | finite-sum Cauchy-Schwarz guard, no-shift denominator bound, and propagation of entrywise weight error to the denominator |
| `Rational.lean` | clipped-domain rational-link bounds, actual radical-free rational-power derivative, equivalent product formula, and slope bound |
| `AffinePower.lean` | actual affine-power derivative, positive base/slope bounds, and shared addition-chain identities for powers 31/32 and 63/64 |
| `Accumulation.lean` | dQ key-partition identity and movement of the common slope after partial reduction |
| `QuadraticPower.lean` | positivity, monotonicity, slope/base bounds, actual derivatives, fourth-order surplus identity, power-pair chain, and conditional Newton reciprocal-error identity |

All **52 project theorems** build successfully. The rational, affine-power, and quadratic
modules cover experimental links; those links are not adopted by the model.

The polynomial coefficients have an independent exact-rational Bernstein
certificate in `results/analysis/attention_polynomial.json`; that certificate
has **not** been imported into Lean. Its bounds concern real evaluation of
the actual float32 coefficients, before arithmetic rounding.

There are no `sorry`, `admit`, or additional axioms in the project. Formal
coverage is deliberately distinguished from the 71 TPU output/gradient
checks and Monte Carlo results. See `paper/attention_exact.md` and
`paper/attention_performance.md` for the connection to the implementation.
