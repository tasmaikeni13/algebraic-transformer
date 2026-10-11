# Algebraic attention: exact definition, sensitivity, and numerical contracts

The deployed v2d link is `kappa(s)=rho(s/8)^16`, where
`rho(x)=x+sqrt(1+x^2)`. For a causal row, the zero-value sink is another
weight `kappa(b)`. Define `Z=sum_visible kappa(s_j)+kappa(b)`, `p_j=kappa(s_j)/Z`,
and `o=sum_visible p_j v_j`. All comparisons of implementation accuracy below
use this definition. Comparisons with softmax concern a different model.

## 1. Algebra and the online invariant

`rho(x)=exp(asinh(x))`, `rho(-x)=1/rho(x)`, and
`rho(x)rho(y)=rho(x sqrt(1+y^2)+y sqrt(1+x^2))`. Lean checks these statements
in `formal/AlgebraicAttention/Radical.lean`, including positivity and
monotonicity. The transcendental expression identifies the function; the
kernel evaluates arithmetic and radicals.

For a running maximum `m`, store

```
L = sum_processed (rho(x_j)rho(-m))^n + sink_weight_at_m
A = sum_processed (rho(x_j)rho(-m))^n v_j
```

If `m` increases to `m'`, multiply both states by
`alpha=(rho(m)rho(-m'))^n`, then add the next tile. Each processed key's
coefficient becomes `(rho(x_j)rho(-m'))^n`; normalization `A/L` therefore
agrees with the dense definition in real arithmetic. A sink starts in `L`
with a zero numerator. A causal mask excludes a key entirely, including its
contribution to `L`; assigning it a finite logit alone is insufficient.

The saved backward statistic may be `sigma=rho(-m)L^(-1/n)`, yielding
`p_j=(rho(x_j)sigma)^n`. In the unshifted case, save **`u=1/Z`** instead:
`p_j=rho(x_j)^n u`. These formulas are equivalent in real arithmetic.
The second removes three row square-root stages for n=16 and avoids their
error amplification. Lean checks both reconstruction formulas.

## 2. Differentiation and conditioning

Let `a_j=(n/c)/sqrt(1+(s_j/c)^2)`. Then

```
J_ij = p_i (delta_ij-p_j) a_j
dL/ds_j = a_j p_j (g dot v_j-g dot o)
dL/db = -a_b p_b (g dot o)
```

The last formula uses the sink's zero value. Lean proves `HasDerivAt` for
the radical/power, both normalized Jacobian entries, and the normalized
weighted-output derivative. These are derivative theorems rather than
identities that assume a derivative formula.

For `beta=n/c`, `0<a_j<=beta` and every Jacobian entry has magnitude at most
`beta/4`. Including the sink as another probability, each absolute column
sum is `2 a_j p_j(1-p_j)<=beta/2`; each row sum is at most
`2 beta p_i(1-p_i)<=beta/2`. Thus
`||J||_1, ||J||_infinity, ||J||_2 <= beta/2`. This dimension-independent
bound explains stability without implying that relative gradient error is
small near a vanishing gradient. The operator-norm argument is a written
proof; it is not yet in the Lean project.

The compressed logit has curvature
`ell''(s)=-(n/c^2)(s/c)/(1+(s/c)^2)^(3/2)`.
Large positive scores sharpen only polynomially: if a finite row contains
positive scores, scaling it by `t->infinity` sends the visible weights to
`s_j^n / sum_positive s_k^n` on positive keys. Negative-score weights decay
as `t^(-n)` and their normalized mass vanishes in that case. This limiting
argument does **not** apply unchanged to rows containing no positive score.

The deployed kernel has `beta=2`; standard softmax has `beta=1`. A comparison
with `softmax(s)` therefore mixes temperature and compression. The
Monte Carlo script also compares `softmax(2s)` with the deployed link, using
the same rows. Retrieval/training results for n=16,c=16 do not directly
establish a result for n=16,c=8.

## 3. Error propagation before rounding

Suppose every true positive weight satisfies
`(1-eps)w_j <= w_hat_j <= (1+eps)w_j`, including the sink, with `eps<1`.
Finite summation gives the same bounds for `Z_hat/Z`. Therefore

```
(1-eps)/(1+eps) <= p_hat_j/p_j <= (1+eps)/(1-eps)
TV(p_hat,p) <= eps/(1-eps)
||o_hat-o|| <= 2 M eps/(1-eps),  M=max_j ||v_j||
```

Lean checks denominator propagation and the entrywise normalized bounds.
The TV/output inequalities follow by summing the entrywise bound and the
triangle inequality; their final vector statements are not yet formalized.
No assumption of independent numerical errors is needed. If rho itself has
relative error `delta`, a weight has relative error at most
`max((1+delta)^n-1, 1-(1-delta)^n)`, approximately `n delta`.

Float32 summation, dot-product precision, bfloat16 casts before MXU products,
and the backward's statistic reconstruction add further errors. A real
polynomial certificate alone cannot certify the complete training VJP.

## 4. Range bounds and the guarded implementation

For normalized vectors with per-coordinate learned gains,

```
|s/c| <= sqrt(d) max|q_gain| max|k_gain| / c.
```

At d=64,c=8 and unit gains this is one. RMS epsilon reduces the norms;
learned gains can enlarge them. A saved proxy checkpoint has gains above
one and uses the earlier n=8,c=1 link, so it cannot establish an operating
range for the final model. Random unit-sphere rows and grown-gain rows are
analyzed separately.

The runtime guard instead checks the actual arrays:
`max_row ||q_scaled||^2 * max_row ||k||^2 <= (B/1.01)^2`, plus the sink
bound. Cauchy-Schwarz implies every kernel-unit score lies in `[-B,B]` in
real arithmetic. The 1% margin is an implementation precaution, not a
Lean proof of float32 reductions. For supported n<=32,
`B=min(16,clamp/2)`; values are bounded too so unnormalized numerators fit.
The fast branch evaluates the original link without clipping. The other
branch uses shifted stable-rho evaluation. Extreme negative rows with no
sink exercise this fallback in the TPU checks.

The corrected sufficient denominator bound is
`Z <= (T+has_sink)(2B+1)^n`; `T(2B)^n` is not a global bound. No-shift
accumulation also needs a lower-range condition to prevent whole-row
underflow and a value bound to prevent numerator overflow. An overflow
clamp alone provides neither. Lean checks the finite-sum upper bound.

The explicit clamp is constructed using the exact inverse
`rho((t-1/t)/2)=t`, which Lean also checks. Set
`t=(1e37/(T+1))^(1/n)`; then the real-arithmetic denominator budget is
`1e37`. The old `0.25*t` heuristic was unsafe for sufficiently large n.
The shifted kernel's first tile establishes the actual maximum rather than
treating a finite sentinel as a processed score. This matters for whole
negative rows below the sentinel.

Explicit unguarded `shift=False` retains the earlier upper-clipped link for
experiments. Its gradient is zero above the clip (zero subgradient at the
kink). The earlier straight-through VJP did not differentiate that primal.
Reciprocal-free rho suffers cancellation in the negative tail; a statement
that its error always affects negligible weights is false for whole
negative rows. `fast_rho=False` uses the stable reciprocal expression.

## 5. Monte Carlo and adversarial evidence

`analysis/attention_numerics.py` compares the float32 formulas with an
independent float64 oracle `softmax(16 asinh(s/8))`. Eight seeds, 128 rows,
three distributions, and three scales give **9,216 random rows**, each with
2,048 keys. It records probability errors, total variation, mass error,
VJP errors, clamping, and nonfinite rows. Sphere QK scores are sampled from
normalized 64-dimensional vectors, rather than assigned Gaussian tails.

Inverse-denominator reconstruction reduces typical numerical error. Heavy
Student-t tails expose the clipping distortion, while whole rows around
kernel-unit score -1,000 underflow in unguarded accumulation even with
stable rho. These counterexamples motivated the runtime guard. Float32 CPU
Monte Carlo does not certify TPU rounding; the separate TPU checks cover
outputs and q/k/v/sink VJPs, rectangular tiles, both data types, and tails.

The full hardware goal remains open. The mathematical definition, formal
coverage, floating-point evidence, and measured kernel speed are tracked
separately so improving one cannot silently substitute for another.

An additional standard-softmax stress test records error from saving
`m+log(l)` in float32: very large common score offsets can erase the small
normalization term and corrupt backward probability reconstruction even
when forward normalization is accurate. This is a numerical limitation of
that statistic, not a typical-score speed result. The recorded stress test
simulates the formula in NumPy; it is not a TPU verification of all standard
implementations.
