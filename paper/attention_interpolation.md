# Direct interpolation of the deployed algebraic attention weight

This is a rejected performance experiment. It approximates the deployed
weight `W(x)=(x+sqrt(1+x^2))^16`, rather than selecting a new attention link.
The current model does not use it. All coefficient generation uses radicals,
integer powers, and arithmetic; the kernel evaluates table lookup and Horner
polynomials. The experiment contains no exponential or logarithm.

## Construction and checked identities

The domain `[-4,4]` has 128 bins of width `h=1/16`. At both endpoints of
each bin, compute

```
W' = 16 W / sqrt(1+x^2)
W'' = W (256/(1+x^2) - 16x/(1+x^2)^(3/2)).
```

The degree-five Hermite polynomial matches the value, first derivative,
and second derivative at the two endpoints in exact arithmetic. Its local
argument is `t=(x-a)/h`. Given `c0=W(a)`, `c1=h W'(a)`, and
`c2=h^2 W''(a)/2`, let

```
f = W(b)-c0-c1-c2
d = h W'(b)-c1-2c2
dd = h^2 W''(b)-2c2
c3 = 10f-4d+dd/2
c4 = -15f+7d-dd
c5 = 6f-3d+dd/2.
```

Lean checks the Horner identity, actual polynomial derivatives, and endpoint
value/slope identities. The independently rounded float32 coefficients can
introduce small jumps at bin boundaries. The prototype VJP differentiates
the selected polynomial and ignores the selection boundary; it is an
approximation to the original smooth derivative, not a proof of a global
derivative for the rounded piecewise function.

A deterministic Cauchy-Schwarz guard, including the sink and value bound,
selects interpolation only when all scores fit the domain with margin.
Out-of-range inputs use the existing shifted, stable radical kernel.

## Exact rational error calculation

`analysis/attention_interpolation_certificate.py` encloses each endpoint
radical between adjacent fractions with denominator `2^90`. Interval
arithmetic then encloses the exact Hermite coefficients and measures their
distance from the actual stored float32 numbers.

Set `r=(1+x^2)^(-1/2)` and `F_k=W^(k)/W`. The recurrence is
`F_(k+1)=D F_k+16 r F_k`, with
`D(x^a r^b)=a x^(a-1) r^b-b x^(a+1) r^(b+2)`. It gives

```
F_6 = 15482880 r^6 - 14933520 x r^7 + 6693120 x^2 r^8
      - 1703520 x^3 r^9 + 241920 x^4 r^10 - 15120 x^5 r^11.
```

For each bin, bound `|W^(6)|` by `M6` using monotonicity of W and endpoint
radical bounds. The classical Hermite remainder gives
`|W-H| <= M6 h^6/(720*64)` because `t^3(1-t)^3 <= 1/64`.

For the derivative, subtract the degree-five Taylor polynomial centered
at the bin midpoint. Its residual, first derivative, and second derivative
are bounded by `M6(h/2)^6/720`, `M6(h/2)^5/120`, and
`M6(h/2)^4/24`. The sums of the absolute derivatives of the two endpoint
Hermite bases are at most `15/4`, `1`, and `1/10` for value, first-derivative,
and second-derivative data. These three rational bounds are checked by exact
Bernstein bounds on 448 subintervals. Applying the Hermite operator to the
Taylor residual gives

```
|W'-H'| <= M6 h^5 * 53/61440.
```

Add coefficient errors `sum |delta c_i|` for the value and
`sum i |delta c_i|/h` for the derivative. Divide by the bin's positive
lower bounds for W and W'. The resulting uniform real-evaluation bounds
for the stored float32 coefficients are:

| Quantity | Relative bound |
|---|---:|
| Weight | 0.000061412 |
| Selected polynomial derivative | 0.00248512 |

The calculation assumes the stated classical Hermite/Taylor remainder
theorems and analytic recurrence. It has not been imported into Lean and
does not certify IEEE evaluation, gather lowering, or the Pallas compiler.
Raw intervals and assumptions are in
`results/analysis/attention_interpolation_certificate.json`.

## Numerical checks and performance rejection

On 524,289 float64 evaluation points, using the actual float32 coefficients,
the maximum weight and derivative relative errors are `2.224e-5` and
`9.455e-5`. TPU float32 output and q/k/v/sink VJP checks pass; the largest
gradient relative L2 error is `7.22e-5`. BF16 checks also pass. The cubic
candidate fails the float32 sink-gradient gate and is not timed.

The installed Mosaic compiler requires coefficients as explicit Pallas
inputs and native dynamic gathers in 8x128 register shapes. The prototype
splits larger tiles into these shapes. This compiles, but the lookup costs
more than the radical computation it replaces:

| 125M attention shape | Forward (ms) | Forward + q/k/v backward (ms) |
|---|---:|---:|
| Standard resident | 1.565 | 3.364 |
| Deployed guarded asinh | 2.232 | 5.419 |
| Quintic lookup | 4.820 | 11.098 |

These are nine randomized paired rounds of 30 calls on one TPU v4 chip.
The successful record is `results/kernels/attention_interpolation_vreg.json`;
earlier files preserve the constant-capture and gather-shape failures.
The result rejects this lookup implementation as a speed optimization.
