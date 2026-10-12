# Theory notes (living document)

Short statements of the mathematics behind the algebraic Transformer, kept
deliberately brief because the design is still being iterated. Status tags:
**[proved]** short proof given here, **[measured]** backed by a recorded
experiment (scripts in `analysis/`, outputs in `results/`).

The attention engineering follow-up expands the mathematics and formal
coverage in [attention_exact.md](attention_exact.md), with paired hardware
measurements and rejected optimizations in
[attention_performance.md](attention_performance.md). Speed parity remains
open. The new Lean project is in [`formal/`](../formal/README.md).
Exact resident-memory dQ accumulation now improves the deployed kernel;
the real key-partition identity is checked in Lean. Alternative links are
analyzed separately in [attention_pade.md](attention_pade.md),
[attention_binomial.md](attention_binomial.md), and
[attention_quadratic.md](attention_quadratic.md). Their primitive counts,
formal derivatives, static routing, numerical VJPs, and hardware timings
are distinguished from a language-model quality result.

## 0. What "algebraic" means here

The training path (forward and backward of the model, attention, and training
head) may use `+ − × ÷`, comparisons/selection, integer powers, and `rsqrt`
(hence `sqrt x = x·rsqrt x`). No `exp, log, sin, cos, tanh, sigmoid, erf` or
non-integer powers. Evaluation (perplexity) may use `log`. The optimizer
schedule is exempt in the new study (cosine AdamW for both arms).

## 1. The radical exponential ρ

Define `ρ(x) = x + √(1+x²)`. Then `ρ(x) = e^{asinh x}`, so:

* `ρ > 0`, increasing, `ρ(0) = 1`, `ρ(−x) = 1/ρ(x)`.
* **Group law [proved].** `ρ(x)ρ(y) = ρ(x ⊕ y)` with
  `x ⊕ y = x√(1+y²) + y√(1+x²)`.
  *Proof:* `sinh(a+b) = sinh a cosh b + cosh a sinh b` and
  `cosh(asinh x) = √(1+x²)`. ∎
* Stable evaluation: `t = |x| + √(1+x²) ≥ 1`, `ρ = t` if `x ≥ 0` else `1/t`
  (no cancellation). `ρ'(x) = ρ(x)/√(1+x²)`.

`(ℝ, ⊕)` is isomorphic to `(ℝ, +)` through `sinh`, so every softmax trick
that relies on `e^{a+b} = e^a e^b` has an exact algebraic counterpart.

## 2. The asinh-power kernel κ

`κ_{n,c}(s) = ρ(s/c)^n = exp(n·asinh(s/c))`, `n` a power of two (log₂ n
squarings).

* Normalizing κ over keys is **exactly softmax of the compressed logit**
  `ℓ(s) = n·asinh(s/c)` **[proved]**.
* Slope at 0 is `n/c` (an inverse temperature); for `|s| ≫ c`,
  `ℓ(s) ≈ sign(s)·n·log(2|s|/c)` (logarithmic soft-cap).
* With `c = n`: `n·asinh(s/n) = s − s³/(6n²) + O(s⁵/n⁴)` → `exp(s)` as
  `n → ∞` **[proved, Taylor]**. The first study used `n = 8, c = 1`
  (8× sharper than softmax near 0, compressed beyond `|s| ≈ 1`).
* **Max shift [proved].** For `m ≥ x`: `(ρ(x)·ρ(−m))^n = ρ(x ⊖ m)^n ∈ (0, 1]`,
  and the common factor `ρ(−m)^n` cancels in every normalized weight. Costs
  one multiply per score; removes the overflow risk of unshifted `ρ^8`.
* **Online (flash) rescaling [proved].** When a running maximum moves from
  `m` to `m'`, previous numerators/denominators are multiplied by
  `(ρ(m)·ρ(−m'))^n`. Identical structure to FlashAttention-2.
* **Algebraic log-sum-exp [proved].** The per-row statistic
  `σ = Z^{−1/n} = ρ(−m)·l^{−1/n}` (log₂n square roots) plays the role of
  `logsumexp`: the backward recomputes `p = (ρ(x)·σ)^n`. Unshifted
  accumulation now saves `u=1/Z` and reconstructs `p=ρ(x)^n u`, avoiding
  the root chain. Both formulas are checked in Lean.
* **No-shift accumulation [proved, corrected range conditions].** If
  `|x|≤X`, including the sink, then
  `Z≤(T+has_sink)(2X+1)^n`. Underflow and numerator overflow also need
  bounds. The model checks actual input norms and value magnitudes before
  choosing unclipped accumulation; otherwise it uses shifted stable rho.
  An upper clamp alone is insufficient, and changes the link when it binds.
  Detailed conditions and counterexamples: [attention_exact.md](attention_exact.md).
* **Degree-n homogeneity ceiling [proved + measured].** For `|s| ≫ c`,
  `κ(s) ≈ (2|s|/c)^n`, so normalized weights become invariant to rescaling
  `q·k`: attention can no longer be sharpened by growing scores. Random-score
  Monte Carlo (T=1024): the v1 kernel (n=8, c=1) saturates at entropy
  `0.42·log T`, max weight 0.27, even at score SD 16 (softmax: 0.06, 0.86).
  Larger `n` (with `c ≈ n`) raises the ceiling (n=32: 0.11, 0.74).

## 3. Attention

`p_ij = κ(s_ij) / (Σ_{j'≤i} κ(s_ij') + κ(b))`, sink logit `b` (the first study
used a constant sink mass `Ω = κ(b)`).

* **Jacobian [proved].** `∂p_i/∂s_j = (n/c)·r_j·(p_i δ_ij − p_i p_j)`,
  `r = (1+(s/c)²)^{−1/2} ≤ 1`. Since `|p_i(δ_ij − p_j)| ≤ 1/4`, entries are
  bounded by `n/(4c)` (= 2 for `n=8, c=1`). Softmax's analogue has
  `r ≡ 1`: the kernel damps gradients of very large scores by `r ≈ c/|s|`.
* Sink: visible mass `1 − κ(b)/D_i`; output norm `≤ max_j ‖v_j‖`.
* Backward: `∂L/∂s_ij = (n/c)·r_ij·p_ij·(g_iᵀv_j − g_iᵀo_i)` — the
  FlashAttention-2 backward with one extra `rsqrt` per score.
* **Retrieval (measured).** One-layer associative recall with 512
  distractors (3 seeds): softmax 99.51% accuracy (loss 0.042 ± 0.011);
  v1 kernel 65% (seed failures); asinh n=16, c=16: 99.71% (loss
  0.031 ± 0.001); n=8, c=2: 99.58%; n=32, c=32: 99.54%. Lifting the
  homogeneity ceiling makes the algebraic kernel at least as precise as
  softmax.

## 4. Likelihood without logarithms

**Theorem [proved].** For an algebraic head `p_j ∝ κ_{n,c}(z_j)` the exact
log-loss gradient is algebraic:

`−log p_t = −n·asinh(z_t/c) + log Σ_j κ(z_j)` ⇒
`∂(−log p_t)/∂z_j = (n/c)·r_j·(p_j − y_j)`.

Hence an algebraic model can be trained by **exact maximum likelihood** — the
same objective it is evaluated on — without evaluating `log`. Only telemetry
needs the loss value.

**Measured (Monte Carlo).** With unconstrained logits every head (softmax,
first-study, asinh) represents Zipf-like targets over 50,257 tokens to
KL ≤ 5·10⁻⁴: there is no representability floor. On a misspecified linear
student (12 trials),
training the v1 link on OACE instead of log-loss costs 0.0289 ± 0.0009 nats of
test cross-entropy; the asinh link (n=c=16) is 0.00226 ± 0.00008 nats *better*
than softmax at equal rank.

**Rank bottleneck [proved + measured].** With `z = Wh`, `h ∈ ℝ^d`, softmax
log-probabilities lie in a rank-(d+1) family. `n·asinh(z/c)` is an entrywise
non-affine map, so its image is not confined to that family (as for
sigsoftmax). Fitting a full-rank teacher with rank-d logits (6 trials), the
asinh link with n=8, c=1 lowers the minimal KL relative to softmax by 7.7%
(d=8), 11.7% (16), 15.4% (32), 18.1% (64); paired SE ≤ 0.0013. Stronger
compression (smaller c) helps more — matching the language-model ablations
where the head improved monotonically from CE to n=8, c=2.

**The first study's power score (OACE)** `L = γ[8 p_t^{−1/8} + (8/7)Σp^{7/8}
− 64/7]` is strictly proper (Bregman of `−Σp^{7/8}`), but its gradient
reweights tokens relative to log-loss by roughly `p_t^{−1/8}` (×2.4 at
`p_t = 10⁻³`, ×1.09 at `p_t = 0.5`). Properness fixes the optimum only for a
well-specified model; a finite model trained on OACE trades average
log-loss for hard-token fit. OACE also normalizes logits across the
vocabulary (`ε_vocab = 100`), coupling all 50,257 logits **[measured: +0.080
nats on the standard model; replacing it by CE inside v1 recovers 0.12]**.

## 5. Positions: Cayley rotations

`R(w) = [[1−w², −2w],[2w, 1−w²]]/(1+w²)` is a rotation by `θ = 2·atan w`
(orthogonal, det 1) and `⟨R^m q, R^n k⟩ = qᵀR^{n−m}k` **[proved]**.

Frequency grids (head dim 64, base 10⁴; pairs completing a turn within L
tokens):

| grid | L=16 | L=64 | L=256 | L=2048 |
|---|---|---|---|---|
| first study `w_k=(1+αk)^{-2}` | 1 | 2 | 3 | 8 |
| RoPE `θ_k = b^{-k/32}` | 4 | 9 | 13 | 21 |
| geometric Cayley `w_k = ½·b^{-k/32}` | 4 | 9 | 13 | 21 |

Position-only selectivity (LP: best margin `s(Δ*) − max_{Δ≠Δ*} s(Δ)` with
`Σ|a_k|+|b_k| ≤ 1`): v1 grid 0.13–0.15 for Δ*≤3 and ≈0.06 for Δ*≥8; RoPE
0.23–0.30; geometric Cayley 0.21–0.28. **Yet at the proxy scale the v1 grid
was not harmful (−0.006 nats)**: its 24 near-static pairs behave like
partial rotary dimensions.

`b^{−1/32}` is built from five nested square roots, so the geometric grid is
algebraic. Tables use binary exponentiation of the unit complex number
`(1+iw)²/(1+w²)` (error `O(log T·ε)` instead of the `O(T·ε)` scan). The
hypothesis that the first-study grid starves local attention was **rejected**
by the ablation; the geometric grid was chosen for v2d (it was 0.007 better
under QK-norm, one seed).

## 6. Activation

`ALU_β(x) = (x/2)(1 + βx/√(1+β²x²))`,
`ALU_β'(x) = ½(1+u) + ½·u·r²` with `u = βx·r`, `r = (1+β²x²)^{−1/2}`.
Gate slope at 0 is `β/2`; `β = ½` matches SiLU, `β ≈ 0.8` GELU, `β = 1` was
used before. Negative tail decays like `−1/(4β²|x|)` (heavier than SiLU's
exponential tail). Gaussian-weighted L2 distance to SiLU is smallest at
`β ≈ 0.6` **[measured: β=1 → ½ improved v2c by 0.012]**.

## 7. Normalization

AVN is RMSNorm without gain. A per-channel gain is a multiplication (still
algebraic); removing it removes per-channel scale control, notably the
final-layer logit temperature (the standard model learned a final gain with
mean square ≈ 2.1) **[measured: removing gains costs 0.037 on the standard
model; restoring them in v1 recovers 0.10]**.

## 8. Schedule

ARDS `η_t = η·(1+((t−T_w)/(0.8T))²)^{−1/2}` ends at **0.634·η** and averages
0.835·η over the 125M run; the cosine arm ends at 0.05·η and averages
0.524·η. No annealing phase means the algebraic arm finished with ~12×
higher terminal learning rate **[measured: ARDS + wd 0.01 cost 0.060 at the
proxy scale; an annealing-law fit predicts 0.115 nats at 125M]**.

## 9. Attribution of the first-study gap (Phase 1, measured)

Proxy model (33M, 197M tokens, 2M held-out tokens; seed SD ≈ 0.003 nats).
Each v1 component swapped alone into the standard model:

| v1 component | Δ NLL |
|---|---|
| OACE head (power score + vocabulary AVN) | **+0.080** |
| ARDS schedule + wd 0.01 | **+0.060** |
| AVN without gains | **+0.037** |
| octic attention (n=8, c=1, Ω=1) | +0.003 |
| ALU-GLU (β=1) | +0.001 |
| v1 Cayley grid | −0.008 |

Full v1: +0.14. Reverse check: replacing OACE by CE inside v1 recovers
0.12. H4 (objective/head), H6 (schedule), H1 (gains) are confirmed;
H2, H3, H5 are rejected: the genuinely algebraic substitutions cost nothing.
At 125M the annealing-law fit predicts ARDS alone costs 0.115 nats
(90% CI 0.087–0.185), larger than the observed 0.009 gap, i.e. the v1
architecture was otherwise competitive.

## 10. Second-generation design (v2d, selected by Phase 1)

| component | choice | why |
|---|---|---|
| head | `p ∝ ρ(z/2)^8`, exact log-loss gradient | trains on the evaluated metric; strong compression breaks the softmax rank bottleneck (−0.035 nats on a standard backbone) |
| attention | `ρ(s/8)^16`, learned sink, QK-norm | softmax-like for `|s|≲8`, soft-capped beyond; n=16 lifts the homogeneity ceiling; QK-norm keeps scores in range |
| positions | geometric Cayley grid | RoPE-equivalent spectrum, algebraic construction |
| FFN | ALU-GLU, β = ½ | matches SiLU's gate slope; β = 1 costs 0.012 |
| norm | RMS with learned gains | gains were worth 0.04–0.10 nats |
| optimizer | AdamW, cosine | the user-specified common optimizer |

Proxy result: v2d 3.7574 ± 0.0016 vs standard 3.8182 ± 0.0029
(paired −0.061 ± 0.004, t = −14). With the same head and QK-norm on a
standard backbone the two backbones tie (−0.004 ± 0.005).

## 11. Phase 3: 60M parameters, 1B tokens (measured)

FineWeb-Edu-1B, context 2048, 1,908 steps of 524,288 tokens, cosine AdamW,
one seed, 8M held-out tokens. Learning rates tried per arm:
{1.5, 2.12, 3, 6}·10⁻³ (standard), {1.5, 3, 4.24, 6}·10⁻³ (algebraic),
{1.5, 3, 6}·10⁻³ (hybrid; 3.4130, 3.3716, 3.3950).

| arm | best LR | NLL | perplexity | tokens/s (16 chips) |
|---|---|---|---|---|
| standard | 3e-3 | 3.4170 | 30.48 | 2.70M |
| **algebraic (v2d)** | 3e-3 | **3.3697** | **29.07** | 1.94M |
| hybrid: standard + QK-norm + algebraic head | 3e-3 | 3.3716 | 29.13 | 2.38M |

Algebraic − standard = −0.047 nats (perplexity ratio 0.954); every algebraic
run (worst: 3.4058 at 1.5e-3) beats the standard's best. The algebraic NLL is
flat over 3–6·10⁻³ (3.3697–3.3750) where the standard degrades sharply at
6·10⁻³ (3.4992). The advantage is nearly uniform over context positions
(−0.04 to −0.05 nats from position 4 on, −0.049 at 1024–2048) and slightly
negative only at position 0 (+0.011, the context-free unigram prediction).
As at the proxy scale, the algebraic backbone ties a standard backbone that
carries the same head and QK-norm (the hybrid: 3.3697 vs 3.3716).

## 12. How much algebra do we really need? (measured)

Algebraic pieces added to the standard model one at a time (proxy scale,
3–4 paired seeds per rung; Δ NLL vs standard with t 95% CIs, throughput on
one chip):

| rung | Δ NLL | share of full gain | tokens/s |
|---|---|---|---|
| standard | 0 | 0% | 333K |
| + algebraic head | −0.036 ± 0.007 | 59% | 317K |
| (QK-norm alone) | −0.029 ± 0.002 | 47% | 309K |
| **+ QK-norm = hybrid** | **−0.059 ± 0.006** | **97%** | **294K** |
| + sink | −0.055 ± 0.005 | 91% | 294K |
| + ALU-GLU | −0.057 ± 0.004 | 94% | 286K |
| + Cayley rotations | −0.056 ± 0.001 | 92% | 286K |
| + asinh attention (fully algebraic) | −0.061 ± 0.004 | 100% | 249K |

Two algebraic pieces — the likelihood head `ρ(z/c)^n` (whose log-loss
gradient is algebraic) and the rsqrt QK-normalization — carry essentially the
whole gain. The remaining substitutions are quality-neutral (their intervals
overlap) and exist to remove every transcendental from the training path; the
asinh attention kernel is the only one that costs noticeable throughput.

