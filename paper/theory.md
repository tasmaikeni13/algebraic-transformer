# Algebraic Transformer

The pure algebraic model replaces softmax attention, SwiGLU, and trigonometric positional rotations with arithmetic and reciprocal-root operations. Its output head has the exact gradient of a normalized maximum-likelihood objective. The standard model uses RMSNorm, SwiGLU, RoPE, softmax attention, and cross-entropy.

## Positive link and stable normalization

Define

$$\rho(x)=x+\sqrt{1+x^2},\qquad \rho(-x)=\rho(x)^{-1}.$$

For negative inputs, the implementation evaluates the reciprocal of $|x|+\sqrt{1+x^2}$ to avoid cancellation. Square roots in the model are implemented using reciprocal square roots and multiplication.

The identity

$$\rho(x)\rho(y)=\rho\!\left(x\sqrt{1+y^2}+y\sqrt{1+x^2}\right)$$

allows normalization by a shared maximum without evaluating a logarithm or exponential. Positive integer powers use repeated multiplication. The analytical equivalence $\rho(x)=\exp(\operatorname{asinh}(x))$ is useful for reference calculations and held-out scoring; those functions are absent from the algebraic model objective and its gradient.

## Attention

For causal scores $s_{ij}=q_i^\top k_j/\sqrt{d_h}$ and a learned per-head sink logit $b$, use

$$a_{ij}=\frac{\rho(s_{ij}/8)^{16}}{\sum_{k\le i}\rho(s_{ik}/8)^{16}+\rho(b/8)^{16}},\quad j\le i.$$

The sink has a zero value vector, so the visible weights may sum to less than one. Queries and keys receive learned-gain RMS normalization before the positional rotation. The positive link is monotone; for $u(s)=\rho(s/c)^n$,

$$u'(s)=\frac{n}{c}\frac{u(s)}{\sqrt{1+(s/c)^2}}.$$

Pallas implements tiled causal attention and its query, key, value, and sink gradients. A range guard selects stable normalization for extreme scores; ordinary scores use compact inverse-denominator statistics. Resident accumulation is selected for supported model shapes. Exact identities, numerical guards, and formal coverage are documented in [attention_exact.md](attention_exact.md) and [../formal/README.md](../formal/README.md).

## Maximum-likelihood head

With tied-embedding logits $z_j$,

$$p_j=\frac{\rho(z_j/2)^8}{\sum_k\rho(z_k/2)^8}.$$

For a target distribution $y$,

$$\frac{\partial[-\sum_k y_k\log p_k]}{\partial z_j}=\frac{4(p_j-y_j)}{\sqrt{1+(z_j/2)^2}}.$$

A custom VJP returns this gradient without computing log loss in the differentiated model. Its forward scalar is a surrogate; it must not be interpreted as NLL. The trainer separately converts stop-gradient target probabilities to NLL for reporting. Held-out evaluation scores the normalized distribution directly.

The dense head and fused projection/head use a finite-range guard to prevent float32 overflow. It does not bind for ordinary logits; beyond that guard the straight-through gradient is an engineering approximation. The exact normalized-link reference and numerical gradient checks are retained in the component tests. The fused TPU head tiles the vocabulary and recomputes projection tiles in backward. Its default 128-token by 512-vocabulary tiles fit the 350M and 1B widths.

## Positional rotations, normalization, and activation

Cayley rotations use geometric frequencies $w_k=\tfrac12\,10000^{-k/(d_h/2)}$. A unit complex rotation is represented by

$$C(w)=\frac{1-w^2}{1+w^2},\qquad S(w)=\frac{2w}{1+w^2}.$$

Position powers use binary exponentiation and reciprocal-root renormalization. No trigonometric function is evaluated. The supported presets use head dimension 64, so geometric frequency ratios can be computed using repeated square roots.

Learned RMS normalization is $g\odot x/\sqrt{\operatorname{mean}(x^2)+\epsilon}$. The algebraic activation, used within a gated feed-forward block, is

$$\operatorname{ALU}(x)=\frac{x}{2}\left(1+\frac{\beta x}{\sqrt{1+\beta^2x^2}}\right),\qquad\beta=\tfrac12.$$

## Scope of the primitive claim

`scripts/audit_algebraic.py` traces the model objective gradient, including nested Pallas bodies, and rejects exponential, logarithmic, trigonometric, hyperbolic, sigmoid, error-function, and general-power primitives. Parameter initialization, AdamW's learning-rate schedule, reporting, and evaluation are outside this model-gradient claim. In particular, the shared cosine schedule uses a cosine, and NLL/perplexity reporting uses logarithms/exponentials.

## Evidence

The 350M / 7B-token and 1B / 20B-token configurations each use one paired training seed. Their complete post-training evaluation protocols are implemented; full training and benchmark results remain to be run by the author. Synthetic smoke measurements verify execution, not learned quality.
