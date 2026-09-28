# Can Algebra Replace Transcendentals in Language Models?

This repository contains a JAX causal language model built from arithmetic and
inverse-square-root operations in its training path, a standard Transformer
reference model, TPU attention and loss kernels, Lean 4 lemmas, and the
measurements used in a [preprint](paper/main.pdf).

The question tested here is concrete: can this particular algebraic model train
at language-model scale and approach the held-out performance of a matched
standard Transformer? At roughly 125 million parameters, three runs per model
processed 2,500,853,760 FineWeb-Edu tokens each. Mean validation perplexity
was **25.774 ± 0.084** for the algebraic model and **25.548 ± 0.043** for the
standard model (mean ± standard error across seeds 42, 43, and 44). The
algebraic model was about 4% slower on the measured 16-chip TPU v4 setup.
See the paper for the paired uncertainty interval, four zero-shot tasks, the
20M-scale WikiText-103 experiment, and limitations.

## Code and data

| Path | Contents |
| --- | --- |
| [`src/`](src/) | Model architectures, algebraic primitives, optimizer, datasets, and kernels |
| [`tests/`](tests/) | Numerical references and regression tests |
| [`scripts/`](scripts/) | Dataset preparation, training, verification, and plotting commands |
| [`configs/`](configs/) | Prespecified hyperparameter candidates |
| [`formal/`](formal/) | Lean 4 certificates and their stated scope |
| [`results/`](results/) | Recorded measurements, run ledgers, and numerical diagnostics |
| [`paper/`](paper/) | LaTeX source, figures, and compiled preprint |

The production algebraic modules use no explicit exponential, logarithmic,
trigonometric, hyperbolic, sigmoid, or non-integer-power calls. The standard
model and evaluation code use their usual functions. The source audit is a
check on selected Python modules; it does not establish compiler-wide or
hardware-wide absence of transcendental instructions.

## Dependencies

The two requirements files serve different machines:

| File | Install on | Contents |
| --- | --- | --- |
| [requirements.txt](requirements.txt) | Local CPU development and dataset preparation | Shared JAX, test, plotting, tokenization, and Parquet dependencies |
| [requirements-tpu.txt](requirements-tpu.txt) | TPU workers | Includes the shared file and adds the pinned TPU runtime package |

Install requirements.txt locally. The distributed launchers install
requirements-tpu.txt on every TPU worker. PyArrow, needed for FineWeb-Edu
preparation, is now pinned in the shared file; the launchers no longer install
it separately. These files describe a fresh environment. The archived
hardware records retain the source and environment hashes of their original
runs and are not relabeled as current hardware evidence after dependency
changes.

## Reproduce local checks

Create an environment with Python 3.10 or newer:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest -q
cd formal && lake build
```

The CPU tests check the implementation and numerical references. They do not
repeat the TPU training runs. The packaged records include source hashes and
historical origin metadata for the private development archive; the public
release does not expose that development history. A fresh hardware run from
this public snapshot is required for current-source evidence.

The tracked PDF is the publication artifact. Manuscript source and
figure-generation files are retained in the private development archive.

The dataset caches and training checkpoints are large and are not part of this
repository. The six run records and benchmark measurements are under
[`results/pretraining/`](results/pretraining/).
