# Attention diagnostics

The octic attention Jacobian bound of 2 applies entrywise to normalized
scores. It does not bound the full matrix spectral norm or derivatives with
respect to raw scores. The numerical Jacobians and counterexamples are in
`jacobians.npz` and `counterexamples.json`.

The raw-score Gaussian-noise comparison does not give a uniform 100-fold
advantage over softmax. The sink sweep in `metrics.json` and the 36
scale-and-sink combinations in `iterations/scale-sink-sweep.json` retain the
unsuccessful cases. These diagnostics should not be read as a general
quantization result.

The complete original records and their execution context are recoverable
from the Git snapshot identified in `results/record_origins.json`.
