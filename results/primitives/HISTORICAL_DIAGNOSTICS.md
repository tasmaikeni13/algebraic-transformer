# Primitive diagnostics

These measurements characterize limits of the normalization and activation
implementations. They are retained as diagnostics and are not used as training
outcomes in the paper.

At input scale $\sigma=0.1$ with normalization regularizer $10^{-5}$, the
measured centered variance was 0.9990019521851182. The normalization controls
the raw second moment; it does not guarantee unit centered variance. The
underlying record is `iterations/variance-v1.json`.

In 200 non-residual network trials per depth at width 64, the gradient range
extended from 0.0332 to 16.6553 (`iterations/deep-unattenuated.json`). A
10,000-trial width-64 study passed its gradient checks but reached a maximum
activation-variance ratio of 2.746072
(`iterations/full-cpu-initial/metrics.json`). These are finite-width results,
not guarantees for arbitrary stacks.

The numerical arrays for these diagnostics remain in the adjacent `.npz`
files. The complete original records and their execution context are
recoverable from the Git snapshot identified in `results/record_origins.json`.
