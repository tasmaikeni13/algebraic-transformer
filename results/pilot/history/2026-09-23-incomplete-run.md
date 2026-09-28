# Incomplete training diagnostic, 23 September 2026

Commit `1d7e7741137686834cdfc85725522464afcde764` ran on 16 TPU v4 chips
across four hosts. The standard arm completed 100,000 steps in 1,328.8 seconds:
2,471,287 tokens/s, final training loss 2.9112, and held-out perplexity 25.23.
The algebraic arm stopped at step 3,500. Its measured throughput was about
2,034,000 tokens/s, or 82.3% of the standard arm. Its loss fell from 27.9953
at step 500 to 15.1393 at step 3,500, with a clipped gradient norm of 1.000
throughout the logged interval.

Profiling attributed the algebraic throughput to four full-vocabulary
projections per update in the fused loss. The incomplete algebraic arm has no
end-of-budget perplexity measurement. These values are diagnostic and are
excluded from the paper's complete-run comparison.
