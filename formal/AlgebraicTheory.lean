import AlgebraicTheory.Gate
import AlgebraicTheory.Kernel
import AlgebraicTheory.Cayley
import AlgebraicTheory.Loss
import AlgebraicTheory.Curvature
import AlgebraicTheory.Variance
import AlgebraicTheory.Composition

/-!
# The Algebraic Stack: Formal Verification in Lean 4
Authors: Tasmai Keni (tas.ken.rt25@dypatil.edu)

This library contains machine-checked formal proofs for the foundational mathematical
theorems of the Algebraic Stack:
1. `Gate`: Reflection symmetry, ALU polynomial backward pass, and inflection point alignment with GELU.
2. `Kernel`: Algebraic kernel rho reciprocal symmetry, positivity, and power-of-eight squaring chain.
3. `Cayley`: Rational Cayley transform SO(2) orthogonality, determinant +1, and shift equivariance.
4. `Loss`: Pearson chi^2 expansion, non-negativity, and OACE three-rsqrt power chain.
5. `Curvature`: Factorized Kronecker Fisher curvature recovery, debiasing, and decoupled weight decay.
6. `Variance`: AVN bounded normalization norm theorem and the Coupling Identity.
-/
