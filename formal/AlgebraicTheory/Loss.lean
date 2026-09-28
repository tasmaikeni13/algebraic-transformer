import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace AlgebraicTheory

theorem pearson_chi_sq_expansion (y p : ℝ) (hp : p ≠ 0) :
    (y - p)^2 / p = y^2 / p - 2 * y + p := by
  field_simp [hp]
  ring

theorem pearson_divergence_expansion (y p : ℝ) (hp : p ≠ 0) :
    (y - p)^2 / p = y^2 / p - 2 * y + p :=
  pearson_chi_sq_expansion y p hp

theorem pearson_divergence_nonneg (y p : ℝ) (hp : 0 < p) :
    0 ≤ (y - p)^2 / p :=
  div_nonneg (sq_nonneg (y - p)) (le_of_lt hp)

theorem pearson_zero_iff_equal (y p : ℝ) (hp : 0 < p) :
    (y - p)^2 / p = 0 ↔ y = p := by
  have hp_ne : p ≠ 0 := ne_of_gt hp
  rw [div_eq_zero_iff]
  simp [hp_ne, sub_eq_zero]

theorem oace_power_chain (z : ℝ) :
    (((z^2)^2)^2) = z^8 := by
  ring

/-- The non-local power-score correction cancels the local term's gradient at
the truthful distribution.  `pinv8` and `pinv9` stand for `p⁻¹/⁸` and
`p⁻⁹/⁸`; the cache relation is stated algebraically so the certificate
does not depend on transcendental real-power machinery. -/
theorem proper_oace_gradient_zero
    (p y pinv8 pinv9 : ℝ) (hy : y = p) (hcache : p * pinv9 = pinv8) :
    pinv8 - y * pinv9 = 0 := by
  rw [hy, hcache]
  ring

/-- Expanding the corrected OACE probability gradient into its local and
non-local components is an exact ring identity. -/
theorem proper_oace_gradient_decomposition
    (y pinv8 pinv9 : ℝ) :
    pinv8 - y * pinv9 = -(y * pinv9) + pinv8 := by
  ring

end AlgebraicTheory
