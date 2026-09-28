import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

namespace AlgebraicTheory

theorem gate_reflection_identity (u : ℝ) :
    (1 / 2 : ℝ) * (1 + u) + (1 / 2 : ℝ) * (1 - u) = 1 := by
  ring

theorem alu_reflection_symmetry (u : ℝ) :
    (1 / 2 : ℝ) * (1 + u) + (1 / 2 : ℝ) * (1 - u) = 1 := by
  ring

theorem alu_cache_invertibility (x s u : ℝ) (hs : s^2 = x^2 + 1) (hu : u * s = x) :
    (1 - u^2) * s^2 = 1 := by
  calc
    (1 - u^2) * s^2 = s^2 - (u * s)^2 := by ring
    _ = (x^2 + 1) - x^2 := by rw [hs, hu]
    _ = 1 := by ring

theorem alu_polynomial_backward_identity (u : ℝ) :
    (1 / 2 : ℝ) * (1 + u) + (1 / 2 : ℝ) * (u - u^3) =
    (1 / 2 : ℝ) * (1 + 2 * u - u^3) := by
  ring

theorem alu_deriv_formula (u : ℝ) :
    (1 + u) / 2 + (u * (1 - u^2)) / 2 =
    (1 / 2 : ℝ) * (1 + 2 * u - u^3) := by
  ring

theorem alu_inflection_identity (u : ℝ) (hu : 3 * u^2 = 2) :
    2 - 3 * u^2 = 0 := by
  linarith

theorem alu_inflection_x_to_u (x_sq s_sq u_sq : ℝ)
    (hx : x_sq = 2) (hs : s_sq = x_sq + 1) (hu : u_sq * s_sq = x_sq) :
    u_sq = 2 / 3 := by
  have hs_val : s_sq = 3 := by linarith
  rw [hs_val] at hu
  linarith

-- The iff includes BOTH inflections; the earlier prose omitted the positive one.
theorem alu_inflection_iff (x s u : ℝ)
    (hs : s ^ 2 = x ^ 2 + 1) (hu : u * s = x) :
    2 - 3 * u ^ 2 = 0 ↔ x ^ 2 = 2 := by
  have h := alu_cache_invertibility x s u hs hu
  have hsq : u ^ 2 * s ^ 2 = x ^ 2 := by nlinarith [sq_nonneg (u * s - x)]
  constructor <;> intro hx <;> nlinarith

-- Conjugate forward evaluation used to preserve the negative tail.
theorem alu_negative_tail_identity (u r : ℝ)
    (h : u ^ 2 + r ^ 2 = 1) (hu : 1 - u ≠ 0) :
    1 + u = r ^ 2 / (1 - u) := by
  apply (eq_div_iff hu).2
  nlinarith

-- Rational upper certificate for the Horner derivative on the cache interval.
theorem alu_derivative_bound (u : ℝ) (hl : -1 ≤ u) (hr : u ≤ 1) :
    -(1 / 20 : ℝ) ≤ (1 / 2 : ℝ) + u * (1 - u ^ 2 / 2) ∧
    (1 / 2 : ℝ) + u * (1 - u ^ 2 / 2) ≤ 21 / 20 := by
  have upper := mul_nonneg (sq_nonneg (u - 4 / 5)) (show 0 ≤ u + 8 / 5 by linarith)
  have lower := mul_nonneg (sq_nonneg (u + 4 / 5)) (show 0 ≤ -u + 8 / 5 by linarith)
  constructor <;> nlinarith

end AlgebraicTheory
