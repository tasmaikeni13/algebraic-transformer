import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

namespace AlgebraicTheory

theorem kernel_reciprocal_identity (x s : ℝ) (hs : s^2 = x^2 + 1) :
    (s + x) * (s - x) = 1 := by
  calc
    (s + x) * (s - x) = s^2 - x^2 := by ring
    _ = (x^2 + 1) - x^2 := by rw [hs]
    _ = 1 := by ring

theorem kernel_power_eight_identity (rho : ℝ) :
    (((rho * rho) * (rho * rho)) * ((rho * rho) * (rho * rho))) = rho^8 := by
  ring

theorem kernel_squaring_step (y : ℝ) :
    (y^2)^2 = y^4 ∧ (y^4)^2 = y^8 := by
  constructor <;> ring

theorem kernel_octa_degree (d0 d1 d2 d3 : ℕ)
    (h0 : d0 = 1) (h1 : d1 = 2 * d0) (h2 : d2 = 2 * d1) (h3 : d3 = 2 * d2) :
    d0 < d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 = 8 := by
  omega

theorem kernel_octic_composition (rho : ℝ) :
    let _k1 := rho
    let _k2 := rho^2
    let _k4 := (rho^2)^2
    let k8 := ((rho^2)^2)^2
    k8 = rho^8 := by
  dsimp
  ring

/-- The factor p(1-p) never exceeds one quarter. -/
theorem attention_diagonal_factor (p : ℝ) : p * (1-p) ≤ 1/4 := by
  nlinarith [sq_nonneg (p - 1/2)]

/-- Distinct probability masses with total at most one obey the same bound. -/
theorem attention_offdiagonal_factor (p q : ℝ) (hp : 0 ≤ p) (hq : 0 ≤ q)
    (ht : p+q ≤ 1) : p*q ≤ 1/4 := by
  nlinarith [sq_nonneg (p-q), mul_nonneg (by linarith : 0 ≤ 1-p-q) (by linarith : 0 ≤ 1+p+q)]

/-- Entrywise magnitude bound, assuming the probability-factor reduction. -/
theorem attention_entry_bound (r a : ℝ) (hr : 0 ≤ r) (hr1 : r ≤ 1)
    (ha : 0 ≤ a) (ha4 : a ≤ 1/4) : 0 ≤ 8*r*a ∧ 8*r*a ≤ 2 := by
  constructor
  · positivity
  · have h : r*a ≤ a := by nlinarith [mul_nonneg (by linarith : 0 ≤ 1-r) ha]
    linarith

/-- Exact routing ratio, before rounding to a nearby integer. -/
theorem kernel_sharpness_exact (s : ℝ) (hs : s^2 = 5) :
    (2+s)^8 = 51841 + 23184*s := by
  have h2 : (2+s)^2 = 9+4*s := by nlinarith
  have h4 : (2+s)^4 = 161+72*s := by
    calc
      (2+s)^4 = ((2+s)^2)^2 := by ring
      _ = (9+4*s)^2 := by rw [h2]
      _ = 161+72*s := by nlinarith
  calc
    (2+s)^8 = ((2+s)^4)^2 := by ring
    _ = (161+72*s)^2 := by rw [h4]
    _ = 51841+23184*s := by nlinarith

/-- A nonnegative sink leaves total token mass at most one. -/
theorem attention_sink_mass (z omega : ℝ) (hz : 0 < z) (hw : 0 ≤ omega) :
    0 < z/(z+omega) ∧ z/(z+omega) ≤ 1 := by
  have hd : 0 < z+omega := by linarith
  exact ⟨div_pos hz hd, (div_le_one hd).2 (by linarith)⟩

/-- Single-pass additive tile accumulation associativity. -/
theorem afa_additive_associativity (p1 v1 p2 v2 : ℝ) :
    (p1 * v1 + p2 * v2) = (p1 * v1) + (p2 * v2) := by
  ring

/-- Numerator-denominator scaling invariance for nonzero alpha. -/
theorem afa_scaling_invariance (o d alpha : ℝ) (ha : alpha ≠ 0) :
    (alpha * o) / (alpha * d) = o / d := by
  exact mul_div_mul_left o d ha

/-- Numerator-denominator scaling invariance for positive alpha. -/
theorem afa_scaling_invariance_pos (o d alpha : ℝ) (ha : 0 < alpha) :
    (alpha * o) / (alpha * d) = o / d := by
  have hne : alpha ≠ 0 := ne_of_gt ha
  exact mul_div_mul_left o d hne

end AlgebraicTheory

