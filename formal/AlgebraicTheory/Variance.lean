import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Analysis.Real.Sqrt

namespace AlgebraicTheory

theorem avn_bounded_norm (S d eps : ℝ) (_hS : 0 ≤ S) (hd : 0 < d) (heps : 0 ≤ eps) :
    S ≤ d * (S / d + eps) := by
  have hd_ne : d ≠ 0 := ne_of_gt hd
  have h1 : d * (S / d + eps) = S + d * eps := by
    calc
      d * (S / d + eps) = d * (S / d) + d * eps := by ring
      _ = S + d * eps := by rw [mul_div_cancel₀ S hd_ne]
  rw [h1]
  have h2 : 0 ≤ d * eps := mul_nonneg (le_of_lt hd) heps
  linarith

theorem avn_coupling_identity (x v tau : ℝ) (htau : tau^2 * v = 1) :
    (tau * x)^2 + 1 = tau^2 * (x^2 + v) := by
  calc
    (tau * x)^2 + 1 = tau^2 * x^2 + 1 := by ring
    _ = tau^2 * x^2 + tau^2 * v := by rw [htau.symm]
    _ = tau^2 * (x^2 + v) := by ring

theorem bounded_avn_norm (S d eps : ℝ) (hS : 0 ≤ S) (hd : 0 < d) (heps : 0 ≤ eps) :
    S ≤ d * (S / d + eps) :=
  avn_bounded_norm S d eps hS hd heps

theorem avn_scale_invariance (x tau α : ℝ) (hα : α ≠ 0) :
    (tau / α) * (α * x) = tau * x := by
  calc
    (tau / α) * (α * x) = ((tau / α) * α) * x := by ring
    _ = tau * x := by rw [div_mul_cancel₀ tau hα]

-- Real sqrt coordinate invariance, rather than an assumed scaling of tau.
theorem avn_positive_scale_invariance (x v α : ℝ) (hα : 0 < α) :
    (α * x) / Real.sqrt (α ^ 2 * v) = x / Real.sqrt v := by
  rw [Real.sqrt_mul (sq_nonneg α), Real.sqrt_sq (le_of_lt hα)]
  exact mul_div_mul_left x (Real.sqrt v) (ne_of_gt hα)

theorem avn_regularized_moment (m eps : ℝ) (h : m + eps ≠ 0) :
    m / (m + eps) = 1 - eps / (m + eps) := by
  field_simp
  ring

theorem avn_gate_coupling (x v tau : ℝ) (htau : tau ^ 2 * v = 1)
    (hpos : 0 < tau) :
    (tau * x) / Real.sqrt ((tau * x) ^ 2 + 1) = x / Real.sqrt (x ^ 2 + v) := by
  rw [avn_coupling_identity x v tau htau]
  exact avn_positive_scale_invariance x (x ^ 2 + v) tau hpos

theorem avn_centered_variance (m mu tau : ℝ) :
    tau ^ 2 * m - (tau * mu) ^ 2 = tau ^ 2 * (m - mu ^ 2) := by
  ring

theorem avn_radial_damping (m eps : ℝ) (h : m + eps ≠ 0) :
    1 - m / (m + eps) = eps / (m + eps) := by
  field_simp
  ring

end AlgebraicTheory
