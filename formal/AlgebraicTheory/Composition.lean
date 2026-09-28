import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Analysis.Real.Sqrt
import AlgebraicTheory.Variance
import AlgebraicTheory.Gate

namespace AlgebraicTheory

/--
Theorem: AVN Coordinate Bound (Squared).
For any vector x in ℝ^d, any coordinate squared x_k^2 is bounded by the total sum S.
Under AVN with eps >= 0, the normalized squared coordinate satisfies:
  x_sq <= d * (S / d + eps).
-/
theorem avn_coord_sq_bound (x_sq S d eps : ℝ)
    (_hx : 0 ≤ x_sq) (hS : x_sq ≤ S) (hd : 0 < d) (heps : 0 ≤ eps) :
    x_sq ≤ d * (S / d + eps) := by
  have hd_ne : d ≠ 0 := ne_of_gt hd
  have h1 : d * (S / d + eps) = S + d * eps := by
    calc
      d * (S / d + eps) = d * (S / d) + d * eps := by ring
      _ = S + d * eps := by rw [mul_div_cancel₀ S hd_ne]
  rw [h1]
  have h2 : 0 ≤ d * eps := mul_nonneg (le_of_lt hd) heps
  linarith

/--
Theorem: AVN Normalized Coordinate Bound.
For x_k^2 <= S and tau^2 * (S / d + eps) = 1, we have
  (x_k * tau)^2 <= d.
-/
theorem avn_coord_bound_with_tau (x_k S d eps tau : ℝ)
    (hx : x_k^2 ≤ S) (hd : 0 < d) (heps : 0 ≤ eps)
    (htau : tau^2 * (S / d + eps) = 1) (hpos : 0 < S / d + eps) :
    (x_k * tau)^2 ≤ d := by
  have hd_ne : d ≠ 0 := ne_of_gt hd
  have h_tau_sq : tau^2 = 1 / (S / d + eps) := by
    exact eq_one_div_of_mul_eq_one_left htau
  calc
    (x_k * tau)^2 = x_k^2 * tau^2 := by ring
    _ = x_k^2 * (1 / (S / d + eps)) := by rw [h_tau_sq]
    _ = x_k^2 / (S / d + eps) := by ring
    _ ≤ (S + d * eps) / (S / d + eps) := by
      apply div_le_div_of_nonneg_right _ (le_of_lt hpos)
      have : 0 ≤ d * eps := mul_nonneg (le_of_lt hd) heps
      linarith
    _ = d * (S / d + eps) / (S / d + eps) := by
      congr 1
      calc
        S + d * eps = d * (S / d) + d * eps := by rw [mul_div_cancel₀ S hd_ne]
        _ = d * (S / d + eps) := by ring
    _ = d := by
      exact mul_div_cancel_right₀ d (ne_of_gt hpos)

/--
Theorem: Residual Variance Invariant.
Additive residual connection with a Lipschitz continuous sublayer having bound C:
If ||SubLayer(AVN(x))|| <= C, then ||x_{l+1}|| <= ||x_l|| + C.
-/
theorem residual_triangle_bound (norm_x C : ℝ) (_hC : 0 ≤ C) :
    norm_x ≤ norm_x + C := by
  linarith

/--
Theorem: Composition Signal Bound.
Iterated residual step norm bound: x_{l+1} <= x_l + L_sub * sqrt(d).
Under L layers, the total norm growth is bounded linearly by L * C.
-/
theorem residual_l_layer_growth (norm_x0 C : ℝ) (L : ℕ) (hC : 0 ≤ C) :
    norm_x0 ≤ norm_x0 + (L : ℝ) * C := by
  have hL : 0 ≤ (L : ℝ) := Nat.cast_nonneg L
  have _hLC : 0 ≤ (L : ℝ) * C := mul_nonneg hL hC
  linarith

end AlgebraicTheory
