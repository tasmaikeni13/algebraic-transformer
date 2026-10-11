import Mathlib.Algebra.BigOperators.Field
import Mathlib.Tactic

namespace AlgebraicAttention

/-- Online rescaling preserves the normalized output, including a zero-value sink. -/
theorem rescaling_invariant (a numerator denominator : ℝ) (ha : a ≠ 0) :
    (a * numerator) / (a * denominator) = numerator / denominator := by
  exact mul_div_mul_left numerator denominator ha

/-- The inverse-denominator statistic reconstructs the same probabilities. -/
theorem inverse_denominator_statistic (w z : ℝ) : w * z⁻¹ = w / z := by
  exact (div_eq_mul_inv w z).symm

/-- Root normalization and direct normalization agree when the stored root is exact. -/
theorem root_statistic (w sigma z : ℝ) (n : ℕ) (h : sigma ^ n = z⁻¹) :
    (w * sigma) ^ n = w ^ n / z := by
  rw [mul_pow, h, div_eq_mul_inv]

/-- The recurrence is a genuine tile decomposition, with a sink in the initial denominator. -/
theorem tile_decomposition (a b sink u v : ℝ) :
    (a * u + b * v) / (sink + a + b) =
    a / (sink + a + b) * u + b / (sink + a + b) * v := by
  ring

/-- Entrywise weight error cannot grow without bound under normalization. -/
theorem normalized_weight_bounds (w approx z approxZ eps : ℝ)
    (hw : 0 ≤ w) (hz : 0 < z) (he0 : 0 ≤ eps) (he1 : eps < 1)
    (hwa : (1 - eps) * w ≤ approx) (hwb : approx ≤ (1 + eps) * w)
    (hza : (1 - eps) * z ≤ approxZ) (hzb : approxZ ≤ (1 + eps) * z) :
    ((1 - eps) / (1 + eps)) * (w / z) ≤ approx / approxZ ∧
    approx / approxZ ≤ ((1 + eps) / (1 - eps)) * (w / z) := by
  have hep : 0 < 1 + eps := by linarith
  have hem : 0 < 1 - eps := by linarith
  have haz : 0 < approxZ := lt_of_lt_of_le (mul_pos hem hz) hza
  have haw : 0 ≤ approx := le_trans (mul_nonneg (le_of_lt hem) hw) hwa
  constructor
  · have h : (1 - eps) * w * (z * (1 + eps)) ≤ approx * (z * (1 + eps)) := by
      exact mul_le_mul_of_nonneg_right hwa (le_of_lt (mul_pos hz hep))
    have h2 : approx * approxZ ≤ approx * ((1 + eps) * z) :=
      mul_le_mul_of_nonneg_left hzb haw
    apply (le_div_iff₀ haz).2
    field_simp
    nlinarith [mul_nonneg haw (sub_nonneg.mpr hzb)]
  · apply (div_le_iff₀ haz).2
    field_simp
    nlinarith [mul_nonneg hw (sub_nonneg.mpr hza)]

end AlgebraicAttention
