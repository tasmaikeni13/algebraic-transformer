import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace AlgebraicTheory

theorem cayley_pythagorean_identity (w : ℝ) :
    (1 - w^2)^2 + (2 * w)^2 = (1 + w^2)^2 := by
  ring

theorem cayley_column_norm_one (w : ℝ) (hw : 1 + w^2 ≠ 0) :
    ((1 - w^2) / (1 + w^2))^2 + ((2 * w) / (1 + w^2))^2 = 1 := by
  field_simp [hw]
  ring

theorem cayley_col1_norm_sq (w : ℝ) (hw : 1 + w^2 ≠ 0) :
    ((1 - w^2) / (1 + w^2))^2 + ((2 * w) / (1 + w^2))^2 = 1 :=
  cayley_column_norm_one w hw

theorem cayley_col2_norm_sq (w : ℝ) (hw : 1 + w^2 ≠ 0) :
    ((-2 * w) / (1 + w^2))^2 + ((1 - w^2) / (1 + w^2))^2 = 1 := by
  field_simp [hw]
  ring

theorem cayley_columns_orthogonal (w : ℝ) :
    (1 - w^2) * (-2 * w) + (2 * w) * (1 - w^2) = 0 := by
  ring

theorem cayley_dot_product_zero (w : ℝ) (hw : 1 + w^2 ≠ 0) :
    ((1 - w^2) / (1 + w^2)) * ((-2 * w) / (1 + w^2)) +
    ((2 * w) / (1 + w^2)) * ((1 - w^2) / (1 + w^2)) = 0 := by
  field_simp [hw]
  ring

theorem cayley_determinant_one (w : ℝ) (hw : 1 + w^2 ≠ 0) :
    ((1 - w^2) * (1 - w^2) - (-2 * w) * (2 * w)) / (1 + w^2)^2 = 1 := by
  field_simp [hw]
  ring

theorem cayley_det_one (w : ℝ) (hw : 1 + w^2 ≠ 0) :
    ((1 - w^2) * (1 - w^2) - (-2 * w) * (2 * w)) / (1 + w^2)^2 = 1 :=
  cayley_determinant_one w hw

theorem cayley_norm_preserving (c s v1 v2 : ℝ) (h : c^2 + s^2 = 1) :
    (c * v1 - s * v2)^2 + (s * v1 + c * v2)^2 = v1^2 + v2^2 := by
  calc
    (c * v1 - s * v2)^2 + (s * v1 + c * v2)^2 = (c^2 + s^2) * (v1^2 + v2^2) := by ring
    _ = 1 * (v1^2 + v2^2) := by rw [h]
    _ = v1^2 + v2^2 := by ring

theorem cayley_rational_norm_preserving (w v1 v2 : ℝ) (hw : 1 + w^2 ≠ 0) :
    (((1 - w^2) / (1 + w^2)) * v1 - ((2 * w) / (1 + w^2)) * v2)^2 +
    (((2 * w) / (1 + w^2)) * v1 + ((1 - w^2) / (1 + w^2)) * v2)^2 = v1^2 + v2^2 := by
  field_simp [hw]
  ring

theorem cayley_composition_norm (c1 s1 c2 s2 : ℝ)
    (h1 : c1^2 + s1^2 = 1) (h2 : c2^2 + s2^2 = 1) :
    (c1 * c2 - s1 * s2)^2 + (c1 * s2 + s1 * c2)^2 = 1 := by
  calc
    (c1 * c2 - s1 * s2)^2 + (c1 * s2 + s1 * c2)^2 =
      (c1^2 + s1^2) * (c2^2 + s2^2) := by ring
    _ = 1 * 1 := by rw [h1, h2]
    _ = 1 := by ring

theorem cayley_shift_equivariance (c1 s1 c2 s2 q1 q2 k1 k2 : ℝ) :
    (c1 * q1 - s1 * q2) * (c2 * k1 - s2 * k2) + (s1 * q1 + c1 * q2) * (s2 * k1 + c2 * k2) =
    q1 * ((c1 * c2 + s1 * s2) * k1 - (c1 * s2 - s1 * c2) * k2) +
    q2 * ((c1 * s2 - s1 * c2) * k1 + (c1 * c2 + s1 * s2) * k2) := by
  ring

end AlgebraicTheory

