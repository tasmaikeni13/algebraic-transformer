import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Arsinh
import Mathlib.Tactic

namespace AlgebraicAttention

noncomputable def rho (x : ℝ) : ℝ := x + Real.sqrt (1 + x ^ 2)

theorem radical_sq (x : ℝ) : (Real.sqrt (1 + x ^ 2)) ^ 2 = 1 + x ^ 2 := by
  exact Real.sq_sqrt (by positivity)

theorem abs_lt_radical (x : ℝ) : |x| < Real.sqrt (1 + x ^ 2) := by
  have hs := radical_sq x
  have hn := Real.sqrt_nonneg (1 + x ^ 2)
  have ha := sq_abs x
  nlinarith [abs_nonneg x]

theorem rho_pos (x : ℝ) : 0 < rho x := by
  have h := abs_lt_radical x
  have hx := neg_abs_le x
  unfold rho
  linarith

theorem rho_eq_exp (x : ℝ) : rho x = Real.exp (Real.arsinh x) := by
  exact (Real.exp_arsinh x).symm

theorem rho_strictMono : StrictMono rho := by
  intro x y h
  rw [rho_eq_exp, rho_eq_exp]
  exact Real.exp_strictMono (Real.arsinh_strictMono h)

noncomputable def oplus (x y : ℝ) : ℝ :=
  x * Real.sqrt (1 + y ^ 2) + y * Real.sqrt (1 + x ^ 2)

theorem arsinh_oplus (x y : ℝ) :
    Real.arsinh (oplus x y) = Real.arsinh x + Real.arsinh y := by
  apply Real.sinh_injective
  simp [oplus, Real.sinh_add, Real.cosh_arsinh, Real.sinh_arsinh, mul_comm]

theorem rho_group_law (x y : ℝ) : rho (oplus x y) = rho x * rho y := by
  simp only [rho_eq_exp, arsinh_oplus, Real.exp_add]

theorem rho_reciprocal (x : ℝ) : rho x * rho (-x) = 1 := by
  have hs := radical_sq x
  unfold rho
  have hn : 1 + (-x) ^ 2 = 1 + x ^ 2 := by ring
  rw [hn]
  nlinarith

theorem rho_neg (x : ℝ) : rho (-x) = 1 / rho x := by
  apply (eq_div_iff (ne_of_gt (rho_pos x))).2
  nlinarith [rho_reciprocal x]

/-- Exact inverse of rho, used to construct the overflow cap for any power. -/
theorem rho_inverse (t : ℝ) (ht : 0 < t) : rho ((t - 1 / t) / 2) = t := by
  have ht0 : t ≠ 0 := ne_of_gt ht
  have hs : 0 ≤ (t + 1 / t) / 2 := by positivity
  have hr : 1 + ((t - 1 / t) / 2) ^ 2 = ((t + 1 / t) / 2) ^ 2 := by
    field_simp
    ring
  unfold rho
  rw [hr, Real.sqrt_sq hs]
  ring

theorem shifted_weight_le_one (x m : ℝ) (n : ℕ) (hxm : x ≤ m) :
    0 < (rho x * rho (-m)) ^ n ∧ (rho x * rho (-m)) ^ n ≤ 1 := by
  have hr : rho x ≤ rho m := rho_strictMono.monotone hxm
  have hprod : rho x * rho (-m) ≤ 1 := by
    rw [rho_neg, one_div, ← div_eq_mul_inv]
    exact (div_le_one (rho_pos m)).2 hr
  constructor
  · exact pow_pos (mul_pos (rho_pos x) (rho_pos (-m))) n
  · simpa using pow_le_pow_left₀ (le_of_lt (mul_pos (rho_pos x) (rho_pos (-m)))) hprod n

/-- Stable negative branch, with no subtraction of nearby positive numbers. -/
theorem rho_stable_negative (x : ℝ) (hx : x ≤ 0) :
    rho x = 1 / (|x| + Real.sqrt (1 + x ^ 2)) := by
  have h := rho_neg (-x)
  simp only [neg_neg] at h
  rw [abs_of_nonpos hx]
  simpa [rho] using h

theorem radical_le_abs_add_one (x : ℝ) :
    Real.sqrt (1 + x ^ 2) ≤ |x| + 1 := by
  have hs := radical_sq x
  have hn := Real.sqrt_nonneg (1 + x ^ 2)
  nlinarith [abs_nonneg x, sq_abs x]

/-- This is the corrected global bound; 2X alone is not a bound at X=0. -/
theorem rho_le_two_abs_add_one (x : ℝ) : rho x ≤ 2 * |x| + 1 := by
  unfold rho
  linarith [radical_le_abs_add_one x, le_abs_self x]

theorem squared_log_slope_bound (x : ℝ) :
    0 < 1 / (1 + x ^ 2) ∧ 1 / (1 + x ^ 2) ≤ 1 := by
  have hd : 0 < 1 + x ^ 2 := by positivity
  constructor
  · positivity
  · apply (div_le_one hd).2
    nlinarith [sq_nonneg x]

end AlgebraicAttention
