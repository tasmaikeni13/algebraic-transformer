import AlgebraicAttention.Derivatives

namespace AlgebraicAttention

noncomputable def rationalRho (x : ℝ) : ℝ := (1 + x) / (1 - x)

theorem rational_rho_bounds (x : ℝ) (hlo : -(3 / 4 : ℝ) ≤ x) (hhi : x ≤ 3 / 4) :
    1 / 7 ≤ rationalRho x ∧ rationalRho x ≤ 7 := by
  have hd : 0 < 1 - x := by linarith
  unfold rationalRho
  constructor
  · apply (le_div_iff₀ hd).2
    linarith
  · apply (div_le_iff₀ hd).2
    linarith

theorem hasDerivAt_rational_rho (x : ℝ) (hx : 1 - x ≠ 0) :
    HasDerivAt rationalRho (2 / (1 - x) ^ 2) x := by
  have h := ((hasDerivAt_const x (1 : ℝ)).add (hasDerivAt_id x)).div
    ((hasDerivAt_const x (1 : ℝ)).sub (hasDerivAt_id x)) hx
  simp only [Pi.add_apply, Pi.sub_apply, id_eq, zero_add, zero_sub] at h
  convert! h using 1
  field_simp
  ring

theorem hasDerivAt_rational_power (n : ℕ) (x : ℝ) (hx : 1 - x ≠ 0) :
    HasDerivAt (fun t : ℝ => rationalRho t ^ n)
      ((n : ℝ) * rationalRho x ^ (n - 1) * (2 / (1 - x) ^ 2)) x := by
  exact (hasDerivAt_rational_rho x hx).pow n

/-- One reciprocal reused in the derivative instead of a second division. -/
theorem rational_derivative_product (n : ℕ) (x : ℝ) (hx : 1 - x ≠ 0) :
    (n : ℝ) * rationalRho x ^ (n - 1) * (2 / (1 - x) ^ 2) =
    2 * (n : ℝ) * ((1 - x)⁻¹) ^ 2 * rationalRho x ^ (n - 1) := by
  field_simp

theorem rational_slope_bound (x beta : ℝ) (hb : 0 < beta)
    (hx : |x| ≤ 3 / 4) :
    0 < beta / (1 - x ^ 2) ∧ beta / (1 - x ^ 2) ≤ (16 / 7) * beta := by
  have hsq : x ^ 2 ≤ 9 / 16 := by nlinarith [sq_abs x, abs_nonneg x]
  have hd : 0 < 1 - x ^ 2 := by linarith
  constructor
  · exact div_pos hb hd
  · apply (div_le_iff₀ hd).2
    nlinarith [mul_nonneg (le_of_lt hb) (show 0 ≤ 9 / 16 - x ^ 2 by linarith)]

end AlgebraicAttention
