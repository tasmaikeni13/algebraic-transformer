import AlgebraicAttention.AffinePower

namespace AlgebraicAttention

noncomputable def quadraticRho (x : ℝ) : ℝ := 1 + x + x ^ 2 / 2

theorem quadratic_rho_lower (x : ℝ) : 1 / 2 ≤ quadraticRho x := by
  dsimp [quadraticRho]
  nlinarith [sq_nonneg (x + 1)]

theorem quadratic_rho_pos (x : ℝ) : 0 < quadraticRho x := by
  linarith [quadratic_rho_lower x]

theorem hasDerivAt_quadratic_rho (x : ℝ) :
    HasDerivAt quadraticRho (1 + x) x := by
  have h := ((hasDerivAt_const x (1 : ℝ)).add (hasDerivAt_id x)).add
    (((hasDerivAt_id x).pow 2).div_const 2)
  convert! h using 1
  simp

theorem hasDerivAt_quadratic_power (n : ℕ) (x : ℝ) :
    HasDerivAt (fun t : ℝ => quadraticRho t ^ n)
      ((n : ℝ) * quadraticRho x ^ (n - 1) * (1 + x)) x := by
  exact (hasDerivAt_quadratic_rho x).pow n

theorem quadratic_rho_monotone (x y : ℝ) (hx : -1 ≤ x) (hxy : x ≤ y) :
    quadraticRho x ≤ quadraticRho y := by
  have hp : 0 ≤ (y - x) * (y + x + 2) := mul_nonneg (by linarith) (by linarith)
  dsimp [quadraticRho]
  nlinarith

theorem quadratic_rho_bounds (x : ℝ) (hx : -1 ≤ x) (hu : x ≤ 8) :
    1 / 2 ≤ quadraticRho x ∧ quadraticRho x ≤ 41 := by
  constructor
  · exact quadratic_rho_lower x
  · have h := quadratic_rho_monotone x 8 hx hu
    norm_num [quadraticRho] at h ⊢
    exact h

/-- The n/c-scaled logarithmic slope never exceeds its value at zero. -/
theorem quadratic_log_slope_bound (x : ℝ) (hx : -1 ≤ x) :
    0 ≤ (1 + x) / quadraticRho x ∧ (1 + x) / quadraticRho x ≤ 1 := by
  have hp := quadratic_rho_pos x
  constructor
  · exact div_nonneg (by linarith) (le_of_lt hp)
  · apply (div_le_iff₀ hp).2
    dsimp [quadraticRho]
    nlinarith [sq_nonneg x]

/-- The quadratic base differs from the exact radical only at fourth order. -/
theorem quadratic_surplus_identity (x : ℝ) :
    quadraticRho x - rho x =
      x ^ 4 / (4 * (1 + x ^ 2 / 2 + Real.sqrt (1 + x ^ 2))) := by
  have hd : 4 * (1 + x ^ 2 / 2 + Real.sqrt (1 + x ^ 2)) ≠ 0 := by positivity
  apply (eq_div_iff hd).2
  dsimp [quadraticRho, rho]
  nlinarith [radical_sq x]

theorem quadratic_surplus_nonneg (x : ℝ) : 0 ≤ quadraticRho x - rho x := by
  rw [quadratic_surplus_identity]
  positivity

theorem shared_power_chain_16 (x : ℝ) :
    let x2 := x * x
    let x3 := x2 * x
    let x6 := x3 * x3
    let x12 := x6 * x6
    let x15 := x12 * x3
    x15 = x ^ 15 ∧ x15 * x = x ^ 16 := by
  dsimp
  constructor <;> ring

/-- One Newton reciprocal refinement squares the relative residual in reals. -/
theorem reciprocal_newton_residual (a y : ℝ) :
    a * (y * (2 - a * y)) - 1 = -(a * y - 1) ^ 2 := by
  ring

theorem reciprocal_newton_error (a y eps : ℝ) (h : |a * y - 1| ≤ eps) :
    |a * (y * (2 - a * y)) - 1| ≤ eps ^ 2 := by
  rw [reciprocal_newton_residual, abs_neg, abs_pow]
  exact pow_le_pow_left₀ (abs_nonneg _) h 2

end AlgebraicAttention
