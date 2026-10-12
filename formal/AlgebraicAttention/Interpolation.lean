import AlgebraicAttention.QuadraticPower

namespace AlgebraicAttention

noncomputable def quinticValue (a0 a1 a2 a3 a4 a5 t : ℝ) : ℝ :=
  ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t + a0

theorem quintic_horner_identity (a0 a1 a2 a3 a4 a5 t : ℝ) :
    quinticValue a0 a1 a2 a3 a4 a5 t =
    a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5 := by
  dsimp [quinticValue]
  ring

theorem hasDerivAt_quintic_value (a0 a1 a2 a3 a4 a5 x : ℝ) :
    HasDerivAt (quinticValue a0 a1 a2 a3 a4 a5)
      (a1 + 2*a2*x + 3*a3*x^2 + 4*a4*x^3 + 5*a5*x^4) x := by
  have h5 := ((hasDerivAt_id x).const_mul a5).add_const a4
  have h4 := (h5.mul (hasDerivAt_id x)).add_const a3
  have h3 := (h4.mul (hasDerivAt_id x)).add_const a2
  have h2 := (h3.mul (hasDerivAt_id x)).add_const a1
  have h1 := (h2.mul (hasDerivAt_id x)).add_const a0
  convert! h1 using 1
  dsimp [quinticValue]
  ring

noncomputable def hermiteQuintic (w0 w1 v0 v1 dd0 dd1 : ℝ) : ℝ → ℝ :=
  let c2 := dd0 / 2
  let f := w1 - w0 - v0 - c2
  let d := v1 - v0 - 2*c2
  let dd := dd1 - 2*c2
  quinticValue w0 v0 c2 (10*f - 4*d + dd/2) (-15*f + 7*d - dd) (6*f - 3*d + dd/2)

theorem hermite_quintic_left_value (w0 w1 v0 v1 dd0 dd1 : ℝ) :
    hermiteQuintic w0 w1 v0 v1 dd0 dd1 0 = w0 := by
  simp [hermiteQuintic, quinticValue]

theorem hermite_quintic_right_value (w0 w1 v0 v1 dd0 dd1 : ℝ) :
    hermiteQuintic w0 w1 v0 v1 dd0 dd1 1 = w1 := by
  dsimp [hermiteQuintic, quinticValue]
  ring

theorem hasDerivAt_hermite_quintic_left (w0 w1 v0 v1 dd0 dd1 : ℝ) :
    HasDerivAt (hermiteQuintic w0 w1 v0 v1 dd0 dd1) v0 0 := by
  dsimp [hermiteQuintic]
  convert! hasDerivAt_quintic_value w0 v0 (dd0/2)
    (10*(w1-w0-v0-dd0/2) - 4*(v1-v0-2*(dd0/2)) + (dd1-2*(dd0/2))/2)
    (-15*(w1-w0-v0-dd0/2) + 7*(v1-v0-2*(dd0/2)) - (dd1-2*(dd0/2)))
    (6*(w1-w0-v0-dd0/2) - 3*(v1-v0-2*(dd0/2)) + (dd1-2*(dd0/2))/2) 0 using 1
  simp

theorem hasDerivAt_hermite_quintic_right (w0 w1 v0 v1 dd0 dd1 : ℝ) :
    HasDerivAt (hermiteQuintic w0 w1 v0 v1 dd0 dd1) v1 1 := by
  dsimp [hermiteQuintic]
  convert! hasDerivAt_quintic_value w0 v0 (dd0/2)
    (10*(w1-w0-v0-dd0/2) - 4*(v1-v0-2*(dd0/2)) + (dd1-2*(dd0/2))/2)
    (-15*(w1-w0-v0-dd0/2) + 7*(v1-v0-2*(dd0/2)) - (dd1-2*(dd0/2)))
    (6*(w1-w0-v0-dd0/2) - 3*(v1-v0-2*(dd0/2)) + (dd1-2*(dd0/2))/2) 1 using 1
  norm_num
  ring

theorem hasDerivAt_finite_polynomial (s : Finset ℕ) (c : ℕ → ℝ) (x : ℝ) :
    HasDerivAt (fun t : ℝ => ∑ i ∈ s, c i * t ^ i)
      (∑ i ∈ s, c i * (i : ℝ) * x ^ (i - 1)) x := by
  apply HasDerivAt.fun_sum
  intro i hi
  convert! ((hasDerivAt_id x).pow i).const_mul (c i) using 1
  simp [mul_assoc]

/-- Relative inverse-square-root residual after a Newton refinement. -/
theorem inverse_root_newton_residual (a y : ℝ) :
    a * (y * ((3 - a*y^2) / 2)) ^ 2 - 1 =
    ((a*y^2 - 1)^3 - 3*(a*y^2 - 1)^2) / 4 := by
  ring

end AlgebraicAttention
