import AlgebraicAttention.Radical
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow

namespace AlgebraicAttention

theorem hasDerivAt_rho (x : ℝ) :
    HasDerivAt rho (rho x * (Real.sqrt (1 + x ^ 2))⁻¹) x := by
  have h := (Real.hasDerivAt_arsinh x).exp
  simpa only [← rho_eq_exp] using h

theorem hasDerivAt_kernel (n : ℕ) (x : ℝ) :
    HasDerivAt (fun t : ℝ => rho t ^ n)
      ((n : ℝ) * rho x ^ n * (Real.sqrt (1 + x ^ 2))⁻¹) x := by
  have h := (hasDerivAt_rho x).pow n
  convert! h using 1
  cases n with
  | zero => simp
  | succ k => simp only [Nat.succ_sub_one, pow_succ]; ring

/-- Diagonal of the normalized-weight Jacobian, stated for an algebraic log slope a. -/
theorem normalized_diagonal (f : ℝ → ℝ) (x a rest : ℝ)
    (hf : HasDerivAt f (a * f x) x) (hz : f x + rest ≠ 0) :
    HasDerivAt (fun t => f t / (f t + rest))
      (a * (f x / (f x + rest)) * (1 - f x / (f x + rest))) x := by
  have h := hf.div (hf.add_const rest) hz
  convert! h using 1
  field_simp

/-- Off-diagonal of the Jacobian when another key's weight is fixed. -/
theorem normalized_offdiagonal (f : ℝ → ℝ) (x a rest fixed : ℝ)
    (hf : HasDerivAt f (a * f x) x) (hz : f x + rest ≠ 0) :
    HasDerivAt (fun t => fixed / (f t + rest))
      (-a * (fixed / (f x + rest)) * (f x / (f x + rest))) x := by
  have h := (hasDerivAt_const x fixed).div (hf.add_const rest) hz
  convert! h using 1
  field_simp
  ring

/-- Weighted attention output's VJP; a zero-value sink is included in rest. -/
theorem normalized_output (f : ℝ → ℝ) (x a rest value otherNumerator : ℝ)
    (hf : HasDerivAt f (a * f x) x) (hz : f x + rest ≠ 0) :
    HasDerivAt (fun t => (f t * value + otherNumerator) / (f t + rest))
      (a * (f x / (f x + rest)) *
        (value - (f x * value + otherNumerator) / (f x + rest))) x := by
  have h := ((hf.mul_const value).add_const otherNumerator).div (hf.add_const rest) hz
  convert! h using 1
  field_simp

end AlgebraicAttention
