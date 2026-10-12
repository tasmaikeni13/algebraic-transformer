import AlgebraicAttention.Rational

namespace AlgebraicAttention

theorem hasDerivAt_affine_power (n : ℕ) (x : ℝ) :
    HasDerivAt (fun t : ℝ => (1 + t) ^ n) ((n : ℝ) * (1 + x) ^ (n - 1)) x := by
  convert! ((hasDerivAt_id x).const_add 1).pow n using 1
  simp

/-- Weight bounds for the n=64 candidate's asymmetric cap. -/
theorem affine_base_bounds (x : ℝ) (hl : -(1 / 2 : ℝ) ≤ x) (hu : x ≤ 3 / 4) :
    1 / 2 ≤ 1 + x ∧ 1 + x ≤ 7 / 4 := by
  constructor <;> linarith

theorem affine_log_slope_bound (x : ℝ) (hx : -(1 / 2 : ℝ) ≤ x) :
    0 < 1 / (1 + x) ∧ 1 / (1 + x) ≤ 2 := by
  have hd : 0 < 1 + x := by linarith
  constructor
  · positivity
  · apply (div_le_iff₀ hd).2
    linarith

/-- Nine multiplications jointly reconstruct x^63 and x^64. -/
theorem shared_power_chain_64 (x : ℝ) :
    let x2 := x * x
    let x3 := x2 * x
    let x6 := x3 * x3
    let x7 := x6 * x
    let x14 := x7 * x7
    let x21 := x14 * x7
    let x42 := x21 * x21
    let x63 := x42 * x21
    x63 = x ^ 63 ∧ x63 * x = x ^ 64 := by
  dsimp
  constructor <;> ring

/-- Eight multiplications jointly reconstruct x^31 and x^32. -/
theorem shared_power_chain_32 (x : ℝ) :
    let x2 := x * x
    let x3 := x2 * x
    let x5 := x3 * x2
    let x10 := x5 * x5
    let x15 := x10 * x5
    let x30 := x15 * x15
    let x31 := x30 * x
    x31 = x ^ 31 ∧ x31 * x = x ^ 32 := by
  dsimp
  constructor <;> ring

end AlgebraicAttention
