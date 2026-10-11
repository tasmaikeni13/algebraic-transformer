import AlgebraicAttention.Radical
import Mathlib.Algebra.Order.BigOperators.Ring.Finset

open scoped BigOperators

namespace AlgebraicAttention

/-- The real-arithmetic condition used by the runtime range guard. -/
theorem guarded_score_bound {ι : Type*} (s : Finset ι) (q k : ι → ℝ)
    (bound : ℝ) (hb : 0 ≤ bound)
    (h : (∑ i ∈ s, q i ^ 2) * (∑ i ∈ s, k i ^ 2) ≤ bound ^ 2) :
    |∑ i ∈ s, q i * k i| ≤ bound := by
  have hc := Finset.sum_mul_sq_le_sq_mul_sq s q k
  nlinarith [sq_abs (∑ i ∈ s, q i * k i), abs_nonneg (∑ i ∈ s, q i * k i)]

/-- Certified no-shift denominator bound for a finite collection of bounded scores. -/
theorem denominator_bound {ι : Type*} (s : Finset ι) (x : ι → ℝ)
    (X : ℝ) (n : ℕ) (_hX : 0 ≤ X) (hx : ∀ i ∈ s, |x i| ≤ X) :
    (∑ i ∈ s, rho (x i) ^ n) ≤ (s.card : ℝ) * (2 * X + 1) ^ n := by
  calc
    (∑ i ∈ s, rho (x i) ^ n) ≤ ∑ _i ∈ s, (2 * X + 1) ^ n := by
      apply Finset.sum_le_sum
      intro i hi
      have hr : rho (x i) ≤ 2 * X + 1 := by
        linarith [rho_le_two_abs_add_one (x i), hx i hi]
      exact pow_le_pow_left₀ (le_of_lt (rho_pos (x i))) hr n
    _ = (s.card : ℝ) * (2 * X + 1) ^ n := by simp

/-- Relative error on each weight also bounds the denominator, without an independence assumption. -/
theorem denominator_error {ι : Type*} (s : Finset ι) (w approx : ι → ℝ) (eps : ℝ)
    (h : ∀ i ∈ s, (1 - eps) * w i ≤ approx i ∧ approx i ≤ (1 + eps) * w i) :
    (1 - eps) * (∑ i ∈ s, w i) ≤ (∑ i ∈ s, approx i) ∧
    (∑ i ∈ s, approx i) ≤ (1 + eps) * (∑ i ∈ s, w i) := by
  constructor
  · rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun i hi => (h i hi).1)
  · rw [Finset.mul_sum]
    exact Finset.sum_le_sum (fun i hi => (h i hi).2)

end AlgebraicAttention
