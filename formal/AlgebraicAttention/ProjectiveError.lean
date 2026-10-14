import AlgebraicAttention.Bounds

open scoped BigOperators

namespace AlgebraicAttention

/-- A secant bound for absolute deviation on a finite likelihood-ratio interval. -/
theorem ratio_deviation_secant (l u r t : ℝ) (hlr : l ≤ r) (hru : r ≤ u)
    (hlt : l ≤ t) (htu : t ≤ u) :
    (u - l) * |r - t| ≤ (u - r) * (t - l) + (r - l) * (u - t) := by
  by_cases hrt : r ≤ t
  · rw [abs_of_nonpos (sub_nonpos.mpr hrt)]
    nlinarith [mul_nonneg (sub_nonneg.mpr hlr) (sub_nonneg.mpr htu)]
  · rw [abs_of_nonneg (sub_nonneg.mpr (le_of_not_ge hrt))]
    nlinarith [mul_nonneg (sub_nonneg.mpr hru) (sub_nonneg.mpr hlt)]

/-- Normalized probabilities turn the secant inequality into a mean-deviation bound. -/
theorem weighted_ratio_deviation {ι : Type*} (s : Finset ι) (p r : ι → ℝ)
    (l u t : ℝ) (hp : ∀ i ∈ s, 0 ≤ p i) (hpsum : ∑ i ∈ s, p i = 1)
    (hmean : ∑ i ∈ s, p i * r i = t)
    (hr : ∀ i ∈ s, l ≤ r i ∧ r i ≤ u) (hlt : l ≤ t) (htu : t ≤ u) :
    (u - l) * (∑ i ∈ s, p i * |r i - t|) ≤ 2 * (u - t) * (t - l) := by
  calc
    (u - l) * (∑ i ∈ s, p i * |r i - t|) =
        ∑ i ∈ s, p i * ((u - l) * |r i - t|) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i hi
      ring
    _ ≤ ∑ i ∈ s, p i * ((u - r i) * (t - l) + (r i - l) * (u - t)) := by
      apply Finset.sum_le_sum
      intro i hi
      exact mul_le_mul_of_nonneg_left
        (ratio_deviation_secant l u (r i) t (hr i hi).1 (hr i hi).2 hlt htu) (hp i hi)
    _ = (∑ i ∈ s, p i) * (u * (t - l) - l * (u - t)) +
        (∑ i ∈ s, p i * r i) * ((u - t) - (t - l)) := by
      simp_rw [Finset.sum_mul]
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro i hi
      ring
    _ = 2 * (u - t) * (t - l) := by rw [hpsum, hmean]; ring

/-- The best scalar bound is attained at t=a*b, requiring only arithmetic. -/
theorem ratio_deviation_optimum (a b t : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    (b ^ 2 - t) * (t - a ^ 2) * (a + b) ≤
      t * (b ^ 2 - a ^ 2) * (b - a) := by
  calc
    (b ^ 2 - t) * (t - a ^ 2) * (a + b) =
        t * (b ^ 2 - a ^ 2) * (b - a) - (a + b) * (t - a * b) ^ 2 := by ring
    _ ≤ t * (b ^ 2 - a ^ 2) * (b - a) :=
      sub_le_self _ (mul_nonneg (add_nonneg ha hb) (sq_nonneg _))

/-- Sharp TV bound when weight ratios lie between a² and b².
The mean ratio t is the denominator ratio. No independence assumption is used. -/
theorem sharp_normalized_ratio_tv {ι : Type*} (s : Finset ι) (p r : ι → ℝ)
    (a b t : ℝ) (ha : 0 < a) (hab : a < b) (ht : 0 < t)
    (hp : ∀ i ∈ s, 0 ≤ p i) (hpsum : ∑ i ∈ s, p i = 1)
    (hmean : ∑ i ∈ s, p i * r i = t)
    (hr : ∀ i ∈ s, a ^ 2 ≤ r i ∧ r i ≤ b ^ 2)
    (hlt : a ^ 2 ≤ t) (htu : t ≤ b ^ 2) :
    (∑ i ∈ s, p i * |r i / t - 1|) / 2 ≤ (b - a) / (b + a) := by
  have hb : 0 < b := lt_trans ha hab
  have hdiff : 0 < b ^ 2 - a ^ 2 := by nlinarith
  have hsum := weighted_ratio_deviation s p r (a ^ 2) (b ^ 2) t
    hp hpsum hmean hr hlt htu
  have hscaled := mul_le_mul_of_nonneg_right hsum (le_of_lt (add_pos ha hb))
  have hopt := ratio_deviation_optimum a b t (le_of_lt ha) (le_of_lt hb)
  have hcancel : (b ^ 2 - a ^ 2) * ((∑ i ∈ s, p i * |r i - t|) * (b + a)) ≤
      (b ^ 2 - a ^ 2) * (2 * t * (b - a)) := by nlinarith [hscaled, hopt]
  have hfinal := le_of_mul_le_mul_left hcancel hdiff
  have hnormalize : (∑ i ∈ s, p i * |r i / t - 1|) =
      (∑ i ∈ s, p i * |r i - t|) / t := by
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i hi
    have hid : r i / t - 1 = (r i - t) / t := by field_simp
    rw [hid, abs_div, abs_of_pos ht]
    ring
  rw [hnormalize, div_div]
  apply (div_le_div_iff₀ (mul_pos ht (by norm_num : (0 : ℝ) < 2)) (add_pos hb ha)).mpr
  nlinarith [hfinal]

end AlgebraicAttention
