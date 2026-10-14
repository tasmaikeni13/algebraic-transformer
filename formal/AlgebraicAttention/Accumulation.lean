import AlgebraicAttention.Normalization

namespace AlgebraicAttention

/-- Moving a common score slope outside key-block reduction preserves dQ. -/
theorem slope_after_partial_reduction {ι : Type*} (blocks : Finset ι)
    (partialDQ : ι → ℝ) (slope : ℝ) :
    (∑ b ∈ blocks, partialDQ b * slope) = (∑ b ∈ blocks, partialDQ b) * slope := by
  exact (Finset.sum_mul blocks partialDQ slope).symm

/-- A disjoint partition of the keys preserves each component of the VJP. -/
theorem partitioned_key_vjp {ι κ : Type*} [DecidableEq κ]
    (blocks : Finset ι) (keys : ι → Finset κ)
    (h : (↑blocks : Set ι).PairwiseDisjoint keys) (ds key : κ → ℝ) :
    (∑ j ∈ blocks.biUnion keys, ds j * key j) =
    ∑ b ∈ blocks, ∑ j ∈ keys b, ds j * key j := by
  exact Finset.sum_biUnion h

/-- Omitting keys excluded by the causal mask preserves the VJP contraction. -/
theorem omit_masked_key_vjp {ι : Type*} (keys : Finset ι)
    (visible : ι → Prop) [DecidablePred visible] (ds key : ι → ℝ) :
    (∑ j ∈ keys, if visible j then ds j * key j else 0) =
      ∑ j ∈ keys.filter visible, ds j * key j := by
  rw [Finset.sum_filter]

/-- Row denominators can be moved from probabilities to value cotangents.
This identity precedes the BF16 operand rounding in the projective probe. -/
theorem projective_value_vjp {ι : Type*} (queries : Finset ι)
    (weight cotangent denominator : ι → ℝ) :
    (∑ i ∈ queries, (weight i / denominator i) * cotangent i) =
      ∑ i ∈ queries, weight i * (cotangent i / denominator i) := by
  apply Finset.sum_congr rfl
  intro i hi
  simp only [div_eq_mul_inv]
  ring

/-- The centered score VJP admits the same row-factor extraction. -/
theorem projective_score_vjp (weight slope cotangent centered denominator : ℝ) :
    (weight / denominator) * slope * (cotangent - centered) =
      weight * slope * (cotangent / denominator - centered / denominator) := by
  simp only [div_eq_mul_inv]
  ring

/-- Appending a value coordinate of one reduces the denominator in the same
matrix contraction as the numerator. The zero-value sink contributes its
weight only to this coordinate. -/
theorem augmented_value_denominator {ι : Type*} (keys : Finset ι)
    (weight : ι → ℝ) (sinkWeight : ℝ) :
    sinkWeight + (∑ j ∈ keys, weight j * 1) =
      sinkWeight + ∑ j ∈ keys, weight j := by
  simp

/-- Keys at or after the end of a query tile cannot be causally visible. -/
theorem causal_suffix_excluded (row tileEnd key : ℕ)
    (hrow : row < tileEnd) (hkey : tileEnd ≤ key) : ¬ key ≤ row := by
  omega

/-- Trimming a key tile at the end of the current query tile preserves a
causal weight or VJP sum. Floating-point reassociation is not asserted. -/
theorem trim_causal_sum (length tileEnd row : ℕ) (value : ℕ → ℝ)
    (hend : tileEnd ≤ length) (hrow : row < tileEnd) :
    (∑ j ∈ Finset.range length, if j ≤ row then value j else 0) =
      ∑ j ∈ Finset.range tileEnd, if j ≤ row then value j else 0 := by
  symm
  apply Finset.sum_subset (Finset.range_mono hend)
  intro j hj hnot
  have hkey : tileEnd ≤ j := by simpa only [Finset.mem_range, not_lt] using hnot
  simp [causal_suffix_excluded row tileEnd j hrow hkey]

/-- Rounding the saved output changes the centered score VJP by this bias. -/
theorem output_center_rounding_bias (weight slope dotCot exactCenter roundedCenter : ℝ) :
    weight * slope * (dotCot - roundedCenter) -
      weight * slope * (dotCot - exactCenter) =
        weight * slope * (exactCenter - roundedCenter) := by
  ring

/-- For equal keys, centering leaves only the zero-value sink contribution. -/
theorem sink_centered_key_sum {ι : Type*} (keys : Finset ι) (p dotCot : ι → ℝ)
    (center sinkMass : ℝ) (hmass : ∑ j ∈ keys, p j = 1 - sinkMass)
    (hcenter : ∑ j ∈ keys, p j * dotCot j = center) :
    (∑ j ∈ keys, p j * (dotCot j - center)) = center * sinkMass := by
  simp_rw [mul_sub]
  rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hcenter, hmass]
  ring

/-- A relative bound on rounding score cotangents controls the dQ contraction
when the key coordinate is constant. Machine rounding is an explicit premise. -/
theorem constant_key_rounding_bound {ι : Type*} (keys : Finset ι)
    (rounded exact : ι → ℝ) (key epsilon : ℝ)
    (hround : ∀ j ∈ keys, |rounded j - exact j| ≤ epsilon * |exact j|) :
    |(∑ j ∈ keys, rounded j * key) - (∑ j ∈ keys, exact j * key)| ≤
      epsilon * |key| * (∑ j ∈ keys, |exact j|) := by
  have hid : (∑ j ∈ keys, rounded j * key) - (∑ j ∈ keys, exact j * key) =
      ∑ j ∈ keys, (rounded j - exact j) * key := by
    simp_rw [sub_mul, Finset.sum_sub_distrib]
  rw [hid]
  calc
    |∑ j ∈ keys, (rounded j - exact j) * key| ≤
        ∑ j ∈ keys, |(rounded j - exact j) * key| := Finset.abs_sum_le_sum_abs _ _
    _ = (∑ j ∈ keys, |rounded j - exact j|) * |key| := by
      simp_rw [abs_mul]
      rw [Finset.sum_mul]
    _ ≤ (∑ j ∈ keys, epsilon * |exact j|) * |key| := by
      exact mul_le_mul_of_nonneg_right (Finset.sum_le_sum hround) (abs_nonneg key)
    _ = epsilon * |key| * (∑ j ∈ keys, |exact j|) := by
      rw [← Finset.mul_sum]
      ring

end AlgebraicAttention
