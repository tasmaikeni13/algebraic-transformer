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

end AlgebraicAttention
