import AlgebraicAttention.Accumulation
import AlgebraicAttention.Derivatives

namespace AlgebraicAttention

/-- RMS row normalization is linear in its learned gain. -/
theorem hasDerivAt_rms_gain (x tau g : ℝ) :
    HasDerivAt (fun t : ℝ => x * tau * t) (x * tau) g := by
  convert! (hasDerivAt_id g).const_mul (x * tau) using 1
  simp

/-- Local row partials sum to the original learned-gain VJP. -/
theorem partitioned_rms_gain_vjp {ι κ : Type*} [DecidableEq κ]
    (blocks : Finset ι) (rows : ι → Finset κ)
    (h : (↑blocks : Set ι).PairwiseDisjoint rows)
    (cot x tau : κ → ℝ) :
    (∑ j ∈ blocks.biUnion rows, cot j * (x j * tau j)) =
    ∑ b ∈ blocks, ∑ j ∈ rows b, cot j * (x j * tau j) := by
  exact Finset.sum_biUnion h

/-- Equivalent radial subtraction in real arithmetic; rounding is separate. -/
theorem rms_radial_reassociation (tau x cot radial : ℝ) :
    tau * (cot - (x * tau) * (tau * radial)) =
    tau * cot + 2 * x * (-(tau ^ 3) / 2 * radial) := by
  ring

end AlgebraicAttention
