import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace AlgebraicTheory

theorem adamw_factorized_curvature_recovery (a_i b_j a_bar b_bar : ℝ)
    (ha : a_bar ≠ 0) (hb : b_bar ≠ 0) :
    ((a_i * b_bar) * (b_j * a_bar)) / (a_bar * b_bar) = a_i * b_j := by
  field_simp [ha, hb]

theorem adamw_debiasing_identity (m beta_t : ℝ) (h : 1 - beta_t ≠ 0) :
    (m / (1 - beta_t)) * (1 - beta_t) = m := by
  field_simp [h]

theorem factorized_rank1_recovery (a_i b_j a_bar b_bar : ℝ)
    (ha : a_bar ≠ 0) (hb : b_bar ≠ 0) :
    ((a_i * b_bar) * (b_j * a_bar)) / (a_bar * b_bar) = a_i * b_j :=
  adamw_factorized_curvature_recovery a_i b_j a_bar b_bar ha hb

theorem debiasing_identity (m beta_t : ℝ) (h : 1 - beta_t ≠ 0) :
    (m / (1 - beta_t)) * (1 - beta_t) = m :=
  adamw_debiasing_identity m beta_t h

theorem adamw_decoupled_weight_decay (w u lr wd : ℝ) :
    w - lr * u - lr * wd * w = (1 - lr * wd) * w - lr * u := by
  ring

theorem decoupled_weight_decay_step (w u lr wd : ℝ) :
    w - lr * u - lr * wd * w = (1 - lr * wd) * w - lr * u :=
  adamw_decoupled_weight_decay w u lr wd

end AlgebraicTheory
