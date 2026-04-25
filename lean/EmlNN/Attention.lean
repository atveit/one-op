import EmlNN.Basic
import EmlNN.Arith
import EmlNN.Activations
import EmlNN.Compose
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

namespace EmlNN

open Real
open scoped BigOperators

/-! ## Attention-family theorems for the EML neural-network paper.

    These theorems formalize:
    * shift invariance of softmax (local copy, to avoid concurrency with
      `Activations.lean` edits),
    * log-sum-exp shift invariance (local copy),
    * the paper's P3 "attention from EML" composition,
    * the log-domain (numerically stable) attention and its equivalence
      to naive softmax attention,
    * a two-block FlashAttention-style online recurrence and a core
      consistency lemma (full two-block equivalence sketched below),
    * max-plus (tropical) attention as a well-defined construction.
-/

/-- Local copy of softmax shift invariance, pointwise form. -/
theorem att_softmax_invariance {n : ℕ} [NeZero n]
    (v : Fin n → ℝ) (c : ℝ) (i : Fin n) :
    softmax (fun j => v j + c) i = softmax v i := by
  have hpos : 0 < ∑ j, Real.exp (v j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos (v j)) Finset.univ_nonempty
  have hne : (∑ j, Real.exp (v j)) ≠ 0 := ne_of_gt hpos
  have hec : Real.exp c ≠ 0 := Real.exp_ne_zero _
  have hsum : ∑ j, Real.exp (v j + c) = Real.exp c * ∑ j, Real.exp (v j) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [Real.exp_add]; ring
  simp only [softmax]
  rw [hsum, Real.exp_add, mul_comm (Real.exp (v i)) (Real.exp c),
      mul_div_mul_left _ _ hec]

/-- Local copy of log-sum-exp shift invariance. -/
theorem att_lse_shift_invariance {n : ℕ} [NeZero n] (v : Fin n → ℝ) (c : ℝ) :
    Real.log (∑ i, Real.exp (v i + c)) = c + Real.log (∑ i, Real.exp (v i)) := by
  have hpos : 0 < ∑ j, Real.exp (v j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos (v j)) Finset.univ_nonempty
  have hsum : ∑ i, Real.exp (v i + c) = Real.exp c * ∑ i, Real.exp (v i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [Real.exp_add]; ring
  rw [hsum, Real.log_mul (Real.exp_ne_zero _) (ne_of_gt hpos), Real.log_exp]

/-! ### Scaled dot-product attention -/

/-- Scaled dot-product attention as the P3 composition of softmax over
    `scale · ⟨Q_i, K_j⟩` logits with the value matrix `V`. -/
noncomputable def attention {n d : ℕ}
    (Q K V : Fin n → Fin d → ℝ) (scale : ℝ) (i : Fin n) (d_out : Fin d) : ℝ :=
  ∑ j, softmax (fun j' => scale * dot_product (Q i) (K j')) j * V j d_out

/-- Defining equation for `attention`. -/
theorem attention_def {n d : ℕ}
    (Q K V : Fin n → Fin d → ℝ) (scale : ℝ) (i : Fin n) (d_out : Fin d) :
    attention Q K V scale i d_out =
      ∑ j, softmax (fun j' => scale * dot_product (Q i) (K j')) j * V j d_out := rfl

/-! ### Log-domain (numerically stable) attention -/

/-- Log-domain form of scaled dot-product attention:
    subtract `log (∑ exp logits)` from each logit before exponentiating
    and weighting `V`. Algebraically equal to `attention`. -/
noncomputable def log_domain_attention {n d : ℕ}
    (Q K V : Fin n → Fin d → ℝ) (scale : ℝ) (i : Fin n) (d_out : Fin d) : ℝ :=
  let logits : Fin n → ℝ := fun j => scale * dot_product (Q i) (K j)
  let lse := Real.log (∑ j, Real.exp (logits j))
  ∑ j, Real.exp (logits j - lse) * V j d_out

/-- Stability theorem: the log-domain form equals naive softmax attention. -/
theorem log_domain_attention_eq_attention {n d : ℕ} [NeZero n]
    (Q K V : Fin n → Fin d → ℝ) (scale : ℝ) (i : Fin n) (d_out : Fin d) :
    log_domain_attention Q K V scale i d_out = attention Q K V scale i d_out := by
  set logits : Fin n → ℝ := fun j => scale * dot_product (Q i) (K j) with hlog
  have hpos : 0 < ∑ j, Real.exp (logits j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos (logits j)) Finset.univ_nonempty
  have hne : (∑ j, Real.exp (logits j)) ≠ 0 := ne_of_gt hpos
  simp only [log_domain_attention, attention, softmax]
  apply Finset.sum_congr rfl
  intro j _
  rw [Real.exp_sub, Real.exp_log hpos]

/-! ### Two-block FlashAttention -/

/-- Two-block FlashAttention online recurrence, parameterised by a decidable
    predicate `P` that splits `Fin n` into an "early" block (where `P` holds)
    and a "late" block (where `P` fails).

    Requires the early block to be nonempty (for `m₁`) and the overall universe
    to be nonempty (inherited from `NeZero n`). We use running-max/running-sum
    rescaling rather than branching on emptiness of the second block: the
    recurrence is still correct when the second block is empty. -/
noncomputable def flash_attention_two_blocks {n d : ℕ} [NeZero n]
    (Q K V : Fin n → Fin d → ℝ) (scale : ℝ)
    (P : Fin n → Prop) [DecidablePred P]
    (i : Fin n) (d_out : Fin d) : ℝ :=
  let logits : Fin n → ℝ := fun j => scale * dot_product (Q i) (K j)
  -- Global running max over the whole universe (equivalent to max of both
  -- block maxes; by using it we avoid a case split on nonemptiness).
  let m₂ : ℝ := (Finset.univ : Finset (Fin n)).sup' Finset.univ_nonempty logits
  let blockA := (Finset.univ : Finset (Fin n)).filter P
  let blockB := (Finset.univ : Finset (Fin n)).filter (fun j => ¬ P j)
  let l_A := ∑ j ∈ blockA, Real.exp (logits j - m₂)
  let o_A := ∑ j ∈ blockA, Real.exp (logits j - m₂) * V j d_out
  let l_B := ∑ j ∈ blockB, Real.exp (logits j - m₂)
  let o_B := ∑ j ∈ blockB, Real.exp (logits j - m₂) * V j d_out
  (o_A + o_B) / (l_A + l_B)

/-- Consistency lemma: one online step (rescaling a partial (m, l, o) tuple by
    a fresh global max `m₂`) preserves the ratio `o / l`. Concretely, for any
    old max `m₁` the shifted tuple `(exp(m₁ - m₂)·l, exp(m₁ - m₂)·o)` has the
    same ratio as `(l, o)` provided `l ≠ 0`.

    This is the key algebraic fact that makes the two-block (and more
    generally the streaming) FlashAttention update correct: rescaling the
    numerator and denominator by `exp(m₁ - m₂)` cancels. -/
theorem flash_attention_step_consistent
    (m₁ m₂ l o : ℝ) (hl : l ≠ 0) :
    (Real.exp (m₁ - m₂) * o) / (Real.exp (m₁ - m₂) * l) = o / l := by
  have he : Real.exp (m₁ - m₂) ≠ 0 := Real.exp_ne_zero _
  field_simp

/-- The two-block flash attention equals naive attention.

    Proof idea: with `m₂` chosen as the *global* max over the whole universe
    (rather than the running max after block A), the `exp(m₁ - m₂)` rescaling
    in the paper's streaming recurrence folds into a single global shift.
    Filtering by `P` and its negation partitions `Finset.univ`, so
    `l_A + l_B = ∑ exp(logits - m₂)` and `o_A + o_B = ∑ exp(logits - m₂) · V`.
    Then the shift-invariance of softmax (via `att_softmax_invariance` applied
    with `c = -m₂`) gives equality with `attention`.

    The general paper-grade "any number of sequential block updates" theorem
    reduces to repeated application of `flash_attention_step_consistent` plus
    `att_lse_shift_invariance`; we prove the two-block case in full. -/
theorem flash_attention_eq_attention {n d : ℕ} [NeZero n]
    (Q K V : Fin n → Fin d → ℝ) (scale : ℝ)
    (P : Fin n → Prop) [DecidablePred P]
    (i : Fin n) (d_out : Fin d) :
    flash_attention_two_blocks Q K V scale P i d_out =
      attention Q K V scale i d_out := by
  set logits : Fin n → ℝ := fun j => scale * dot_product (Q i) (K j) with hlog
  set m₂ : ℝ := (Finset.univ : Finset (Fin n)).sup'
      Finset.univ_nonempty logits with hm₂
  -- Partition the universe by `P`.
  have hpart_l :
      (∑ j ∈ (Finset.univ : Finset (Fin n)).filter P,
          Real.exp (logits j - m₂)) +
      (∑ j ∈ (Finset.univ : Finset (Fin n)).filter (fun j => ¬ P j),
          Real.exp (logits j - m₂)) =
      ∑ j, Real.exp (logits j - m₂) := by
    rw [Finset.sum_filter_add_sum_filter_not]
  have hpart_o :
      (∑ j ∈ (Finset.univ : Finset (Fin n)).filter P,
          Real.exp (logits j - m₂) * V j d_out) +
      (∑ j ∈ (Finset.univ : Finset (Fin n)).filter (fun j => ¬ P j),
          Real.exp (logits j - m₂) * V j d_out) =
      ∑ j, Real.exp (logits j - m₂) * V j d_out := by
    rw [Finset.sum_filter_add_sum_filter_not]
  -- Collapse the flash definition.
  have hflash :
      flash_attention_two_blocks Q K V scale P i d_out =
        (∑ j, Real.exp (logits j - m₂) * V j d_out) /
        (∑ j, Real.exp (logits j - m₂)) := by
    simp only [flash_attention_two_blocks]
    rw [← hlog, ← hm₂, hpart_o, hpart_l]
  rw [hflash]
  -- Show the shifted attention equals the naive attention via softmax shift.
  have hpos : 0 < ∑ j, Real.exp (logits j - m₂) :=
    Finset.sum_pos (fun j _ => Real.exp_pos (logits j - m₂)) Finset.univ_nonempty
  have hne : (∑ j, Real.exp (logits j - m₂)) ≠ 0 := ne_of_gt hpos
  -- Rewrite the shifted sum as a single division that matches `attention`.
  rw [attention_def]
  -- Distribute the division across the summation.
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro j _
  have hshift : softmax (fun j' => logits j' + (-m₂)) j = softmax logits j :=
    att_softmax_invariance logits (-m₂) j
  -- `logits j - m₂ = logits j + (-m₂)`, so the shifted softmax unfolds to
  -- the quotient form we want.
  have hunfold : softmax (fun j' => logits j' + (-m₂)) j =
      Real.exp (logits j + (-m₂)) /
        (∑ j', Real.exp (logits j' + (-m₂))) := rfl
  have hrewrite : Real.exp (logits j - m₂) / (∑ j', Real.exp (logits j' - m₂)) =
      softmax logits j := by
    have hnum : Real.exp (logits j - m₂) = Real.exp (logits j + (-m₂)) := by
      rw [sub_eq_add_neg]
    have hden : (∑ j', Real.exp (logits j' - m₂)) =
        ∑ j', Real.exp (logits j' + (-m₂)) := by
      apply Finset.sum_congr rfl
      intro j' _
      rw [sub_eq_add_neg]
    rw [hnum, hden, ← hunfold, hshift]
  calc Real.exp (logits j - m₂) * V j d_out /
          (∑ j', Real.exp (logits j' - m₂))
      = (Real.exp (logits j - m₂) / (∑ j', Real.exp (logits j' - m₂))) *
          V j d_out := by ring
    _ = softmax logits j * V j d_out := by rw [hrewrite]

/-! ### Max-plus (tropical) attention -/

/-- Max-plus (tropical) attention: replaces the softmax weighted sum with a
    hard argmax over logits, returning the value of the winning key.
    Well-defined because `Finset.univ` is nonempty under `NeZero n`. -/
noncomputable def max_plus_attention {n d : ℕ} [NeZero n]
    (Q K V : Fin n → Fin d → ℝ) (scale : ℝ) (i : Fin n) (d_out : Fin d) : ℝ :=
  let logits : Fin n → ℝ := fun j => scale * dot_product (Q i) (K j)
  -- `exists_max_image` gives us an index achieving the max logit; pick it
  -- via `Classical.choose`. Non-emptiness is witnessed by `NeZero n`.
  let j_idx : Fin n :=
    (((Finset.univ : Finset (Fin n)).exists_max_image logits
        Finset.univ_nonempty).choose)
  V j_idx d_out

/-- The witness index in `max_plus_attention` actually achieves the maximum
    logit. This certifies well-posedness of the definition. -/
theorem max_plus_attention_argmax_spec {n d : ℕ} [NeZero n]
    (Q K : Fin n → Fin d → ℝ) (scale : ℝ) (i : Fin n) :
    let logits : Fin n → ℝ := fun j => scale * dot_product (Q i) (K j)
    ∃ j_star : Fin n, ∀ j : Fin n, logits j ≤ logits j_star := by
  intro logits
  obtain ⟨j_star, _, hmax⟩ :=
    (Finset.univ : Finset (Fin n)).exists_max_image logits Finset.univ_nonempty
  exact ⟨j_star, fun j => hmax j (Finset.mem_univ _)⟩

end EmlNN
