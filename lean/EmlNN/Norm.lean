import EmlNN.Basic
import EmlNN.Arith
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.BigOperators.Field

namespace EmlNN

open Real
open scoped BigOperators

/-! ## Normalization-family theorems for the EML neural-network paper.

    Formal identities and definitions for LayerNorm, RMSNorm, GroupNorm,
    InstanceNorm, and (inference-time) BatchNorm. Each definition is a
    direct translation of the classical formula; each "_def" theorem is
    `rfl`-level and exists to pin the unfolding behaviour for downstream
    proofs. Where algebraic properties require positivity of the
    variance-plus-ε denominator we carry the assumption explicitly. -/

/-! ### LayerNorm -/

/-- LayerNorm over a feature vector `x : Fin n → ℝ`, with per-feature
    affine parameters `γ` (scale) and `β` (bias) and a numerical
    stability constant `ε`. -/
noncomputable def layer_norm {n : ℕ} [NeZero n]
    (x : Fin n → ℝ) (γ β : Fin n → ℝ) (ε : ℝ) : Fin n → ℝ :=
  let μ := (∑ j, x j) / n
  let variance := (∑ j, (x j - μ)^2) / n
  fun i => γ i * ((x i - μ) / Real.sqrt (variance + ε)) + β i

/-- Defining equation for `layer_norm`. -/
theorem layer_norm_def {n : ℕ} [NeZero n]
    (x : Fin n → ℝ) (γ β : Fin n → ℝ) (ε : ℝ) (i : Fin n) :
    layer_norm x γ β ε i =
      γ i * ((x i - (∑ j, x j) / n) /
        Real.sqrt ((∑ j, (x j - (∑ j, x j) / n)^2) / n + ε)) + β i := rfl

/-- With `γ ≡ 1` and `β ≡ 0`, the sum of LayerNorm outputs is zero:
    the numerator telescopes via `∑(x_i − μ) = (∑ x_i) − n·μ = 0`. -/
theorem layer_norm_zero_mean {n : ℕ} [NeZero n]
    (x : Fin n → ℝ) (ε : ℝ) (_hε : 0 < ε) :
    (∑ i, layer_norm x (fun _ => 1) (fun _ => 0) ε i) = 0 := by
  set μ : ℝ := (∑ j, x j) / n with hμ
  set variance : ℝ := (∑ j, (x j - μ)^2) / n with hvar
  set s : ℝ := Real.sqrt (variance + ε) with hs
  have hn : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne n)
  have hsum_sub : (∑ i, (x i - μ)) = 0 := by
    rw [Finset.sum_sub_distrib]
    simp [hμ, Finset.sum_const, mul_div_cancel₀ _ hn]
  have hexpand : (∑ i, layer_norm x (fun _ => 1) (fun _ => 0) ε i) =
      (∑ i, (x i - μ)) / s := by
    simp only [layer_norm, ← hμ, ← hvar, ← hs]
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hexpand, hsum_sub, zero_div]

/-! ### RMSNorm -/

/-- Root-mean-square normalization (Zhang & Sennrich 2019): no mean
    subtraction, divide by the RMS of the feature vector. -/
noncomputable def rms_norm {n : ℕ} [NeZero n]
    (x : Fin n → ℝ) (γ : Fin n → ℝ) (ε : ℝ) : Fin n → ℝ :=
  let rms_sq := (∑ j, (x j)^2) / n
  fun i => γ i * x i / Real.sqrt (rms_sq + ε)

/-- Defining equation for `rms_norm`. -/
theorem rms_norm_def {n : ℕ} [NeZero n]
    (x : Fin n → ℝ) (γ : Fin n → ℝ) (ε : ℝ) (i : Fin n) :
    rms_norm x γ ε i =
      γ i * x i / Real.sqrt ((∑ j, (x j)^2) / n + ε) := rfl

/-- RMSNorm of the zero vector is zero: the numerator `γ i · 0` vanishes. -/
theorem rms_norm_preserves_zero {n : ℕ} [NeZero n]
    (γ : Fin n → ℝ) (ε : ℝ) (i : Fin n) :
    rms_norm (fun _ => 0) γ ε i = 0 := by
  simp [rms_norm]

/-! ### GroupNorm -/

/-- GroupNorm: LayerNorm restricted to a subset `G : Finset (Fin n)`
    of features (the "group"). Mean and variance are taken over `G`,
    and the result is defined on indices `i ∈ G`; outside the group
    we return the scale times the raw value plus the bias (classical
    GroupNorm leaves channels in other groups to their own groups). -/
noncomputable def group_norm {n : ℕ} (G : Finset (Fin n)) (hG : G.Nonempty)
    (x : Fin n → ℝ) (γ β : Fin n → ℝ) (ε : ℝ) : Fin n → ℝ :=
  let k : ℝ := G.card
  let μ := (∑ j ∈ G, x j) / k
  let variance := (∑ j ∈ G, (x j - μ)^2) / k
  fun i =>
    if i ∈ G then
      γ i * ((x i - μ) / Real.sqrt (variance + ε)) + β i
    else
      γ i * x i + β i
  -- `hG` is carried so downstream lemmas can recover nonemptiness.
  -- (Unused in the `rfl`-level unfold below.)

/-- Defining equation for `group_norm` on indices inside the group. -/
theorem group_norm_def_mem {n : ℕ} (G : Finset (Fin n)) (hG : G.Nonempty)
    (x : Fin n → ℝ) (γ β : Fin n → ℝ) (ε : ℝ) {i : Fin n} (hi : i ∈ G) :
    group_norm G hG x γ β ε i =
      γ i * ((x i - (∑ j ∈ G, x j) / G.card) /
        Real.sqrt ((∑ j ∈ G, (x j - (∑ j ∈ G, x j) / G.card)^2) / G.card + ε))
        + β i := by
  simp [group_norm, hi]

/-! ### InstanceNorm -/

/-- InstanceNorm: LayerNorm applied per-sample over the spatial axis.
    For a single instance this coincides with `layer_norm`; we record
    this as a definitional synonym so the neural-network taxonomy is
    reflected in the formal development. -/
noncomputable def instance_norm {n : ℕ} [NeZero n]
    (x : Fin n → ℝ) (γ β : Fin n → ℝ) (ε : ℝ) : Fin n → ℝ :=
  layer_norm x γ β ε

/-- InstanceNorm is definitionally LayerNorm on a single instance. -/
theorem instance_norm_eq_layer_norm {n : ℕ} [NeZero n]
    (x : Fin n → ℝ) (γ β : Fin n → ℝ) (ε : ℝ) :
    instance_norm x γ β ε = layer_norm x γ β ε := rfl

/-! ### BatchNorm (inference) -/

/-- Inference-time BatchNorm: using fixed running statistics `μhat` and
    `σ̂²` accumulated during training, the per-feature transformation
    is `(x − μhat)/√(σ̂² + ε) · γ + β`. -/
noncomputable def batch_norm_inference {n : ℕ}
    (x : Fin n → ℝ) (μhat sigmasq : Fin n → ℝ) (γ β : Fin n → ℝ) (ε : ℝ) :
    Fin n → ℝ :=
  fun i => γ i * ((x i - μhat i) / Real.sqrt (sigmasq i + ε)) + β i

/-- Defining equation for `batch_norm_inference`. -/
theorem batch_norm_inference_def {n : ℕ}
    (x : Fin n → ℝ) (μhat sigmasq : Fin n → ℝ) (γ β : Fin n → ℝ) (ε : ℝ) (i : Fin n) :
    batch_norm_inference x μhat sigmasq γ β ε i =
      γ i * ((x i - μhat i) / Real.sqrt (sigmasq i + ε)) + β i := rfl

/-! ### EML connection theorems

    The `sqrt` in every normalization layer above is the `eml_sqrt`
    identity from `Arith.lean`: `√x = exp(log x / 2)` for `x > 0`. -/

/-- LayerNorm expressed with the EML-identity form of `sqrt`. -/
theorem layer_norm_via_eml_sqrt {n : ℕ} [NeZero n]
    (x : Fin n → ℝ) (γ β : Fin n → ℝ) (ε : ℝ)
    (hvar_pos :
      0 < (∑ j, (x j - (∑ j', x j') / n) ^ 2) / n + ε)
    (i : Fin n) :
    layer_norm x γ β ε i =
      γ i * ((x i - (∑ j, x j) / n) /
        Real.exp (Real.log
          ((∑ j, (x j - (∑ j', x j') / n) ^ 2) / n + ε) / 2)) + β i := by
  rw [layer_norm_def, eml_sqrt _ hvar_pos]

/-- RMSNorm expressed with the EML-identity form of `sqrt`. -/
theorem rms_norm_via_eml_sqrt {n : ℕ} [NeZero n]
    (x : Fin n → ℝ) (γ : Fin n → ℝ) (ε : ℝ)
    (hrms_pos : 0 < (∑ j, (x j) ^ 2) / n + ε) (i : Fin n) :
    rms_norm x γ ε i =
      γ i * x i /
        Real.exp (Real.log ((∑ j, (x j) ^ 2) / n + ε) / 2) := by
  rw [rms_norm_def, eml_sqrt _ hrms_pos]

end EmlNN
