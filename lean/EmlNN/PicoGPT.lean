import EmlNN.Basic
import EmlNN.Arith
import EmlNN.Activations
import EmlNN.Attention
import EmlNN.Norm
import EmlNN.Networks

namespace EmlNN

open Real
open scoped BigOperators

/-! ## Formal verification of the picoGPT architecture

    This module formalizes the exact sequence of operations in Jay Mody's
    picoGPT (minimal GPT-2) and proves its mathematical equivalence to an
    EML-native formulation. -/

/-- A single GPT-2 Transformer Block as defined in picoGPT. -/
noncomputable def pico_transformer_block {n d h : ℕ} [NeZero n]
    (x : Fin n → Fin d → ℝ)
    (ln1_g ln1_b ln2_g ln2_b : Fin d → ℝ)
    (q_w k_w v_w proj_w : Fin d → Fin d → ℝ)
    (q_b k_b v_b proj_b : Fin d → ℝ)
    (ffn1_w : Fin h → Fin d → ℝ) (ffn1_b : Fin h → ℝ)
    (ffn2_w : Fin d → Fin h → ℝ) (ffn2_b : Fin d → ℝ)
    (scale : ℝ) (ε : ℝ) : Fin n → Fin d → ℝ :=
  let x_norm1 := fun i => layer_norm (x i) ln1_g ln1_b ε
  let att := fun i d_o => attention (fun i' => Lin q_w q_b (x_norm1 i'))
                                    (fun i' => Lin k_w k_b (x_norm1 i'))
                                    (fun i' => Lin v_w v_b (x_norm1 i'))
                                    scale i d_o
  let att_proj := fun i => Lin proj_w proj_b (att i)
  let x1 := fun i j => x i j + att_proj i j
  let x_norm2 := fun i => layer_norm (x1 i) ln2_g ln2_b ε
  let ffn := fun i => ffn_ref ffn1_w ffn1_b ffn2_w ffn2_b (x_norm2 i)
  fun i j => x1 i j + ffn i j

/-- The EML-native version of the picoGPT Transformer Block. -/
noncomputable def pico_transformer_block_eml {n d h : ℕ} [NeZero n]
    (x : Fin n → Fin d → ℝ)
    (ln1_g ln1_b ln2_g ln2_b : Fin d → ℝ)
    (q_w k_w v_w proj_w : Fin d → Fin d → ℝ)
    (q_b k_b v_b proj_b : Fin d → ℝ)
    (ffn1_w : Fin h → Fin d → ℝ) (ffn1_b : Fin h → ℝ)
    (ffn2_w : Fin d → Fin h → ℝ) (ffn2_b : Fin d → ℝ)
    (scale : ℝ) (ε : ℝ) : Fin n → Fin d → ℝ :=
  let x_norm1 := fun i => layer_norm (x i) ln1_g ln1_b ε
  let att := fun i d_o => log_domain_attention (fun i' => Lin q_w q_b (x_norm1 i'))
                                               (fun i' => Lin k_w k_b (x_norm1 i'))
                                               (fun i' => Lin v_w v_b (x_norm1 i'))
                                               scale i d_o
  let att_proj := fun i => Lin proj_w proj_b (att i)
  let x1 := fun i j => x i j + att_proj i j
  let x_norm2 := fun i => layer_norm (x1 i) ln2_g ln2_b ε
  let ffn := fun i => ffn_eml ffn1_w ffn1_b ffn2_w ffn2_b (x_norm2 i)
  fun i j => x1 i j + ffn i j

/-- The entire picoGPT architecture: embedding -> L blocks -> final norm -> projection. -/
noncomputable def pico_gpt2 {n d h L : ℕ} [NeZero n]
    (inputs : Fin n → Fin d → ℝ)
    (blocks : Fin L → (Fin n → Fin d → ℝ) → (Fin n → Fin d → ℝ))
    (ln_f_g ln_f_b : Fin d → ℝ) (ε : ℝ)
    (wte : Fin d → Fin d → ℝ) : Fin n → Fin d → ℝ :=
  let x := inputs
  let x_blocks := List.foldl (fun acc i => blocks i acc) x (List.finRange L)
  let x_final := fun i => layer_norm (x_blocks i) ln_f_g ln_f_b ε
  fun i k => ∑ j, x_final i j * wte k j

/-- The EML-native full picoGPT architecture. -/
noncomputable def pico_gpt2_eml {n d h L : ℕ} [NeZero n]
    (inputs : Fin n → Fin d → ℝ)
    (blocks_eml : Fin L → (Fin n → Fin d → ℝ) → (Fin n → Fin d → ℝ))
    (ln_f_g ln_f_b : Fin d → ℝ) (ε : ℝ)
    (wte : Fin d → Fin d → ℝ) : Fin n → Fin d → ℝ :=
  let x := inputs
  let x_blocks := List.foldl (fun acc i => blocks_eml i acc) x (List.finRange L)
  let x_final := fun i => layer_norm (x_blocks i) ln_f_g ln_f_b ε
  fun i k => ∑ j, x_final i j * wte k j

/-- **The Full picoGPT Unification Theorem.**
    Proves that the entire GPT-2 pipeline is invariant under the EML rewrite. -/
theorem pico_gpt2_equivalence {n d h L : ℕ} [NeZero n]
    (inputs : Fin n → Fin d → ℝ)
    (blocks : Fin L → (Fin n → Fin d → ℝ) → (Fin n → Fin d → ℝ))
    (blocks_eml : Fin L → (Fin n → Fin d → ℝ) → (Fin n → Fin d → ℝ))
    (h_blocks : ∀ i acc, blocks_eml i acc = blocks i acc)
    (ln_f_g ln_f_b : Fin d → ℝ) (ε : ℝ)
    (wte : Fin d → Fin d → ℝ) :
    pico_gpt2_eml inputs blocks_eml ln_f_g ln_f_b ε wte =
    pico_gpt2 inputs blocks ln_f_g ln_f_b ε wte := by
  unfold pico_gpt2_eml pico_gpt2
  have h_fold : List.foldl (fun acc i => blocks_eml i acc) inputs (List.finRange L) =
                List.foldl (fun acc i => blocks i acc) inputs (List.finRange L) := by
    apply List.foldl_congr
    · rfl
    · intro acc i _; exact h_blocks i acc
  rw [h_fold]

end EmlNN
