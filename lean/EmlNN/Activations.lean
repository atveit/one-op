import EmlNN.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset

namespace EmlNN

open Real
open scoped BigOperators

/-- Sigmoid expressed with the EML operator. -/
theorem sigmoid_from_eml (x : ℝ) :
    1 / (1 + Real.exp (-x)) = 1 / (1 + eml (-x) 1) := by
  rw [eml_exp]

/-- Sigmoid is strictly positive. -/
theorem sigmoid_pos (x : ℝ) : 0 < 1 / (1 + Real.exp (-x)) := by
  have h : 0 < 1 + Real.exp (-x) := by
    have := Real.exp_pos (-x)
    linarith
  exact one_div_pos.mpr h

/-- Tanh in terms of `exp`. -/
theorem tanh_from_exp (x : ℝ) :
    Real.tanh x = (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x)) :=
  Real.tanh_eq x

/-- Softplus via the EML operator. -/
theorem softplus_from_eml (x : ℝ) :
    Real.log (1 + Real.exp x) = Real.log (1 + eml x 1) := by
  rw [eml_exp]

/-- Exact ReLU from `√(x²)`. -/
theorem relu_exact_from_sqrt (x : ℝ) :
    (x + Real.sqrt (x^2)) / 2 = max x 0 := by
  rw [Real.sqrt_sq_eq_abs]
  rcases le_or_gt 0 x with hx | hx
  · rw [abs_of_nonneg hx, max_eq_left hx]; ring
  · rw [abs_of_neg hx, max_eq_right hx.le]; ring

/-- Softmax. -/
noncomputable def softmax {n : ℕ} (v : Fin n → ℝ) (i : Fin n) : ℝ :=
  Real.exp (v i) / ∑ j, Real.exp (v j)

theorem softmax_eq_exp_div_sum {n : ℕ} (v : Fin n → ℝ) (i : Fin n) :
    softmax v i = Real.exp (v i) / ∑ j, Real.exp (v j) := rfl

/-- Softmax outputs sum to 1. -/
theorem softmax_sum_one {n : ℕ} [NeZero n] (v : Fin n → ℝ) :
    (∑ i, softmax v i) = 1 := by
  have hpos : 0 < ∑ j, Real.exp (v j) := by
    refine Finset.sum_pos (fun j _ => Real.exp_pos (v j)) ?_
    exact Finset.univ_nonempty
  have hne : (∑ j, Real.exp (v j)) ≠ 0 := ne_of_gt hpos
  simp only [softmax]
  rw [← Finset.sum_div, div_self hne]

/-- SiLU / Swish: `x * σ(x)` expressed via the EML sigmoid form. -/
theorem silu_from_eml (x : ℝ) :
    x * (1 / (1 + Real.exp (-x))) = x * (1 / (1 + eml (-x) 1)) := by
  rw [sigmoid_from_eml]

/-- SwiGLU gating with SiLU: `swiglu g v i = SiLU(g_i) * v_i`. -/
noncomputable def swiglu {n : ℕ} (g v : Fin n → ℝ) (i : Fin n) : ℝ :=
  (g i * (1 / (1 + Real.exp (-(g i))))) * v i

theorem swiglu_def {n : ℕ} (g v : Fin n → ℝ) (i : Fin n) :
    swiglu g v i = (g i * (1 / (1 + Real.exp (-(g i))))) * v i := rfl

/-- GELU via the tanh approximation. -/
noncomputable def gelu (x : ℝ) : ℝ :=
  0.5 * x * (1 + Real.tanh (Real.sqrt (2 / Real.pi) * (x + 0.044715 * x^3)))

/-- GeGLU: `geglu x y = gelu x * y`. -/
noncomputable def geglu (x y : ℝ) : ℝ := gelu x * y

theorem geglu_def (x y : ℝ) : geglu x y = gelu x * y := rfl

/-- Mish: `mish x = x * tanh(softplus x)`. -/
noncomputable def mish (x : ℝ) : ℝ := x * Real.tanh (Real.log (1 + Real.exp x))

theorem mish_def (x : ℝ) :
    mish x = x * Real.tanh (Real.log (1 + Real.exp x)) := rfl

/-- Log-softmax. -/
noncomputable def log_softmax {n : ℕ} (v : Fin n → ℝ) (i : Fin n) : ℝ :=
  v i - Real.log (∑ j, Real.exp (v j))

/-- Log-softmax equals the log of softmax (pointwise). -/
theorem log_softmax_def {n : ℕ} [NeZero n] (v : Fin n → ℝ) (i : Fin n) :
    log_softmax v i = Real.log (softmax v i) := by
  have hpos : 0 < ∑ j, Real.exp (v j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos (v j)) Finset.univ_nonempty
  simp only [log_softmax, softmax]
  rw [Real.log_div (Real.exp_ne_zero _) (ne_of_gt hpos), Real.log_exp]

/-- Softmax is invariant under additive shifts of its input. -/
theorem softmax_invariance {n : ℕ} [NeZero n] (v : Fin n → ℝ) (c : ℝ) :
    softmax (fun i => v i + c) = softmax v := by
  funext i
  have hpos : 0 < ∑ j, Real.exp (v j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos (v j)) Finset.univ_nonempty
  have hec : Real.exp c ≠ 0 := Real.exp_ne_zero _
  simp only [softmax]
  have hsum : ∑ j, Real.exp (v j + c) = Real.exp c * ∑ j, Real.exp (v j) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [Real.exp_add]; ring
  rw [hsum, Real.exp_add]
  field_simp

/-- Log-sum-exp shift invariance: adding `c` to every input shifts LSE by `c`. -/
theorem lse_shift_invariance {n : ℕ} [NeZero n] (v : Fin n → ℝ) (c : ℝ) :
    Real.log (∑ i, Real.exp (v i + c)) = c + Real.log (∑ i, Real.exp (v i)) := by
  have hpos : 0 < ∑ j, Real.exp (v j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos (v j)) Finset.univ_nonempty
  have hsum : ∑ i, Real.exp (v i + c) = Real.exp c * ∑ i, Real.exp (v i) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [Real.exp_add]; ring
  rw [hsum, Real.log_mul (Real.exp_ne_zero _) (ne_of_gt hpos), Real.log_exp]

end EmlNN
