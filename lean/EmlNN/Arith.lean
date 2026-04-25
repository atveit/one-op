import EmlNN.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace EmlNN

open Real

/-- Paper eq. 5: `ln z = eml 1 (eml (eml 1 z) 1)` for `z > 0`.
    Unfold: inner `eml 1 z = e - log z`; then `eml _ 1 = exp _`;
    outer `eml 1 (...) = e - log(exp(e - log z)) = e - (e - log z) = log z`. -/
theorem eml_ln (z : ℝ) (hz : 0 < z) :
    Real.log z = eml 1 (eml (eml 1 z) 1) := by
  simp [eml, Real.log_one, Real.log_exp]

/-- `e = eml 1 1`. -/
theorem eml_e_eq : Real.exp 1 = eml 1 1 := eml_e.symm

/-- `x * y = exp(log x + log y)` for positive `x, y`. -/
theorem eml_mul (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    x * y = Real.exp (Real.log x + Real.log y) := by
  rw [Real.exp_add, Real.exp_log hx, Real.exp_log hy]

/-- `1 / z = exp(- log z)` for positive `z`. -/
theorem eml_recip (z : ℝ) (hz : 0 < z) :
    1 / z = Real.exp (- Real.log z) := by
  rw [Real.exp_neg, Real.exp_log hz, one_div]

/-- `x / y = exp(log x - log y)` for positive `x, y`. -/
theorem eml_div (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    x / y = Real.exp (Real.log x - Real.log y) := by
  rw [Real.exp_sub, Real.exp_log hx, Real.exp_log hy, div_eq_mul_inv]

/-- `sqrt x = exp(log x / 2)` for positive `x`. -/
theorem eml_sqrt (x : ℝ) (hx : 0 < x) :
    Real.sqrt x = Real.exp (Real.log x / 2) := by
  rw [Real.sqrt_eq_rpow, Real.rpow_def_of_pos hx]
  ring_nf

/-- `eml 0 (e · z) = - log z` for positive `z`. -/
theorem eml_neg_log (z : ℝ) (hz : 0 < z) :
    eml 0 (Real.exp 1 * z) = - Real.log z := by
  have he : (0 : ℝ) < Real.exp 1 := Real.exp_pos 1
  rw [eml, Real.exp_zero,
      Real.log_mul (ne_of_gt he) (ne_of_gt hz), Real.log_exp]
  ring

/-- `x + y = log(exp x · exp y)`. -/
theorem eml_add_log (x y : ℝ) :
    x + y = Real.log (Real.exp x * Real.exp y) := by
  rw [Real.log_mul (Real.exp_ne_zero x) (Real.exp_ne_zero y),
      Real.log_exp, Real.log_exp]

end EmlNN
