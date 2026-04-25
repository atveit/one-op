import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace EmlNN

/-- The EML Sheffer operator from Odrzywołek 2603.21852:
    eml(x, y) = exp(x) - ln(y). `Real.log` extends by 0 for
    non-positive args — callers track positivity where needed. -/
noncomputable def eml (x y : ℝ) : ℝ := Real.exp x - Real.log y

theorem eml_def (x y : ℝ) : eml x y = Real.exp x - Real.log y := rfl

example : eml 0 1 = 1 := by
  simp [eml, Real.exp_zero, Real.log_one]

/-- exp(x) = eml(x, 1). Depth-1 identity from the paper. -/
theorem eml_exp (x : ℝ) : eml x 1 = Real.exp x := by
  simp [eml, Real.log_one]

/-- The constant e arises as eml(1, 1). -/
theorem eml_e : eml 1 1 = Real.exp 1 := by
  simp [eml, Real.log_one]

end EmlNN
