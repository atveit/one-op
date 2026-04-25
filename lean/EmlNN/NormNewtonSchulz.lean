import EmlNN.Basic
import EmlNN.Arith
import EmlNN.Norm
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace EmlNN

open Real

/-! ## Newton-Schulz rsqrt — dual-space fix for additive fragility.

    The iteration ``y ↦ 0.5 · y · (3 - x · y²)`` has the exact inverse
    square root ``1/√x`` as a fixed point and converges quadratically.
    Because the iteration uses only addition and multiplication, it
    complements the log-domain Min-Plus construction: Min-Plus wins on
    multiplicative fragility, Newton-Schulz wins on additive fragility
    (norm variance sums). -/

/-- One Newton-Schulz rsqrt refinement step. -/
noncomputable def newton_schulz_rsqrt_iteration (x y : ℝ) : ℝ :=
  (1 / 2) * y * (3 - x * y * y)

/-- The exact inverse-square-root is a fixed point of the iteration:
    if ``y = 1/√x`` for ``x > 0`` then the step returns ``y`` unchanged.
    Proof: ``x · y² = 1`` so the bracket becomes ``3 - 1 = 2`` and the
    prefactor ``1/2`` cancels, leaving ``y``. -/
theorem newton_schulz_fixed_point (x : ℝ) (hx : 0 < x) :
    newton_schulz_rsqrt_iteration x (1 / Real.sqrt x) = 1 / Real.sqrt x := by
  unfold newton_schulz_rsqrt_iteration
  have hs : Real.sqrt x ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hx)
  have hsq : Real.sqrt x * Real.sqrt x = x := Real.mul_self_sqrt hx.le
  -- Show x · (1/√x)² = 1 directly, then the bracket becomes 2.
  have hxy2 : x * (1 / Real.sqrt x) * (1 / Real.sqrt x) = 1 := by
    rw [show x * (1 / Real.sqrt x) * (1 / Real.sqrt x)
        = x / (Real.sqrt x * Real.sqrt x) from by ring, hsq]
    exact div_self (ne_of_gt hx)
  have : (1 / 2 : ℝ) * (1 / Real.sqrt x) * (3 - x * (1 / Real.sqrt x) * (1 / Real.sqrt x))
       = (1 / 2) * (1 / Real.sqrt x) * (3 - 1) := by rw [hxy2]
  rw [this]; ring

/-- Qualitative quadratic convergence: the residual ``r := x · y² - 1``
    satisfies ``r_{k+1} = -(3/4) · r_k² - (1/4) · r_k³``, so
    ``|r_{k+1}| ≤ |r_k|²`` whenever ``|r_k| ≤ 1``.  This captures the
    quadratic rate of the Newton-Schulz iteration. -/
theorem newton_schulz_quadratic_convergence (x y : ℝ)
    (h : |x * y * y - 1| ≤ 1) :
    |x * (newton_schulz_rsqrt_iteration x y) * (newton_schulz_rsqrt_iteration x y) - 1|
      ≤ |x * y * y - 1| ^ 2 := by
  -- Set r = x · y² − 1.  Direct algebra gives
  --   x · (y')² − 1  =  − r² · (3 + r) / 4
  -- where y' is the updated value.  Then |−r²(3+r)/4| = r² · |3+r| / 4.
  -- From |r| ≤ 1 we get 2 ≤ 3 + r ≤ 4, so |3 + r| / 4 ≤ 1 and the bound follows.
  set r : ℝ := x * y * y - 1 with hr_def
  have hy' : x * (newton_schulz_rsqrt_iteration x y)
        * (newton_schulz_rsqrt_iteration x y) - 1
      = r^2 * (r - 3) / 4 := by
    unfold newton_schulz_rsqrt_iteration
    have hr : r = x * y * y - 1 := hr_def
    have : r^2 * (r - 3) / 4
         = (x*y*y - 1)^2 * ((x*y*y - 1) - 3) / 4 := by rw [hr]
    rw [this]
    ring
  rw [hy']
  have h1 : |r^2 * (r - 3) / 4| = r^2 * |r - 3| / 4 := by
    have h4 : |(4 : ℝ)| = 4 := abs_of_pos (by norm_num)
    rw [abs_div, h4, abs_mul, abs_of_nonneg (sq_nonneg r)]
  rw [h1]
  have h3r_abs : |r - 3| ≤ 4 := by
    rw [abs_le] at h ⊢
    constructor <;> linarith [h.1, h.2]
  have hr2_nn : 0 ≤ r^2 := sq_nonneg r
  have hmul : r^2 * |r - 3| ≤ r^2 * 4 :=
    mul_le_mul_of_nonneg_left h3r_abs hr2_nn
  have hgoal : r^2 * |r - 3| / 4 ≤ r^2 := by
    have : r^2 * |r - 3| / 4 ≤ r^2 * 4 / 4 := by linarith
    linarith [this]
  have habs : |r| ^ 2 = r^2 := sq_abs r
  linarith [hgoal, habs]

/-- LayerNorm computed with the Newton-Schulz-converged rsqrt equals the
    exact LayerNorm.  Because ``1/√(var + ε)`` is the fixed point of the
    iteration, substituting any converged value (the fixed point itself,
    or the limit of the iteration) reproduces LayerNorm exactly. -/
theorem layer_norm_ns_eq_layer_norm_at_fixed_point {n : ℕ} [NeZero n]
    (x : Fin n → ℝ) (γ β : Fin n → ℝ) (ε : ℝ)
    (hvar_pos : 0 < (∑ j, (x j - (∑ j', x j') / n) ^ 2) / n + ε) (i : Fin n) :
    let μ := (∑ j, x j) / n
    let variance := (∑ j, (x j - μ) ^ 2) / n
    let y := 1 / Real.sqrt (variance + ε)
    -- At the fixed point the iteration is identity.
    γ i * ((x i - μ) * newton_schulz_rsqrt_iteration (variance + ε) y) + β i
      = layer_norm x γ β ε i := by
  intro μ variance y
  have hfp : newton_schulz_rsqrt_iteration (variance + ε) y = y :=
    newton_schulz_fixed_point _ hvar_pos
  rw [hfp]
  change γ i * ((x i - μ) * (1 / Real.sqrt (variance + ε))) + β i
     = layer_norm x γ β ε i
  rw [layer_norm_def]
  have hμ : (∑ j, x j) / (n : ℝ) = μ := rfl
  rw [hμ]
  congr 1
  rw [mul_one_div]

end EmlNN
