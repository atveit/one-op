# Review by openai/gpt-5.4-nano\n\nBelow is a review of what these snippets *actually* prove/specify, whether they’re nontrivial/useful, and where logical gaps/edge cases remain. I’ll treat each language separately.

---

## 1) Lean 4 (Basic)

### What do they prove vs. what’s claimed?
This file defines
```lean
noncomputable def eml (x y : ℝ) : ℝ := Real.exp x - Real.log y
```
and then proves theorems that are essentially definitional/unfolding facts:

- `eml_def`: `eml x y = Real.exp x - Real.log y` is `rfl` (definition equality).
- `eml_exp`: `eml x 1 = Real.exp x` follows from `Real.log_one`.
- `eml_e`: `eml 1 1 = Real.exp 1` follows similarly.

**So they prove only the correctness of the definition and a couple of evaluation identities** (depth‑1 identity / constant e identity).

### Are these proofs interesting or useful?
They are **trivial but necessary**—good for building up a library. They are not verifying any nontrivial “EML operator properties” beyond substitution.

### Edge cases / invariants
The comment says:
> “Real.log extends by 0 for non-positive args — callers track positivity where needed.”

However, the theorems here are only for `y = 1`, so no problematic domain issues arise. If later proofs use `Real.log` of arbitrary `y` without maintaining positivity hypotheses, that’s where soundness would become tricky.

**Idiomatic Lean improvement:** For Basic, nothing major. For later theorems, expect to use lemmas like `Real.log_pos` / `Real.log_le_...` with explicit hypotheses `0 < y`.

---

## 2) Lean 4 (Attention)

### What do they actually prove?
The key theorem `att_softmax_invariance` claims the standard identity:
\[
\mathrm{softmax}(v + c)_i = \mathrm{softmax}(v)_i
\]
under `[NeZero n]` (so `n ≠ 0`) and for `i : Fin n`.

Looking at the proof:
- It proves `0 < ∑ j, exp (v j)` using `Finset.sum_pos` and `Real.exp_pos`.
- It proves the denominator is nonzero.
- It proves the algebraic sum identity:
  \[
  \sum_j e^{v_j + c} = e^c \sum_j e^{v_j}
  \]
  using `Finset.mul_sum` and `Real.exp_add`.
- Then it rewrites inside `softmax` and cancels the common factor using nonzero `Real.exp_ne_zero _`.

So **it does prove the softmax shift invariance pointwise** (assuming `softmax` is defined in the usual “exp normalized by sum exp” way over a `Finset.univ`).

The second theorem `att_lse_shift_invariance` is intended to prove:
\[
\log\left(\sum_i e^{v_i + c}\right) = c + \log\left(\sum_i e^{v_i}\right)
\]
but the snippet is truncated, so I can’t confirm the completed tactic sequence.

Still, the *shape* is correct: prove positivity of the sum, pull out `c` as a factor `e^c`, then use `Real.log_mul_eq_log ...` (or an equivalent lemma) with side conditions.

### Do they prove what the blog post claims?
- For `att_softmax_invariance`: **yes, in substance** (shift invariance, pointwise, for `Fin n → ℝ`), and it is not merely definitional.
- For the larger list in the comment (P3 composition, FlashAttention recurrence, tropical attention): **those are not actually present** in what you pasted. The only fully visible theorem is the softmax invariance, plus the start of LSE invariance.

So **they do not yet prove the broad “attention-family theorems” claimed in the blog post**, at least not from the material shown.

### Are these proofs interesting or useful, or are they trivial?
This one is **nontrivial and genuinely useful**:
- It handles the annoying proof obligations around **denominators** and **positivity** of sums of exponentials.
- It correctly uses `Real.exp_ne_zero` and cancellation.
- Shift invariance is a key property used for numeric stability and for reasoning about log-domain attention.

So it’s not just “simp rfl”; it’s doing the standard formalization work.

### Unhandled edge cases / logical soundness
Potential issues to watch:

1. **Division by 0 / log of nonpositive**
   - Softmax: the proof explicitly builds `0 < ∑ exp (v j)` and hence nonzero denominator. Good.
   - LSE: the log requires the argument to be strictly positive. The intended approach (as started) is to prove the sum of exponentials is `> 0`. That should work because `exp_pos` is strict and the sum is over a nonempty `Fin n` (hence `[NeZero n]` and `Finset.univ_nonempty`).

2. **`NeZero n` correctness**
   - If `n = 0`, there is no `Fin n` anyway, but the theorem still quantifies `v : Fin n → ℝ` and `i : Fin n`. Depending on how Lean handles `Fin 0 → ℝ`, the proposition could become vacuously true or ill-typed.
   - Requiring `[NeZero n]` is a reasonable guard to avoid empty sums and to get `univ_nonempty`.

3. **Algebraic rewriting uses `ring`**
   - The line:
     ```lean
     rw [Real.exp_add]; ring
     ```
     is likely fine but somewhat brittle. If `ring` is succeeding because the goal is purely multiplicative/algebraic, good. If it’s doing more than expected, it could be fragile to definition changes.

### More idiomatic tactics / suggestions
- For both proofs, you’d typically aim to:
  - use `by
      classical
      ...
    `
    if needed (Finset sums often trigger classical).
  - replace manual steps with library lemmas:
    - There are often lemmas in Mathlib like `sum_exp_add_const`-style results, or at least `Finset.sum_congr` with a clearer structure.
  - For cancellation in softmax:
    - ensure you’re using `mul_div_mul_left` (as you did) with a lemma that gives nonzero of the factor.

- A common simplification improvement:
  - Instead of proving `hsum` via `Finset.mul_sum` + `Finset.sum_congr`, you could sometimes do:
    \[
    \sum_j e^{v_j + c} = e^c \sum_j e^{v_j}
    \]
    by a direct `simp` using `Real.exp_add` and `Finset.mul_sum` if the simp normal form matches.

- For LSE:
  - In idiomatic Lean, you often see:
    - `have hpos : 0 < ... := ...`
    - then:
      - `have := Real.log_mul_eq_log hpos h...` (exact lemma names vary)
  - Ensure the lemma used expects `0 < a` not just `a ≠ 0` (Mathlib distinguishes `log` domain side conditions).

**Bottom line:** the softmax invariance proof is solid and nontrivial; the LSE proof is plausible but cannot be fully audited from the truncated snippet.

---

## 3) Coq

### What do they actually prove?
You have:
```coq
Theorem scaled_dot_linear : forall (q k : list R) (scale : R),
  scale * (dot_product q k) = dot_product (map (fun x => scale * x) q) k.
Proof.
  ...
Admitted.
```

**There is no proof**: it is `Admitted`, so Coq accepts the theorem as an axiom. Therefore:

- **It proves nothing.**
- The spec does not establish correctness of any compiled implementation, or even the mathematical lemma.

### Are these proofs interesting or useful?
As written: no—because it’s unproved.

However, the *statement* is potentially useful: it’s the “linearity” property that scaling queries scales the dot product. But the current dot product definition uses lists and `combine`, so you must be careful about length mismatches.

### Unhandled edge cases / logical soundness
There is a major edge case:

- `combine q k` truncates to the shorter list length in Coq.
- Therefore the theorem is not the usual linearity of dot product over aligned vectors; it is linearity over the *zipped portion*.
- The lemma may still be true (because scaling distributes over multiplication elementwise), but:
  - you need to ensure ring/scalar properties and how `fold_right` is defined for `Rplus`.
  - you need to specify what `R` is (your snippet imports `Reals.Reals` but doesn’t show `R` notation/instance; also the type class/operations need to be consistent).

More idiomatic useful spec would be:
- Use `Vector.t` (length-indexed) instead of `list` if you want a real dot product.
- Or explicitly state behavior under mismatched lengths.

### Proof effort / idiom suggestions
If you were to actually prove it:
- Use lemmas about `map`/`combine` and distributivity:
  - show that mapping scale over `q` results in each pair `(q_i,k_i)` being scaled on the q-component.
- Use rewriting under `fold_right` with distributivity:
  \[
  scale*(a+b)=scale*a+scale*b
  \]
  and
  \[
  scale*(x*y)=(scale*x)*y
  \]
- In Coq, this is often handled smoothly with `ring`/`nlinarith` depending on the algebraic structure, but for reals you may still need `lra`/`field_simplify` style tactics.

---

## 4) TLA+

### What do they actually prove?
TLA+ here is not a proof script; it is a **specification** of a state machine plus invariants written as formulas:
- `NoDoubleAllocation`
- `ConservationOfBlocks`
- `TypeOK`

But **you have not stated or checked that these invariants hold**. There’s no “theorems” or model-checking results. So it proves nothing in the mathematical sense; it defines candidate invariants.

If this were paired with a TLC run or a theorem proving setup (`TLC`/`TLAPS`), then you could say something like “these invariants are invariant under `Next`”. But that’s not included here.

### Do they prove what the blog post claims?
Not from the snippet. The blog claim (KV cache manager correctness under concurrent generation + compression/async behavior) is not addressed. The spec is a simplified allocator with sequential interleaving of `Allocate/Finish`.

Also, “TurboQuant compresses KV blocks” is mentioned but not modeled at all.

### Interesting/useful or trivial?
- The allocator core is **a common illustrative model**.
- It’s not trivial in TLA+ terms because `Next` is nondeterministic and concurrent requests are interleaved, so invariants are non-obvious.
- But **it’s incomplete** relative to real paged attention:
  - No tracking of which blocks are tied to which request beyond `allocations[r]`.
  - No reference counts or sharing semantics.
  - No modeling of compression/eviction correctness under reads.

### Unhandled edge cases / logical soundness issues

1. **`Allocate` breaks `NoDoubleAllocation` unless you carefully interpret updates**
   The `Allocate(r)` step updates `free_blocks'` and removes one `b` from `free_blocks`.
   If `NoDoubleAllocation` is truly meant to be about physical blocks not being in two different `allocations[...]` simultaneously, then `Allocate` seems intended to ensure it because it only draws from `free_blocks`.

   But there is a subtlety: `Finish(r)` frees blocks by reconstructing them from `allocations[r]`:
   ```tla
   free_blocks' = free_blocks \cup {b in 1..MaxBlocks : exists i ... allocations[r][i]=b}
   ```
   This is okay if `allocations[r]` contains only block IDs in `1..MaxBlocks`. But that’s not guaranteed unless `TypeOK` is established as an invariant.

2. **Conservation equation likely isn’t correct / too handwavy**
   `ConservationOfBlocks` uses:
   ```tla
   LET SumAlloc[r \in 0..NumRequests] == 
          IF r = 0 THEN 0 ELSE Len(allocations[r]) + SumAlloc[r-1]
    IN SumAlloc[NumRequests]
   ```
   This assumes:
   - `r` indexing works with numeric ranges,
   - `allocations` is defined for those `r`,
   - `0` is not in `1..NumRequests`.
   
   But your `allocations` and `state` are only defined for `r in 1..NumRequests`. TLA+ function domain checks matter: does `allocations[0]` even exist? You avoid it with `IF r=0 THEN 0`, but the recursion still mentions `allocations[r]` only when `r ≠ 0`, so perhaps safe—yet it’s still a fragile encoding.

   Also, conservation usually should be:
   \[
   |free\_blocks| + \sum_{r} Len(allocations[r]) = MaxBlocks
   \]
   but you must ensure blocks are neither duplicated nor lost across all allocations. That’s exactly the kind of coupling that invariants must establish consistently.

3. **No fairness / no liveness**
   Only safety invariants are hinted. If you care about “requests eventually finish” you need fairness conditions in TLA+.

4. **`Next` allows “Finish(r)” even if `allocations[r]` is empty**
   `Finish(r)` only checks `state[r] = "generating"`. It does not require `Len(allocations[r]) > 0`.
   Then:
   - `allocations' = <<>>` and
   - `free_blocks'` unions over elements found in `allocations[r]` (none),
   so it’s harmless.
   But it may not match the intended operational semantics: “Finish frees all its blocks” is still true, but it might allow finishing immediately without allocation, which may or may not be intended.

5. **TypeOK does not ensure uniqueness of blocks across allocations**
   `TypeOK` only constrains membership of `state` tags and that `free_blocks` is subset of valid IDs.
   It does not constrain `allocations[r]` contents to `1..MaxBlocks` nor uniqueness/no overlap. That is left to `NoDoubleAllocation`, but again `NoDoubleAllocation` assumes lengths are nonzero and uses indices within `Len(...)`.

### Idiomatic improvements / what to add
- Add:
  - an explicit type invariant for allocations contents:
    ```tla
    allocations[r] \in Seq(1..MaxBlocks)
    ```
    or equivalent formula.
- Strengthen `NoDoubleAllocation` to mention `i`/`j` range and ensure it matches the actual “no free block is simultaneously allocated” semantics.
- Model `Allocate` nondeterminism more clearly:
  - e.g., choose a single `b` rather than quantifying `\E b` and updating inside; TLA+ updates with primes should be structured so each action has a well-defined chosen `b`.
- Include model checking goals or a statement in TLAPS:
  - “Assume Init; prove invariants hold under Next”:
    \[
    Init \models Inv \quad\text{and}\quad Inv \wedge [Next] \models Inv'
    \]
  This is the part currently missing.

---

## Overall conclusions

1. **Do they prove what is claimed in the blog post?**
   - Lean Basic: proves only definition/evaluation identities for `eml`.
   - Lean Attention: at least one nontrivial theorem is correctly aiming at softmax shift invariance; the broader set of claimed results is not shown.
   - Coq: nothing is proven (`Admitted`).
   - TLA+: only invariants are stated; no check/proof that they actually hold.

2. **Are the proofs interesting/useful or trivial?**
   - Lean Basic: mostly trivial (but groundwork).
   - Lean Attention softmax invariance: **useful and nontrivial**—it encodes standard numeric-stability properties with correct side conditions.
   - Coq: trivial/empty because there is no proof.
   - TLA+: potentially useful as a starting model, but incomplete and not connected to actual correctness claims.

3. **Edge cases / invariant soundness / idiom**
   - Lean attention: likely sound but you must be vigilant that every `log`/division has strict positivity/nonzero hypotheses.
   - Coq: list length mismatch semantics via `combine` is a major edge case; also the theorem as given is unproved.
   - TLA+: conservation/type assumptions likely need strengthening; invariants are not proven. Also, modeling choices (finish can happen immediately, no fairness, no compression behavior) may diverge from the blog’s intended system.

If you paste the remainder of the Lean Attention file (especially the full `att_lse_shift_invariance` and any FlashAttention/tropical parts), I can do a much more concrete audit of each invariant and whether the proofs are robust or brittle.