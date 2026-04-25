# Review by google/gemini-3.1-pro-preview\n\nAs a senior formal methods researcher, I have reviewed the provided specifications. My overall assessment is that these snippets represent a **very preliminary sketch** or "specification washing" of a blog post, rather than a mature, verified system. The gap between what is claimed in the comments and what is actually proven in the code is massive.

Here is my detailed review broken down by your questions.

---

### 1. What do these actually prove? Do they prove what is claimed?

**LEAN 4 (Basic):**
*   **What it proves:** It verifies trivial algebraic identities (e.g., $e^x - \log(1) = e^x$ and $e^1 - \log(1) = e$) using Lean’s axiomatic Real numbers (`ℝ`).
*   **Versus claims:** The comments are accurate to the code, but the code does not prove anything meaningful about the proposed "EML Neural Network" architecture itself, just basic properties of its activation function’s definition.

**LEAN 4 (Attention):**
*   **What it proves:** It proves that standard Softmax and Log-Sum-Exp (LSE) are translation invariant mathematically (i.e., over `ℝ`). 
*   **Versus claims:** There is a **severe mismatch** here. The header claims to formalize a "two-block FlashAttention-style online recurrence" and "max-plus tropical attention." None of that is in the provided code. The code only contains mathematically standard proofs for $\text{softmax}(x+c)$.

**COQ:**
*   **What it proves:** **Absolutely nothing.** The proof ends in `Admitted.`, which in Coq means "skip the proof and assume this is true."
*   **Versus claims:** The comment boasts about using the Verified Software Toolchain (VST) to map to a C memory model and prove the absence of Undefined Behavior. Because the proof is admitted and there is no VST AST/separation logic here, this claim is entirely fictional in the context of the provided snippet.

**TLA+:**
*   **What it proves:** It models a strictly generic, basic memory allocator (malloc/free). Requests take blocks sequentially and hold them until they finish.
*   **Versus claims:** The module claims to model "PagedAttention" and "TurboQuant KV block compression." However, the state machine has zero concepts of KV caching, shared prefixes, block eviction policies, or compression. It is a textbook ticket-allocator. 

---

### 2. Are these proofs actually interesting or useful, or are they trivial?

**Overall:** They are **trivial**.

*   **Lean:** Proving properties over abstract mathematical Reals (`ℝ`) is the easiest part of verifying numerical software. The actual hard part—which makes formal methods interesting for AI—is proving **floating-point numerical stability** (e.g., handling IEEE 754 `Float` overflow/underflow bounds). Shift invariance is only useful if you use it to prove that a specific `Float` implementation of FlashAttention avoids `NaN`s, which is not done here.
*   **Coq:** It is a pure specification. Writing a specification is a good first step, but without a proof, it provides zero utility.
*   **TLA+:** This is a "Hello World" level TLA+ specification. It lacks the complexity that makes PagedAttention difficult (e.g., copy-on-write, out-of-memory handling, fragmented generation). 

---

### 3. Edge Cases, Logical Soundness, and Idioms

#### LEAN 4 Bugs & Idioms
*   **The `Real.log` Trap:** Lean 4 mathematically defines `Real.log x = 0` for $x \le 0$ to make the function total. The comment says "callers track positivity where needed," but the theorems `eml_exp` and `eml_e` **do not require positivity proofs** as premises. If a developer maps this to standard C++ `<cmath>` `log()`, negative inputs will yield `-NaN` (or trap). A sound specification must require $y > 0$ in the type signature or as an explicit hypothesis for the theorems to be valid representations of software behavior.
*   **Idiomatic tactics:** In `att_softmax_invariance`, you have:
    ```lean
    have hpos : 0 < ∑ j, Real.exp (v j) := Finset.sum_pos (fun j _ => Real.exp_pos (v j)) Finset.univ_nonempty
    ```
    This is overly manual. In modern Mathlib, you should just use the `positivity` tactic, which can automatically solve goals stating that sums of exponentials are positive.

#### COQ Bugs & Idioms
*   **The List Length Edge Case:** In Coq, `combine q k` (which zips lists) stops at the length of the shorter list. Therefore, `dot_product` silently ignores trailing elements if lengths differ. 
*   **Theorem Soundness:** `Theorem scaled_dot_linear` is **unprovable** without the premise `length q = length k`. If `q` is longer than `k`, `map` scales the dropped elements, changing the structural equivalence.
*   **Idioms:** Using raw `list R` and `fold_right` is a pedagogical toy. A serious formal methods researcher verifying AI code would use matrices from `MathComp`, `CoqEAL`, or formal arrays, not standard library linked lists.

#### TLA+ Errors & Idioms
*   **Syntax/Logic Error in `ConservationOfBlocks`:**
    ```tla
    LET SumAlloc[r \in 0..NumRequests] == 
          IF r = 0 THEN 0 ELSE Len(allocations[r]) + SumAlloc[r-1]
    ```
    This is **invalid TLA+**. You cannot recursively define functions inline using a standard `LET` unless you use the `RECURSIVE` keyword at the module scope, or formulate it properly using sequence folds. TLC will fail to parse this.
*   **Idiomatic alternative:** Instead of sequences, if we only care about tracking total allocations, we should flatten the sequences. A better TLA+ way to count is:
    ```tla
    TotalAllocated == 
      LET blocks(r) == { allocations[r][i] : i \in 1..Len(allocations[r]) }
      IN Cardinality(UNION { blocks(r) : r \in 1..NumRequests })
    ConservationOfBlocks == Cardinality(free_blocks) + TotalAllocated = MaxBlocks
    ```
*   **Missing Liveness Properties:** The system lacks fairness (`WF_vars(Next)`). There is no proof that an allocation eventually unblocks or that a request eventually finishes, meaning the system mathematically is allowed to deadlock forever.

### Summary
If this were submitted to a formal methods conference (like CAV, POPL, or CPP) or presented in an academic lab, it would be rejected as incomplete. To make this a serious verification effort, the authors must:
1. Move the Lean proofs from `ℝ` to IEEE 754 floats to prove real-world numerical stability.
2. Provide the actual VST Separation Logic bindings for the Coq definitions, or remove the claims.
3. Fix the syntax errors in the TLA+ spec and model the *actual memory layout/eviction logic* of PagedAttention, rather than a generic ticket stub.