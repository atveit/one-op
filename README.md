# one-op: Exp minus Log is all you need

Exp minus Log is all you need. Reducing Deep Learning to a single continuous Sheffer primitive.

## 📄 Read the Blog Post
The full story, including detailed technical breakdowns and the Lean 4 walkthrough, is available on **amund.blog**:
👉 **[Exp minus Log is all you need for Deep Learning?](https://amund.blog/exp-minus-log-is-all-you-need)**

---

## TL;DR: The EML Framework

Based on the March 2026 breakthrough by Andrzej Odrzywołek, we demonstrate that the single binary operator **`eml(x, y) = exp(x) - ln(y)`** (plus the constant `1`) is a continuous Sheffer primitive—functionally complete for the entire mathematical vocabulary of deep learning.

### Key Highlights:
- **Unification:** Every layer (Softmax, LayerNorm, GELU, etc.) is rewritten as a bounded-depth tree of `eml`.
- **Numerical Stability:** We solve "multiplicative fragility" by shifting fragile operations like Softmax into the **Min-Plus (Log-domain)** space, replacing division with stable subtraction.
- **Formal Verification:** The entire stack is verified with "Zero-Sorry" correctness in **Lean 4**, while **Gappa** bounds FP32 errors and **TLA+** ensures operational safety.
- **Real-World Proof:** Applied to **picoGPT** and scaled to 2026 frontier models (Nemotron 3 Super, Qwen 3.6), maintaining perfect loss parity while eliminating NaN crashes.

---

## 🛠️ Resources & Code (Post 1)

| Resource | Description |
| :--- | :--- |
| 📄 **[post1.md](post1.md)** | Local markdown copy of the technical tour. |
| 🧮 **[lean/EmlNN/Basic.lean](lean/EmlNN/Basic.lean)** | Foundational Lean 4 axioms for the `eml` operator. |
| 📐 **[lean/EmlNN/Attention.lean](lean/EmlNN/Attention.lean)** | Lean 4 proof of mathematical equivalence for Log-domain attention. |
| 🛡️ **[proofs/smt/mlp_robustness.py](proofs/smt/mlp_robustness.py)** | Z3 SMT solver script proving local adversarial robustness. |
| 🤖 **[picoGPT/](picoGPT/)** | Minimalist GPT-2 reference implementation used for the EML rewrite. |
| 💬 **[llm_reviews/](llm_reviews/)** | Hard-hitting peer reviews from GPT-5.4, Claude Opus 4.6, and Gemini 3.1. |

---
*This repository contains the first installment of the "one-op" series. Speculative future posts on Tropical SSMs and Neuromorphic EML hardware are coming soon.*
