# Complementary Verification Plan for GPT-2

While our core work uses Lean 4, Gappa, TLA+, and SymPy to prove the mathematical and structural completeness of the EML Sheffer primitive within neural networks, a complete "end-to-end" verification of a model like GPT-2 requires a broader ecosystem of tools.

This document outlines a plan for simple, highly targeted proofs we could execute using complementary formal methods tools to verify different abstraction layers of a GPT-2 system.

---

## 1. SMT Solvers (Z3, Reluplex, Marabou)
**Target Layer:** Local Robustness and Adversarial Bounds.
**What to Prove for GPT-2:** 
*   **Adversarial Token Perturbation:** Take a single Feed-Forward (MLP) block of a frozen, pre-trained GPT-2 model. We can use an SMT solver to formally prove that for a specific input token embedding vector $x$, any adversarial perturbation $\epsilon$ within a defined $L_\infty$ norm cannot alter the top-1 predicted output logit. 
*   *Why this matters:* This guarantees that microscopic, imperceptible changes to the embedding space cannot flip the model's deterministic prediction, offering absolute proof of local robustness against certain adversarial attacks.

## 2. Compiler Verification (Coq / CompCert)
**Target Layer:** Assembly/Machine Code Translation.
**What to Prove for GPT-2:** 
*   **QKV Matrix Multiplication Preservation:** Deep learning relies heavily on optimized C/C++ or CUDA code for matrix multiplications (e.g., generating Query, Key, and Value matrices). We can use Coq and the CompCert C compiler to formally prove that a C-reference implementation of the QKV projection strictly preserves the algebraic properties of the mathematical specification.
*   *Why this matters:* It proves the absence of Undefined Behavior (UB), buffer overflows, or silent precision-dropping bugs introduced during the compilation from high-level Python/C down to the hardware assembly instructions.

## 3. KeY / JML (Java Modeling Language)
**Target Layer:** Algorithmic Implementation and Reference Ports.
**What to Prove for GPT-2:** 
*   **BPE Tokenizer Invertibility & Safety:** Similar to how de Gouw et al. (2015) used KeY to verify Java's TimSort, we can implement the GPT-2 Byte-Pair Encoding (BPE) tokenizer in Java/Kotlin. Using JML contracts, we can formally verify that the tokenizer loop never encounters an `IndexOutOfBoundsException` and that the `decode` function is a strict, perfect inverse of the `encode` function (`decode(encode(text)) == text`).
*   *Why this matters:* Tokenization bugs are notoriously subtle and cause catastrophic downstream failures in LLMs. Verifying the reference port guarantees algorithmic purity before the text ever reaches the neural network.

## 4. ABS (Active Behavioral Specifications - Univ. of Oslo)
**Target Layer:** Resource-Aware Distributed Training and Concurrency.
**What to Prove for GPT-2:** 
*   **Distributed Gradient Synchronization (Data Parallelism):** We can write an ABS model of a distributed GPT-2 training cluster consisting of a Parameter Server and multiple Worker Nodes. We can formally verify that the asynchronous gradient accumulation and synchronization steps never result in a distributed deadlock, and that the peak memory consumption of any individual worker node stays strictly within its defined VRAM limit.
*   *Why this matters:* Modern LLM training fails frequently due to silent cluster deadlocks or Out-Of-Memory (OOM) spikes. ABS allows us to model the exact physical deployment costs and concurrent scheduling timing to mathematically rule out these operational failures before spinning up thousands of expensive GPUs.

---

## 5. EML Core Verifications (Lean 4 & Gappa)
**Target Layer:** Algorithmic Numerical Stability and Identity.
**What to Prove for GPT-2 (Inspired by the Paper):**

*   **LayerNorm Stability (Newton-Schulz):**
    *   *Proof:* In `lean/EmlNN/NormNewtonSchulz.lean`, we prove that the EML-native LayerNorm (using Newton-Schulz iterative refinement for the $1/\sqrt{x}$ term) is algebraically equivalent to the reference LayerNorm.
    *   *Why it matters:* This validates the **Additive Fragility** dual-space claim. The paper reports a **6.2x precision tightening** using this method under wide per-row variance.
*   **Log-Domain Cross-Entropy:**
    *   *Proof:* In `lean/EmlNN/Losses.lean`, we prove the equivalence of the EML-native Cross-Entropy loss (constructed entirely from LSE subtractions) to the standard generative Softmax loss.
    *   *Why it matters:* It ensures the GPT-2 training objective is preserved while operating in the NaN-proof **Min-Plus** dual space.
*   **Muon Orthogonalization Isomorphism:**
    *   *Proof:* In `lean/EmlNN/OptimizerClassification.lean`, we formally verify a single step of the Muon optimizer's `newtonschulz5` iteration.
    *   *Why it matters:* The paper claims Muon is "precisely isomorphic to our additive-fragility dual space." This proof confirms that the EML framework isn't just compatible with modern optimizers, but fundamentally describes their internal logic.
*   **Sinusoidal Positional Encoding:**
    *   *Proof:* In `lean/EmlNN/Positional.lean`, we prove the EML construction for sinusoidal positional encodings (using the EML $sin$/$cos$ circuits) matches the reference mathematical specification.
    *   *Why it matters:* It ensures that the geometric "meaning" of position in the sequence is preserved after the Sheffer reduction.