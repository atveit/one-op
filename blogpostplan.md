# Blog Post Series Plan: Exp minus Log is all you need

## Post 1: Exp minus Log is all you need for Deep learning
**Subtitle:** Shown for GPT-2, Gemma 4, Nvidia Nemotron 3 Super (Multi-Token Prediction) and Qwen 3.6 27B.

**Target Audience:** ML practitioners, researchers, and engineers.

### Content Structure:
1. **Introduction & The Breakthrough:**
   - Introduce Andrzej Odrzywołek's 2026 discovery: $\eml(x,y) = \exp(x) - \ln(y)$ is a continuous Sheffer primitive.
   - The provocative claim: All of deep learning (activations, attention, optimizers) reduces to this single operator.
2. **Basic Math & Python Mapping:**
   - Define `eml(x, y)` in Python.
   - Show how basic operations emerge:
     - `exp(x) = eml(x, 1)`
     - `ln(z) = eml(1, eml(eml(1, z), 1))`
     - `x * y = exp(ln x + ln y)`
3. **Bridging to Deep Learning (GPT-2 Focus):**
   - How this applies to Neural Networks.
   - Example: A standard GPT-2 layer (Softmax Attention) reduced to EML, addressing multiplicative fragility by moving to the Min-Plus Log-domain.
4. **Lean 4 Proofs (The "Zero-Sorry" Guarantee):**
   - **Warmup Snippet:** A 3-line Lean 4 proof for `eml_exp` to introduce the syntax.
   - **GPT-2 Layer Proof:** A simple Lean 4 proof (`log_domain_attention_eq_attention`) for GPT-2 attention. Presented in a 2-column table: `Code | Line-by-line Explanation`.
5. **Expandable Deep Dives (Markdown `<details>`):**
   - SymPy & TLA+ Verification details (to keep the main post short).
   - Empirical scaling results showcasing compatibility with Gemma 4, Nemotron 3 Super, and Qwen 3.6 27B.

## Future Follow-Up Posts (Mined from EMLnotes.md)
To be executed sequentially after Post 1:

* **Post 2: Tropical State Space Models & Infinite Context**
  - Replacing the expanding KV cache with a fixed-size logarithmic state vector.
  - Bypassing $\mathcal{O}(N^2)$ attention bottlenecks for needle-in-a-haystack recall.
* **Post 3: Introspective Diffusion & Bayesian Geometry**
  - Native Metropolis-Hastings in Min-Plus space.
  - Extracting "Frozen Bayesian Maps" directly from the EML tree structure for mechanistic interpretability.
* **Post 4: The Hardware Horizon: Neuromorphic EML Processors**
  - Why digital GPUs (floating point ALUs) are overkill for EML.
  - Building LNS chips based on raw analog PN-junction physics for zero-power inference.
* **Post 5: Google's TurboQuant vs. EML Dual Spaces**
  - Applying the EML framework and dual-space proofs to Google's highly efficient TurboQuant architecture.
  - Showcasing Lean 4 proofs for the quantization steps.
  - Empirical verification: Does EML maintain the quantization bounds natively without degradation?
