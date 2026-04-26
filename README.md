# one-op: Continuous Sheffer Primitives for Deep Learning

This repository contains the machine-checked proofs and empirical evidence for the **Exp-minus-Log (EML)** unified substrate.

## 🚀 Frontier Speed Test (M3 Ultra)
We have established empirical speed gains on the Apple M3 Ultra by improving SLC residency and amortizing framework overhead using EML and Tropical math.

📂 **[frontier-speed-test/](./frontier-speed-test/)**: Direct CLI tools to compare standard MLX against EML-native SLC-optimized inference for **Qwen 3.6-35B** and **Gemma 4-31B**.

## 🧱 Evidence Stack
- `eml-picogpt/`: Bit-for-bit parity proof for GPT-2 (124M) using EML circuits.
- `eml-mlx-grokking/`: 100% accuracy generalization on modular addition tasks.
- `scripts/jepa/`: Preventing representation collapse in world models via stable energy losses.
- `lean/`: Formal algebraic proofs in Lean 4 (Identity, Universality, Tropical Convergence).
- `proofs/`: Numerical stability proofs (Gappa) and concurrency models (TLA+).

## 📐 Zero-Sorry Verification
- **Functional Identity:** EML-native layers are mathematically identical to standard ones.
- **Hardware Native:** Aligns with PN-junction physics for next-gen analog LNS hardware.

---
**Read the research series:** [amund.blog](https://amund.blog/finding-gemma4-performance-floor)

## 🔮 Future Work
We are currently exploring further optimizations without requiring retraining:
- **dflash:** Low-latency speculative decoding.
- **ddtree:** Distributed decision trees for sparse expert routing.
- **Tri-Attention:** Spatial Trigonometric Pruning for extreme SLC residency.
- **DeepSeek V4 Integration:** Leveraging lightning indexers for 1M+ token contexts.
