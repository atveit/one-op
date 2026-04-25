# one-op: Exp minus Log is all you need

Exp minus Log is all you need. Reducing Deep Learning to a single continuous Sheffer primitive.

## 📄 Read the Blog Post
The full story, including detailed technical breakdowns and the Lean 4 walkthrough, is available on **amund.blog**:
👉 **[Exp minus Log is all you need for Deep Learning?](https://amund.blog/exp-minus-log-is-all-you-need)**

---

## 🛠️ Complete Verification & Evidence Stack

This repository contains the full "Zero-Sorry" formalization and empirical evidence for the EML framework.

| Layer | Component | Verification Tool | Resource Path |
| :--- | :--- | :--- | :--- |
| **Foundations** | EML Axioms & Arith | 🧮 Lean 4 | `lean/EmlNN/Basic.lean`, `Arith.lean` |
| **Architecture** | Full picoGPT (GPT-2) | 🧮 Lean 4 | `lean/EmlNN/PicoGPT.lean` |
| **Activations** | SwiGLU, GELU, SiLU | 🧮 Lean 4 | `lean/EmlNN/Activations.lean` |
| **Stability** | LayerNorm (Newton-Schulz) | 🧮 Lean 4 | `lean/EmlNN/NormNewtonSchulz.lean` |
| **Numerics** | FP32 Error Bounds | 🛡️ Gappa | `proofs/gappa/` |
| **Robustness** | Adversarial SMT Proof | 🛡️ Z3 | `proofs/smt/mlp_robustness.py` |
| **Concurrency** | KV-Cache Deadlock Safety | ⏱️ TLA+ | `proofs/tla+/PagedAttention.tla` |
| **Algorithmic** | BPE Tokenizer Safety | 🤖 KeY / JML | `proofs/key/Tokenizer.java` |
| **Systems** | Distributed Training | 🌀 ABS | `proofs/abs/Cluster.abs` |
| **Compilers** | QKV Kernel (VST) | 🎖️ Coq | `proofs/coq/QKV.v`, `QJL.v` |

---

## 🚀 Experiments & World Models

We provide native **Apple MLX** implementations for our empirical demonstrations:
- **`picoGPT_eml.py`**: The full GPT-2 architecture rewritten as a single-operator circuit.

- **`eml-mlx-grokking/`**: An EML-native port of [stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking), demonstrating functional parity and 100% accuracy on modular addition tasks.
- **`scripts/jepa/`**: Toy World Models (V-JEPA & I-JEPA) proving EML stability during representation unrolling.

---
*This repository contains the full source for the "one-op" series. Follow-up posts on Tropical SSMs, Neuromorphic EML hardware, and TurboQuant are included as draft plans.*
