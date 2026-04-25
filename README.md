# one-op: Exp minus Log is all you need

Exp minus Log is all you need. Reducing Deep Learning to a single continuous Sheffer primitive.

## 📄 Read the Blog Post
The full story, including detailed technical breakdowns and the Lean 4 walkthrough, is available on **amund.blog**:
👉 **[Exp minus Log is all you need for Deep Learning?](https://amund.blog/exp-minus-log-is-all-you-need)**

---

## 🏗️ The One-Op Evidence Stack

This repository contains the full "Zero-Sorry" formalization and empirical evidence for the EML framework.

| Layer | Component | Verification Tool | Resource Path |
| :--- | :--- | :--- | :--- |
| **Architecture** | Full picoGPT (GPT-2) | 🧮 Lean 4 | `lean/EmlNN/PicoGPT.lean` |
| **Evidence** | EML-native Grokking | 🚀 Apple MLX | `eml-mlx-grokking/main_eml.py` |
| **Stability** | LayerNorm (Newton-Schulz) | 🧮 Lean 4 | `lean/EmlNN/NormNewtonSchulz.lean` |
| **Numerics** | FP32 Error Bounds | 🛡️ Gappa | `proofs/gappa/` |
| **Concurrency** | KV-Cache Safety | ⏱️ TLA+ | `proofs/tla+/PagedAttention.tla` |

---

## 🚀 How to Rerun the Evidence

### 1. EML-native Grokking (Modular Addition)
Achieve 100% validation accuracy in under 60 seconds using the Sheffer primitive.
```bash
cd eml-mlx-grokking
python3 main_eml.py --epochs 150 --p 97 --train-fraction 0.5
```

### 2. Out-of-the-Box GPT-2 Inference
Run real inference using official GPT-2 weights via the EML-native architecture.
```bash
python3 picoGPT_eml.py "Exp minus log is"
```

---
*This repository contains the full source for the "one-op" series. Follow-up posts on Tropical SSMs, Neuromorphic EML hardware, and TurboQuant are included as draft plans.*
