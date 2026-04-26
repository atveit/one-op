# one-op: Exp minus Log is all you need for Deep Learning?

Exp minus Log is all you need. Reducing Deep Learning to a single continuous Sheffer primitive.

## 📄 Read the Full Story
The complete technical journey, including formal proofs and the analog hardware roadmap, is available on **amund.blog**:
👉 **[Exp minus Log is all you need for Deep Learning?](https://amund.blog/exp-minus-log-is-all-you-need)**

---

## 🏗️ The One-Op Evidence Stack

Based on the 2026 breakthrough by **Dr. Andrzej Odrzywołek**, we demonstrate that the single binary operator **`eml(x, y) = exp(x) - ln(y)`** is a continuous Sheffer primitive—the "NAND gate" of continuous mathematics.

This repository provides the "Zero-Sorry" formalization and empirical evidence for the EML framework.

| Layer | Component | Verification Tool | Resource Path |
| :--- | :--- | :--- | :--- |
| **Foundations** | Functional Universality | 🧮 Lean 4 | [**`lean/EmlNN/Basic.lean`**](./lean/EmlNN/Basic.lean) |
| **Architecture** | Full picoGPT (GPT-2) | 🧮 Lean 4 | [**`lean/EmlNN/PicoGPT.lean`**](./lean/EmlNN/PicoGPT.lean) |
| **World Models** | JEPA / VICReg Stability | 🚀 Apple MLX | [**`scripts/jepa/`**](./scripts/jepa/) |
| **Empiri** | Grokking Phase Transitions | 🚀 Apple MLX | [**`eml-mlx-grokking/`**](./eml-mlx-grokking/) |
| **Stability** | LayerNorm (Newton-Schulz) | 🧮 Lean 4 | [**`lean/EmlNN/NormNewtonSchulz.lean`**](./lean/EmlNN/NormNewtonSchulz.lean) |
| **Numerics** | FP32 Error Bounds | 🛡️ Gappa | [**`proofs/gappa/`**](./proofs/gappa/) |
| **Concurrency** | KV-Cache Safety | ⏱️ TLA+ | [**`proofs/tla+/`**](./proofs/tla+/) |

---

## 🚀 Key Results & How to Rerun

### 1. EML-native Grokking (Modular Addition)
Achieve 100% validation accuracy on Apple Silicon. The EML-native model achieves perfect functional parity but exhibits a "Numerical Friction" delay compared to standard baselines.
```bash
cd eml-mlx-grokking
# 1,000 epochs to observe the full phase transition
python3 main_eml.py --epochs 1000 --p 97 --train-fraction 0.5
```

### 2. Out-of-the-Box GPT-2 Inference
Run real-time inference using official OpenAI weights (124M) via the EML-native circuit.
```bash
cd eml-picogpt
python3 main_inference.py "The EML operator is"
```

### 3. JEPA World Models
Verify stability against representation collapse in predictive architectures.
```bash
cd scripts/jepa
python3 jepa_1d_kinematics.py # Induces collapse in baseline vs EML stability
```

---

## ⚡ The Analog Horizon
Beyond auditability, the EML operator is the native language of **PN-junction physics**. A standard MOSFET in sub-threshold operation computes the exponential; a diode computes the logarithm. 

This framework provides a blueprint for **neuromorphic LNS (Logarithmic Number System) hardware** that aligns AI with the native physics of its substrate, potentially achieving 1000x better energy efficiency than digital GPUs.

---
*This repository contains the full source for the "one-op" series. Follow-up posts on Tropical SSMs and Neuromorphic EML hardware are included as draft plans in the root.*
