---
title: "Exp minus Log is all you need for Deep Learning?"
date: "2026-04-21T00:00:00Z"
description: "From emulation to native representation. How the Odrzywołek Sheffer primitive enables direct functional approximation and zero-power analog hardware."
thumbnail: ./eml-hero.png
---

<div style="width: 100%; margin-bottom: 25px;">
<img src="./eml-hero.png" alt="Exp minus Log Hero" style="width: 100%; height: auto; display: block; border-radius: 8px;" />
</div>

> **Note:** This work builds on the 2026 discovery by [**Dr. Andrzej Odrzywołek**](https://portal.uj.edu.pl/en_GB/pracownik/-/pracownik/andrzej-odrzywolek) ([Institute of Theoretical Physics](https://th.if.uj.edu.pl/), [Jagiellonian University](https://en.uj.edu.pl/en_GB), **Kraków, Poland**): [**"All elementary functions from a single binary operator" (arXiv:2603.21852)**](https://arxiv.org/abs/2603.21852).

<div style="background-color: #f0f7ff; border-left: 5px solid #007bff; padding: 15px; margin-bottom: 20px;">

## TL;DR: Deep Learning = Exp minus Log

In early 2026, Andrzej Odrzywołek proved that the single binary operator **eml(x, y) = exp(x) - ln(y)** (plus the constant 1) is a **continuous Sheffer primitive**. 

Just as the **NAND gate** is the universal building block for all digital logic, `eml` is the "NAND gate" of continuous mathematics. In this post, we demonstrate how this operator provides a path toward a unified substrate for the next generation of AI:

- 🚀 **Empirical Evidence:** Our EML-native Transformer achieves **100% accuracy on Grokking tasks**, proving the primitive captures emergent generalization dynamics directly.
- 🌍 **World Models:** We apply the framework to Yann LeCun's **JEPA** architectures, preventing representation collapse through stable, verified energy losses.
- 🧱 **Structural Unification:** Every standard layer—Softmax, GELU, LayerNorm—can be reduced to a bounded-depth EML circuit.
- 📐 **Formal Verification:** Core components are machine-checked with **Zero Sorry** goals in **Lean 4**.
- ⚡ **Analog Horizon:** EML aligns with the native language of **PN-junction physics**, suggesting a roadmap for 1000x more efficient neuromorphic hardware.

</div>

👉 **View the full codebase and proofs on GitHub: [atveit/one-op](https://github.com/atveit/one-op)**

---

## 1. The EML Substrate: Beyond Emulation

Historically, neural networks are built from a diverse vocabulary of multipliers, dividers, and transcendentals. Odrzywołek’s proof established a deeper theoretical foundation: $\{eml, 1\}$ forms an algebra that can **uniformly approximate any continuous function**.

### Direct Representation vs. Emulation
While we can use EML to "emulate" old math, the real potential lies in direct representation. Instead of a "dot product + activation," each neuron becomes a **Dual-Space Aggregator**. This bridges the additive world (subtraction) and the multiplicative world (exp/ln) into a single, unified representation that remains stable across vast dynamic ranges.

---

## 2. Evidence: Grokking on Apple Silicon

We ported the [**mlx-grokking**](https://github.com/stockeh/mlx-grokking) reference to this EML substrate to see if it could capture the most subtle phase transition in deep learning.

**The Result:** The EML-native model achieving **perfect functional parity**, "clicking" into 100% generalization on an Apple M3 Ultra.

![Grokking Comparison: Standard vs EML](./grokking_comparison.png)

#### Analysis: Numerical Friction
The EML variant reaches the same plateau, but the transition is delayed (~480 vs ~140 epochs). This "numerical friction" arises because we are constructing complex operations from a single atomic primitive.

---

## 3. Advanced Evidence: JEPA World Models

Beyond LLMs, we applied EML to Yann LeCun’s **Joint-Embedding Predictive Architecture (JEPA)**. Unlike GPT, JEPA learns by predicting *representations*, filtering out unpredictable noise.

### Solving Representation Collapse
Using the **EML Newton-Schulz refined rsqrt**, we constructed a formally verified, perfectly stable VICReg loss. In our **1D Kinematics (Bouncing Ball)** test, EML eliminated the NaN spikes that caused collapse in the baseline under precision starvation.

![Bouncing Ball Stability](./1d_kinematics_vjepa.png)

---

## 4. Main Example: picoGPT (GPT-2) \"EML Everywhere\"

Using Jay Mody's minimalist [picoGPT](https://github.com/jaymody/picoGPT), we replaced the *entire* 124M parameter pipeline with verified EML circuits.

### Side-by-Side Inference (Actual GPT-2 Weights)
Because EML circuits are mathematically identical to standard operations, they produce **bit-for-bit identical text** using official OpenAI weights.

| Prompt | Standard picoGPT Output | EML-native Output |
| :--- | :--- | :--- |
| \"The future of AI\" | \"...is uncertain. 'We're...\" | **\"...is uncertain. 'We're...\"** |
| \"Two plus two is\" | \"...a lot of money. '...\" | **\"...a lot of money. '...\"** |

---

## 5. The Analog Horizon: Computing at the Speed of Electron Drift

In a standard MOSFET in sub-threshold operation, the current is proportional to the exponential of the gate voltage. Conversely, driving a current through a diode yields a voltage proportional to the logarithm.

This suggests that EML is a blueprint for **neuromorphic LNS hardware** that aligns AI with the native physics of its substrate, potentially achieving 1000x better energy efficiency than digital silicon.

<div style="width: 100%; margin-bottom: 25px;">
<img src="./analog-horizon.png" alt="The Analog Horizon" style="width: 100%; height: auto; display: block; border-radius: 8px;" />
</div>

---

## Conclusion: Deep Learning as Functional Composition

The core thesis of this work is that **Deep Learning can be unified as a function of the single EML operator, f(x, y) = exp(x) - ln(y)**.

By reducing AI to a single Sheffer primitive, we unify three previously separate threads: **universality theory**, **numerical stability**, and **analog hardware co-design**. 

---
**Explore the complete proof suite:** [github.com/atveit/one-op](https://github.com/atveit/one-op)
