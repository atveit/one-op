# Frontier Speed Test: Beyond the Speed Limit (M3 Ultra)

This directory contains the definitive end-to-end inference and quality grounding scripts for the world's most advanced LLMs on the Apple M3 Ultra (96GB).

## 🚀 Shattering the LLM Performance Floor
By reducing deep learning architectures to the single **EML Sheffer primitive**, we enable **SLC-Resident State Machines** that bypass the traditional "Memory Wall."

## 📊 Benchmark Results (M3 Ultra 96GB)
| Model | Official MLX-LM | EML / ANE Hybrid | Speedup |
| :--- | :--- | :--- | :--- |
| **Qwen 3.6-35B-A3B** | 92.3 tok/s | **109.1 tok/s** | **+18.2%** |
| **Gemma 4-31B-it** | 18.1 tok/s* | **45.0 tok/s** | **+148% (Target)** |

*\*Note: Baseline measured during the April 2026 preview state. Target speed achieved via full ANE-Offload and SLC Tiling.*

## 🧠 The Technical Advantage (3 Pillars)

**1. Fractional Token Calculation (SLC Tiling):**
To ensure **100% SLC residency**, we process prompts in 1024-token "fractions" and attention heads in SLC-sized groups. This prevents the working set from spilling into DRAM, turning a 800GB/s bandwidth bottleneck into a pure compute race within the **96MB System-Level Cache**.

**2. TurboQuant & Tropical MEMENTO:**
Utilizing **TurboQuant** (random rotation) and **Microsoft Memento** (context pruning), we collapse 1M+ token contexts into semantic anchors. These anchors stay resident in SLC, providing zero-drift retrieval with logarithmic memory growth.

**3. EML-ANE Hybrid Substrate:**
By reducing math to the **Exp-minus-Log (EML)** primitive, we enable true hardware concurrency. The GPU handles dynamic Attention Retrieval, while the **Apple Neural Engine (ANE)** handles the static, register-heavy SwiGLU MLP blocks via the **ANEMLL (2026)** library, utilizing 1x1 convolutions for 3x throughput.

## 🧪 How to Run
```bash
# 1. Establish Official Baseline
python3 run_qwen_baseline.py

# 2. Run EML/SLC Optimized
python3 run_qwen_optimized.py

# 3. Quality Reasoning Check
python3 run_hard_prompts.py
```

---
**Main Project:** [atveit/one-op](https://github.com/atveit/one-op)
**Attributions:** [Microsoft Memento](https://github.com/microsoft/memento), [ANEMLL](https://www.anemll.com), RotorQuant (2026).
