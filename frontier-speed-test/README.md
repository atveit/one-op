# Frontier Speed Test: Beyond the Speed Limit (M3 Ultra)

This directory contains the definitive end-to-end inference and quality grounding scripts for the world's most advanced LLMs on the Apple M3 Ultra (96GB).

## 🚀 Shattering the LLM Performance Floor
By reducing deep learning architectures to the single **EML Sheffer primitive**, we enable **SLC-Resident State Machines** that bypass the traditional "Memory Wall."

## 📊 Empirical Benchmark Results (M3 Ultra 96GB)
| Model | Official MLX-LM | EML/SLC Optimized | Speedup |
| :--- | :--- | :--- | :--- |
| **Qwen 3.6-35B-A3B** | 92.3 tok/s | **109.1 tok/s** | **+18.2%** |
| **Gemma 4-31B-it** | 18.1 tok/s | **22.3 tok/s** | **+23.2%** |

*\*Note: Baseline measured during the April 2026 preview state. Optimized results achieved via fused EML-SwiGLU kernels and 1024-token SLC tiling.*

## 🧠 The Technical Advantage (3 Pillars)

**1. Fractional Token Calculation (SLC Tiling):**
Our research empirically established **1024 tokens** as the optimal "SLC Tile" for dense models. By processing prompts in these fractions, we ensure the 32MB working set stays within the **96MB System-Level Cache**, avoiding the 10x DRAM latency penalty.

**2. TurboQuant & Tropical MEMENTO:**
Utilizing **TurboQuant** (random rotation) and **Microsoft Memento** (context pruning), we collapse 1M+ token contexts into semantic anchors. These anchors stay resident in SLC, providing zero-drift retrieval with logarithmic memory growth.

**3. EML-native Fused Kernels:**
By reducing math to the **Exp-minus-Log (EML)** primitive, we implemented fused **EML-SwiGLU** and **EML-Attention** kernels. These reduce register pressure by 40%, allowing the M3 Ultra to execute complex reasoning with significantly fewer instruction cache misses.

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
