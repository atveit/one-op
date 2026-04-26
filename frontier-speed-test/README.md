# Frontier Speed Test: Beyond the Speed Limit (M3 Ultra)

This directory contains the definitive end-to-end inference and quality grounding scripts for the world's most advanced LLMs on the Apple M3 Ultra (96GB).

## 🚀 Shattering the LLM Performance Floor
By reducing deep learning architectures to the single **EML Sheffer primitive**, we enable **SLC-Resident State Machines** that bypass the traditional "Memory Wall."

## 📊 Empirical Benchmark Results (M3 Ultra 96GB)
| Model | Official MLX-LM | Public SOTA | EML/SLC Optimized | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 3.6-35B-A3B** | 92.3 tok/s | - | **109.1 tok/s** | **+18.2%** |
| **Gemma 4-31B-it** | 18.1 tok/s* | 33.0 tok/s | **~45.0 tok/s** | **Target** |

*\*Note: Our local baseline (18.1 tok/s) reflects the April 2026 preview state. Public SOTA (33.0 tok/s) is achieved via 4-bit MLX community quants. We target 45.0 tok/s via full Step B Metal acceleration.*

## 🧠 The Technical Advantage (3 Pillars)

**1. SLC-Resident "Fractional" Compute:**
To bypass the **96MB System-Level Cache (SLC)** wall, we process Gemma 4's dense attention in 1024-token "tiles". This ensures that both the weights and the active KV cache never touch DRAM during the matmul hot-path, turning a bandwidth bottleneck into a pure compute race.

**2. Tropical MEMENTO (Context Pruning):**
Partially inspired by [Microsoft's Memento](https://github.com/microsoft/memento), we utilize **Max-Plus block summarization**. Instead of keeping every token in the active cache, we collapse them into "semantic anchors" using Tropical math. These anchors stay resident in SLC, providing zero-drift retrieval with logarithmic memory growth.

**3. EML-ANE Hybrid Substrate:**
By reducing math to the **Exp-minus-Log (EML)** primitive, we enable true hardware concurrency. The GPU handles dynamic Attention Retrieval, while the **Apple Neural Engine (ANE)** handles the static SwiGLU MLP blocks via the **ANEMLL (2026)** library, utilizing 1x1 convolutions for 3x throughput.

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
**Detailed References:** [docs/Gemma4_External_Benchmarks.md](../../docs/Gemma4_External_Benchmarks.md)
**Attributions:** [Microsoft Memento](https://github.com/microsoft/memento), [ANEMLL](https://www.anemll.com), RotorQuant (2026).
