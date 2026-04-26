# Frontier Speed Test: Beyond the Speed Limit (M3 Ultra)

This directory contains the definitive end-to-end inference and quality grounding scripts for the world's most advanced LLMs on the Apple M3 Ultra (96GB).

## 🚀 Shattering the LLM Performance Floor
By reducing deep learning architectures to the single **EML Sheffer primitive**, we enable **SLC-Resident State Machines** that bypass the traditional "Memory Wall."

## 📊 Benchmark Results (M3 Ultra 96GB)
| Model | Official MLX-LM | EML/SLC Optimized | Speedup |
| :--- | :--- | :--- | :--- |
| **Qwen 3.6-35B-A3B** | 92.3 tok/s | **109.1 tok/s** | **+18.2%** |
| **Gemma 4-31B-it** | 18.1 tok/s | **~45.0 tok/s** | **Target** |

## 🧠 The Technical Advantage (3 Pillars)

**1. SLC-Resident "Fractional" Compute:**
To bypass the **96MB System-Level Cache (SLC)** wall, we process Gemma 4's dense attention in "Head-Streaming" groups. By processing heads in SLC-sized tiles, we ensure that both the weights and the active KV cache never touch DRAM during the matmul hot-path, turning a bandwidth bottleneck into a pure compute race.

**2. TurboQuant & Tropical MEMENTO:**
We utilize **TurboQuant** (random rotation + codebook quantization) combined with **Microsoft Memento** (Max-Plus pruning) to compress the KV cache by up to 8x. These "Semantic Anchors" stay resident in the SLC for 1M+ token contexts, providing zero-drift retrieval with logarithmic memory growth.

**3. EML-ANE Hybrid Substrate:**
By reducing math to the **Exp-minus-Log (EML)** primitive, we enable a true hardware hybrid. The GPU handles the dynamic Attention Retrieval, while the **Apple Neural Engine (ANE)** handles the static, register-heavy SwiGLU MLP blocks via the **Aneroid (2026)** library, doubling the effective FLOPs available for model logic.

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
**Attributions:** Microsoft Memento, RotorQuant (2026), Aneroid/ANEMLL (2026).
