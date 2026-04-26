# Frontier Speed Test: Exploring the Performance Floor (M3 Ultra)

This directory contains end-to-end inference and quality grounding scripts for advanced LLMs on the Apple M3 Ultra (96GB), focused on identifying empirical speed gains via the EML substrate.

## 🚀 Performance Observations
By mapping deep learning architectures to the **EML Sheffer primitive**, we observe significant throughput improvements by improving **SLC Residency** and amortizing framework dispatch overhead.

| Model | Phase | Official Baseline | EML-SLC Optimized | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen 3.6-35B** | **Prefill (PP)** | 1,478.5 tokens/sec | **1,906.8 tokens/sec** | **+29.0%** |
| **Qwen 3.6-35B** | **Decoding (TG)** | 89.2 tokens/sec | **157.2 tokens/sec** | **+76.2%** |
| **Gemma 4-31B** | **Prefill (PP)** | 250.8 tokens/sec | 250.4 tokens/sec | ~0.0% |
| **Gemma 4-31B** | **Decoding (TG)** | 18.1 tokens/sec | **42.4 tokens/sec** | **+134.3%** |

*\*Benchmarks performed on M3 Ultra (96GB) using 4-bit Instruct weights.*

## 🧠 Technical Analysis (3 Pillars)

**1. SLC-Resident Tiling:**
Standard Transformers often hit DRAM bandwidth limits. Our framework processes prompts in **1024-token "SLC Tiles"** to keep the 32MB working set resident in the **96MB System-Level Cache (SLC)**, aiming to bypass the DRAM latency penalty.

**2. EML-MicroBatching:**
Decoding throughput is often limited by framework dispatch overhead. We utilize **Micro-Batching** to amortize this cost. This is likely why **Gemma 4** sees a significant generation speedup despite its dense memory footprint.

**3. Tropical MEMENTO:**
Utilizing **Max-Plus block summarization** (inspired by [Microsoft's Memento](https://github.com/microsoft/memento)), we collapse context into semantic anchors to maintain SLC residency for long sequences.

## 🧪 How to Run

### Gemma 4 Reasoning Suite (10 Prompts)
Compare the official baseline against the EML-optimized variant:
```bash
./run_gemma_suite_baseline.sh
./run_gemma_suite_optimized.sh
```

### Individual Benchmarks
```bash
python3 run_frontier.py {qwen,gemma} {baseline,optimized,quality}
```

---
**Main Project:** [atveit/one-op](https://github.com/atveit/one-op)
**Attributions:** Microsoft Memento, RotorQuant (2026), ANEMLL (2026).
