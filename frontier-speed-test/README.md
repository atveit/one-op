# Frontier Speed Test: EML/Tropical SLC vs. Standard MLX

This directory contains the definitive end-to-end inference and quality grounding scripts for the Apple M3 Ultra.

## 1. The Matrix (M3 Ultra 96GB)
| Model | Mode | Command |
| :--- | :--- | :--- |
| **Qwen 3.6-35B-A3B** | Baseline | `python3 run_qwen_baseline.py` |
| **Qwen 3.6-35B-A3B** | EML Optimized | `python3 run_qwen_optimized.py` |
| **Qwen 3.6-35B-A3B** | Quality Check | `python3 run_qwen_quality.py` |
| **Gemma 4-31B-it** | Baseline | `python3 run_gemma_baseline.py` |
| **Gemma 4-31B-it** | EML Optimized | `python3 run_gemma_optimized.py` |
| **Gemma 4-31B-it** | Quality Check | `python3 run_gemma_quality.py` |

## 2. Universal Master CLI
Run any combination via the unified interface:
```bash
python3 run_frontier.py {qwen,gemma} {baseline,optimized,quality}
```

## 3. Guaranteed Quality: Zero-Sorry Goals
All `optimized` runs are verified using our **Buildup Strategy**:
- **Identity (Step A):** Functional bit-for-bit parity with official MLX-LM.
- **Reasoning (Step B):** High-quality semantic grounding preserved despite **Tropical MEMENTO** context pruning.

## 4. Requirements
```bash
pip install -U mlx-lm
```
