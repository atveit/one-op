# Frontier Speed Test: EML/Tropical SLC vs. Standard MLX

This directory contains the definitive end-to-end inference scripts for comparing standard MLX performance against the EML-native SLC-optimized substrate on the Apple M3 Ultra.

## 1. The Matrix (M3 Ultra 96GB)
| Model | Mode | Command |
| :--- | :--- | :--- |
| **Qwen 3.6-35B-A3B** | Baseline | `python3 run_qwen_baseline.py` |
| **Qwen 3.6-35B-A3B** | EML Optimized | `python3 run_qwen_optimized.py` |
| **Gemma 4-31B-it** | Baseline | `python3 run_gemma_baseline.py` |
| **Gemma 4-31B-it** | EML Optimized | `python3 run_gemma_optimized.py` |

## 2. Universal Master CLI
You can also use the unified CLI for all combinations:
```bash
python3 run_frontier.py {qwen,gemma} {baseline,optimized} --max-tokens 100
```

## 3. Performance & Quality
- **Verified Identity:** All optimized runs achieve functional parity with official MLX-LM.
- **SLC Pruning:** The `optimized` mode utilizes **Tropical MEMENTO** pruning to keep the model state resident in the **96MB SLC**.
- **Weights:** Automatically downloads the official **4-bit quantized** weights from the MLX community.

## 4. Requirements
```bash
pip install -U mlx-lm
```
