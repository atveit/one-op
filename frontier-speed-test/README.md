# Frontier Speed Test: EML/Tropical SLC vs. Standard MLX

This directory contains the definitive end-to-end inference scripts for comparing standard MLX performance against the EML-native SLC-optimized substrate on the Apple M3 Ultra.

## 1. Benchmarking Targets
- **Qwen 3.6-35B-A3B-Instruct (4-bit)**: `mlx-community/Qwen3.6-35B-A3B-4bit`
- **Gemma 4-31B-it (4-bit)**: `mlx-community/gemma-4-31b-it-4bit`

## 2. Installation
Ensure you have the latest `mlx-lm` installed:
```bash
pip install -U mlx-lm
```

## 3. Usage

### A. Run Official MLX Baseline
Establish the "Speed Floor" using the official, un-modified MLX implementation.
```bash
python3 run_baseline.py --model mlx-community/Qwen3.6-35B-A3B-4bit
```

### B. Run EML/Tropical Optimized
Run the SLC-optimized version using Tropical MEMENTO pruning and EML-native LayerNorms.
```bash
python3 run_optimized.py --model mlx-community/Qwen3.6-35B-A3B-4bit
```

## 4. Key Performance Differentiators
- **Standard MLX:** Bound by DRAM latency once the context window hits the "Memory Wall".
- **EML/Tropical SLC:** Bypasses the wall by pruning the context window down to "semantic anchors" that fit within the **96MB System-Level Cache (SLC)**.

## 5. Quality Grounding
All EML-native runs are verified for **Perfect Functional Identity** (Step A) or **Semantic Parity** (Step B). Every token produced by the EML substrate is grounded in the official weights with zero retraining.
