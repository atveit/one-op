# Empirical Scripts & World Models

This directory contains the Python-based empirical evidence for the EML framework, utilizing the **Apple MLX** framework for high-performance execution on Apple Silicon.

## Key Experiments

### 1. `jepa/` (World Models)
Demonstrates EML stability in Yann LeCun's **Joint-Embedding Predictive Architecture (JEPA)**.
- **`jepa_1d_kinematics.py`**: Proves that the verified VICReg loss prevents representation collapse under precision starvation.
- **`jepa_trajectory_drift.py`**: Demonstrates how the Min-Plus dual space prevents drift during long-horizon unrolled predictions.

### 2. `sympy/` (Symbolic Calculus)
Mechanically verifies the gradient chain (derivative) for EML circuits, ensuring that the backpropagation logic matches the forward Sheffer reduction.

## Attribution
The JEPA experiments are inspired by Yann LeCun's Vision-JEPA and V-JEPA (2024-2026).
