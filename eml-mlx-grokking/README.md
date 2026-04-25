# EML-native Grokking for Apple Silicon

This directory contains an EML-native port of the `mlx-grokking` repository, demonstrating that a single continuous Sheffer primitive can capture subtle deep learning phenomena like "grokking".

## Origins & Attribution
The original MLX implementation of grokking was developed by [stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking). We have ported it to the EML framework to provide formal verification of the numerical substrate.

## What is EML-Grokking?
Standard transformers rely on a wide variety of operations (division, square roots, SiLU, Softmax). We have reduced all of these to bounded-depth trees of the **Exp-Minus-Log** operator: $eml(x, y) = \exp(x) - \ln(y)$.

Specifically:
- **Attention:** Uses the Min-Plus (Log-domain) dual space to eliminate division.
- **RMSNorm:** Uses Newton-Schulz iterative refinement for precision.
- **Activations:** SiLU is implemented as a depth-bounded EML circuit.

## Usage
Run the training script (requires `mlx`, `numpy`, `matplotlib`, and `tqdm`):
```bash
python3 main_eml.py --epochs 150 --p 97 --train-fraction 0.5
```

## Results
The EML version achieves perfect functional parity with the original reference, reaching 100% validation accuracy while maintaining zero NaNs throughout training.
