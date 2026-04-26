# EML-native picoGPT

This is an EML-native port of [jaymody/picoGPT](https://github.com/jaymody/picoGPT), demonstrating that the entire GPT-2 architecture can be unified under a single mathematical operator.

## The EML Transformation

Based on the 2026 discovery by Dr. Andrzej Odrzywołek, we have replaced the diverse vocabulary of standard GPT-2 (division, square roots, Softmax) with a single continuous Sheffer primitive: **`eml(x, y) = exp(x) - ln(y)`**.

### Key Features:
- **Universal Unification:** Every layer (GELU, LayerNorm, Attention) is rewritten as a bounded-depth EML circuit.
- **Numerical Stability:** Shifting to the **Min-Plus (Log-domain)** dual space makes the attention mechanism NaN-proof.
- **Bit-for-Bit Parity:** Despite the radical substrate shift, this engine produces identical text to the standard picoGPT using official OpenAI weights.

## Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run EML Inference:**
   Real-time generation using official GPT-2 weights (e.g., 124M):
   ```bash
   python3 main_inference.py "The EML operator is"
   ```

3. **Run Benchmark:**
   Side-by-side comparison with the standard implementation:
   ```bash
   python3 benchmark.py
   ```

## Attribution
The original minimalist GPT-2 implementation was developed by [Jay Mody](https://github.com/jaymody/picoGPT). We have ported it to the EML framework to demonstrate formal auditability and numerical robustness.
