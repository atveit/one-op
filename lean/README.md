# Lean 4: Mathematical Identity Proofs

This directory contains the foundational formalization of the EML framework in the **Lean 4** theorem prover.

## Core Mandate
We use Lean 4 to prove that our composed EML circuits are **mathematically identical** to standard neural network layers over the Real numbers ($\mathbb{R}$).

## Key Files
- **`Basic.lean`**: Axiomatic definition of the Sheffer primitive $eml(x, y) = \exp(x) - \ln(y)$.
- **`Arith.lean`**: Proving Professor Odrzywołek's depth-bounded identities for $\ln(z)$ and $\exp(x)$.
- **`PicoGPT.lean`**: The **Full Unification Theorem**, certifying that the entire GPT-2 pipeline is invariant under the EML rewrite.
- **`Attention.lean`**: Proving the equivalence of the NaN-proof Min-Plus attention mechanism.

## Verification
If these files compile, the mathematical identity of the EML architecture is machine-checked and "Zero-Sorry".
```bash
lake build EmlNN.PicoGPT
```
