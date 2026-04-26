# Formal Verification Stack

This directory contains the multi-language verification suite used to secure the EML framework across different layers of abstraction.

## The Formal Stack

| Folder | Tool | Layer Verified |
| :--- | :--- | :--- |
| **`gappa/`** | 🛡️ Gappa | **Numerics:** Bounds relative error within FP32 silicon limits. |
| **`tla+/`** | ⏱️ TLA+ | **Concurrency:** Proves the KV-cache and optimizers never deadlock. |
| **`smt/`** | 🛡️ Z3 | **Robustness:** Proves decision stability under adversarial noise. |
| **`coq/`** | 🎖️ Coq | **Compilers:** Secures hardware-level QJL bitwise kernels. |
| **`key/`** | 🤖 KeY | **Algorithmic:** Certifies BPE Tokenizer encode/decode invertibility. |

## Why so many languages?
Deep learning systems are complex. Lean proves the *math*, Gappa proves the *silicon*, and TLA+ proves the *distributed state*. Together, they provide a 360-degree formal audit of the EML substrate.
