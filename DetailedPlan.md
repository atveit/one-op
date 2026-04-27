# Detailed Plan: High-Rigidity ANE-GPU Hybrid for Gemma 4 31B

## 1. Phase 1: High-Speed Substrate (Complete)
- [x] **AOT Weight Pre-processing:** Complete. Moved dequantization to init phase.
- [x] **Baseline Recovery:** Complete. Gemma 4 31B now runs at **23.67 tok/s** on GPU with EML norms.
- [x] **LUT Dequantization:** Verified. +32% speedup in dequantization logic.

## 2. Phase 2: "Proper" ANE Execution (In Progress)
- [ ] **Task 2.1: CoreML SwiGLU Block.**
    - Action: Finalize `build_fused_ane_mlp.py` using a non-blocking environment.
- [x] **Task 2.2: IOSurface Zero-Copy Bridge.**
    - Action: Preliminary `iosurface_bridge.py` infrastructure ready.
- [x] **Task 2.3: Layer-Level Concurrency.**
    - Action: Verified **20.6% architectural gain** via `smoke_test_overlap.py`.

## 3. Phase 3: Final Scrutiny & Launch
- [ ] **NPU Residency Check:** Use `powermetrics` once the CoreML bridge is active.
- [ ] **Verified Record:** Achieve >35 tok/s (Measured) on M3 Ultra.
- [ ] **Un-hide Blog Post:** Publish definitive results to `amund.blog`.
