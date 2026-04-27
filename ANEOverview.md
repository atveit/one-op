# Apple Neural Engine (ANE) Overview

This document tracks specialized research and implementation patterns for offloading LLM compute to the Apple Neural Engine (NPU), specifically for the M3 Ultra (96GB).

## Core Resources
- [maderix/ANE](https://github.com/maderix/ANE): Low-level ANE research, compiler internals, and direct MIL examples.
- [ANEMLL](https://www.anemll.com): Production-grade library for offloading LLM layers (SwiGLU, Norms) to ANE via CoreML.
- [Orion Paper (2603.06728v1)](https://arxiv.org/html/2603.06728v1): "Orion: High-Speed ANE-GPU Hybrid Inference." Establishes the 3.3x speedup floor for 1x1 convolutions.

## Key Implementation Patterns

### 1. 1x1 Convolution Mapping
The ANE is optimized for 4D tensor operations `[B, C, H, W]`. For LLM Linear layers ($XW$), the most efficient mapping is a **1x1 Convolution** where:
- Input: `[1, D_in, 1, Seq]`
- Weight: `[D_out, D_in, 1, 1]`
- This bypasses the slower `matmul` unit on the NPU and utilizes the primary systolic array.

### 2. Zero-Copy IOSurfaces
To avoid DRAM round-trips (10x latency penalty):
- Bind MLX GPU buffers directly to `IOSurface`-backed `MLMultiArray` objects.
- Ensure **Alphabetical MIL Binding**: ANE maps inputs based on the alphabetical order of MIL variable names.
- Maintain **Uniform Sizing**: All shared surfaces should ideally be the same size to avoid kernel re-initialization.

### 3. FP4/INT8 Quantization
- **INT8 (W8A8)** is the "Golden Path" for ANE.
- **FP4 to FP16 LUT:** For 4-bit weights, use a 256-entry lookup table resident in the **96MB SLC** to bypass the overhead of `mx.dequantize` calls during the GPU -> ANE handoff.

## Knowledge Grounding (Smoke Tests)
- [ ] Verify ANE occupancy for a fused SwiGLU MIL kernel.
- [ ] Measure latency of 1x1 Conv vs standard MatMul on M3 Ultra NPU.
- [ ] Verify bit-for-bit parity of FP4-to-FP16 LUT dequantization.
