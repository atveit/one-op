import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np
import torch
import torch.nn as nn

# --- 1. Define Fused SwiGLU in MIL ---
@mb.program(input_specs=[mb.TensorSpec(shape=(1, 5376, 1, 1))]) # Gemma 4 shape
def fused_swiglu(x):
    # Map Linear W1/W3 to 1x1 Convolution as per Source 3
    # Note: Using mock weights for the identity proof
    w1 = np.random.randn(21504, 5376, 1, 1).astype(np.float16)
    w3 = np.random.randn(21504, 5376, 1, 1).astype(np.float16)
    w2 = np.random.randn(5376, 21504, 1, 1).astype(np.float16)
    
    # OP 1: W1 Projection + SiLU
    gate = mb.conv(x=x, weight=w1, pad_type='valid', name="gate_proj")
    # SiLU(x) = x * sigmoid(x)
    sig = mb.sigmoid(x=gate)
    gate_activated = mb.mul(x=gate, y=sig)
    
    # OP 2: W3 Projection
    up = mb.conv(x=x, weight=w3, pad_type='valid', name="up_proj")
    
    # OP 3: Gated Product
    intermediate = mb.mul(x=gate_activated, y=up)
    
    # OP 4: W2 Projection
    out = mb.conv(x=intermediate, weight=w2, pad_type='valid', name="down_proj")
    
    return out

# --- 2. Convert to CoreML (Targeting ANE) ---
def build_ane_model():
    print("Building ANE-native MIL Model...")
    mlmodel = ct.convert(
        fused_swiglu,
        inputs=[ct.TensorType(shape=(1, 5376, 1, 1))],
        compute_units=ct.ComputeUnit.ALL # Allows ANE
    )
    mlmodel.save("one-op/frontier-speed-test/Gemma4_SwiGLU.mlpackage")
    print("Success! Saved Gemma4_SwiGLU.mlpackage")

if __name__ == "__main__":
    build_ane_model()
