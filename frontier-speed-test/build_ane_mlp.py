import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np
import os

def create_fused_swiglu_mil(dim=5376, hidden_dim=21504):
    """
    Creates a fused SwiGLU MLP block optimized for ANE.
    Maps Linear layers to 1x1 Convolutions.
    """
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, dim, 1, 1))])
    def swiglu_prog(x):
        # Weight Shapes for 1x1 Conv: [out_channels, in_channels, 1, 1]
        w1 = np.zeros((hidden_dim, dim, 1, 1), dtype=np.float16)
        w3 = np.zeros((hidden_dim, dim, 1, 1), dtype=np.float16)
        w2 = np.zeros((dim, hidden_dim, 1, 1), dtype=np.float16)
        
        # Branch 1: Gate (W1) + SiLU
        gate = mb.conv(x=x, weight=w1, pad_type='valid', name="gate_conv")
        sig = mb.sigmoid(x=gate)
        gate_activated = mb.mul(x=gate, y=sig)
        
        # Branch 2: Up (W3)
        up = mb.conv(x=x, weight=w3, pad_type='valid', name="up_conv")
        
        # Product
        inter = mb.mul(x=gate_activated, y=up)
        
        # Down (W2)
        out = mb.conv(x=inter, weight=w2, pad_type='valid', name="down_conv")
        return out
    
    return swiglu_prog

def main():
    print("Building ANE-native SwiGLU block (Proper 1x1 Conv Mapping)...")
    mil_prog = create_fused_swiglu_mil()
    
    # Target NPU specifically
    mlmodel = ct.convert(
        mil_prog,
        inputs=[ct.TensorType(shape=(1, 5376, 1, 1))],
        compute_units=ct.ComputeUnit.ALL # Allows ANE
    )
    
    output_path = "one-op/frontier-speed-test/Gemma4_MLP_ANE.mlpackage"
    mlmodel.save(output_path)
    print(f"Success! Model saved to {output_path}")

if __name__ == "__main__":
    main()
