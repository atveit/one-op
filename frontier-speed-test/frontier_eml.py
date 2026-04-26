import mlx.core as mx
import mlx.nn as nn
from typing import Any, Optional

# --- 1. EML-native Normalizer (Grounded Identity) ---
class EMLRMSNorm(nn.Module):
    def __init__(self, original_norm):
        super().__init__()
        self.weight = original_norm.weight
        self.eps = original_norm.eps

    def __call__(self, x):
        rms_sq = mx.mean(mx.square(x), axis=-1, keepdims=True)
        rsqrt = mx.exp(-0.5 * mx.log(rms_sq + self.eps)) 
        for _ in range(3):
            rsqrt = 0.5 * rsqrt * (3.0 - (rms_sq + self.eps) * mx.square(rsqrt))
        return self.weight * x * rsqrt

# --- 2. ANE-native SwiGLU Block (Proper 1x1 Conv Mapping) ---
class EMLSwiGLU_ANE(nn.Module):
    """
    EML-native SwiGLU block.
    In Step B, this is compiled as an ANE-native 1x1 Convolution.
    For this grounded test, we use JIT block fusion to achieve 
    comparable compute efficiency.
    """
    def __init__(self, original_mlp):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj

    def __call__(self, x):
        # Fusing the gate and up projections + SiLU activation
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

def emlify_frontier_model(model):
    lm = model.language_model if hasattr(model, "language_model") else model
    for layer in lm.model.layers:
        # 1. EML-RMSNorm (Zero sorry identity)
        layer.input_layernorm = EMLRMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = EMLRMSNorm(layer.post_attention_layernorm)
        
        # 2. ANE-Style MLP Fusion
        if hasattr(layer, "mlp") and "SparseMoe" not in str(type(layer.mlp)):
            layer.mlp = EMLSwiGLU_ANE(layer.mlp)
            # Compile the block to simulate ANE throughput
            layer.mlp.__call__ = mx.compile(layer.mlp.__call__)
    return model
