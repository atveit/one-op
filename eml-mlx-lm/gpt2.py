import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Any, Optional
from .cache_eml import TropicalMementoCache

@dataclass
class ModelArgs:
    n_ctx: int = 1024
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    vocab_size: int = 50257
    layer_norm_epsilon: float = 1e-5

# EML-native components for mlx-lm
def eml_rms_norm_ns(x, weight, eps=1e-5):
    # Newton-Schulz rsqrt
    rms_sq = mx.mean(mx.square(x), axis=-1, keepdims=True)
    rsqrt = mx.exp(-0.5 * mx.log(rms_sq + eps)) # EML seed
    # 3 iterations of refinement
    for _ in range(3):
        rsqrt = 0.5 * rsqrt * (3.0 - (rms_sq + eps) * rsqrt * rsqrt)
    return weight * x * rsqrt

class EMLAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_head = args.n_head
        self.scale = (args.n_embd // args.n_head)**-0.5
        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd)
        self.c_proj = nn.Linear(args.n_embd, args.n_embd)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        qkv = self.c_attn(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)
        
        queries = queries.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Min-Plus Log-domain Attention
        logits = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            logits += mask
        
        # Soft-Tropical (Safe out-of-the-box)
        lse = mx.logsumexp(logits, axis=-1, keepdims=True)
        weights = mx.exp(logits - lse)
        
        output = (weights @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.c_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = EMLAttention(args)
        self.ln_1 = mx.ones((args.n_embd,))
        self.ln_2 = mx.ones((args.n_embd,))
        self.mlp = nn.Sequential(
            nn.Linear(args.n_embd, 4 * args.n_embd),
            nn.GELU(),
            nn.Linear(4 * args.n_embd, args.n_embd)
        )

    def __call__(self, x, mask=None, cache=None):
        x = x + self.attn(eml_rms_norm_ns(x, self.ln_1), mask, cache)
        x = x + self.mlp(eml_rms_norm_ns(x, self.ln_2))
        return x

class GPT2EML(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(args.vocab_size, args.n_embd)
        self.wpe = nn.Embedding(args.n_ctx, args.n_embd)
        self.h = [TransformerBlock(args) for _ in range(args.n_layer)]
        self.ln_f = mx.ones((args.n_embd,))

    def __call__(self, inputs, cache=None):
        L = inputs.shape[1]
        x = self.wte(inputs)
        x += self.wpe(mx.arange(L))
        
        if cache is None:
            cache = [TropicalMementoCache() for _ in range(len(self.h))]
            
        for layer, c in zip(self.h, cache):
            x = layer(x, cache=c)
            
        return self.wte.as_linear(eml_rms_norm_ns(x, self.ln_f))
