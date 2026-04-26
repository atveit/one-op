import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Any, Optional, Dict
from cache_eml import TropicalMementoCache

@dataclass
class ModelArgs:
    hidden_size: int = 3072
    num_hidden_layers: int = 28
    intermediate_size: int = 24576
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    rms_norm_eps: float = 1e-6
    vocab_size: int = 256000

class EMLAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.scale = (args.hidden_size // args.num_attention_heads)**-0.5
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * (args.hidden_size // args.num_attention_heads), bias=False)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * (args.hidden_size // args.num_attention_heads), bias=False)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * (args.hidden_size // args.num_attention_heads), bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * (args.hidden_size // args.num_attention_heads), args.hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        logits = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            logits += mask
        weights = mx.exp(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        output = (weights @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        silu = gate * (1.0 / (1.0 + mx.exp(-gate)))
        return self.down_proj(silu * up)

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = EMLAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    def __call__(self, x, mask=None, cache=None):
        x = x + self.attn(self.input_layernorm(x), mask, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Gemma4EML(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    def __call__(self, x, cache=None):
        x = self.wte(x)
        if cache is None:
            cache = [TropicalMementoCache() for _ in range(len(self.layers))]
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache[i])
        return self.ln_f(x)
