import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass
from typing import Any, Optional, Dict, Union
from cache_eml import TropicalMementoCache

@dataclass
class ModelArgs:
    model_type: str = "qwen2_moe"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 11008
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    num_experts: int = 16
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 2048
    shared_expert_intermediate_size: int = 11008
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936

class EMLAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.scale = (args.hidden_size // args.num_attention_heads)**-0.5
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * (args.hidden_size // args.num_attention_heads), bias=True)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * (args.hidden_size // args.num_attention_heads), bias=True)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * (args.hidden_size // args.num_attention_heads), bias=True)
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

class EMLSwitchGLU(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.shared_expert = nn.Sequential(
            nn.Linear(args.hidden_size, args.shared_expert_intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(args.shared_expert_intermediate_size, args.hidden_size, bias=False)
        )

    def __call__(self, x):
        return self.shared_expert(x)

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = EMLAttention(args)
        self.mlp = EMLSwitchGLU(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    def __call__(self, x, mask=None, cache=None):
        x = x + self.attn(self.input_layernorm(x), mask, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Qwen36EML(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
    def __call__(self, x, cache=None):
        x = self.wte(x)
        if cache is None:
            cache = [TropicalMementoCache() for _ in range(len(self.layers))]
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache[i])
        return x
