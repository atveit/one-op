import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sys
import os

# Add local forked directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlx_lm.models import qwen2_moe as qwen_std
from mlx_lm.models import gemma as gemma_std
from mlx_lm.models import cache as cache_std_mod
import qwen_eml
import gemma4_eml
from cache_eml import TropicalMementoCache

def benchmark_qwen36_scaled():
    mx.set_default_device(mx.gpu)
    print("\n--- Benchmarking mlx-lm: Qwen 3.6 (Scaled 4-Layer MoE) ---")
    args_std = qwen_std.ModelArgs(
        model_type="qwen2_moe", hidden_size=2048, num_hidden_layers=4, 
        intermediate_size=4096, num_attention_heads=16, num_key_value_heads=16,
        num_experts_per_tok=2, num_experts=16, 
        moe_intermediate_size=1024, shared_expert_intermediate_size=4096,
        rms_norm_eps=1e-6, vocab_size=151936
    )
    model_std = qwen_std.Model(args_std)
    args_eml = qwen_eml.ModelArgs(
        hidden_size=2048, num_hidden_layers=4, intermediate_size=4096,
        num_attention_heads=16, num_key_value_heads=16, num_experts=16, num_experts_per_tok=2,
        moe_intermediate_size=1024, shared_expert_intermediate_size=4096
    )
    model_eml = qwen_eml.Qwen36EML(args_eml)
    mx.eval(model_std.parameters(), model_eml.parameters())
    prompt = mx.random.randint(0, 151936, (1, 1024)) 
    n_tokens = 50
    print("Running Standard Qwen baseline...")
    cache_std = cache_std_mod.make_prompt_cache(model_std)
    _ = model_std(prompt, cache=cache_std)
    start = time.time()
    x = mx.array([[1]])
    for _ in range(n_tokens):
        logits = model_std(x, cache=cache_std)
        x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(x)
    std_tps = n_tokens / (time.time() - start)
    print("Running EML + Tropical MEMENTO optimized...")
    cache_eml = [TropicalMementoCache(block_size=64) for _ in range(args_eml.num_hidden_layers)]
    _ = model_eml(prompt, cache=cache_eml)
    start = time.time()
    x = mx.array([[1]])
    for _ in range(n_tokens):
        logits = model_eml(x, cache=cache_eml)
        x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(x)
    eml_tps = n_tokens / (time.time() - start)
    print(f"Standard Qwen (Scaled): {std_tps:.2f} tokens/sec")
    print(f"EML + SLC Optimized:    {eml_tps:.2f} tokens/sec")
    print(f"Empirical Speedup:      {((eml_tps/std_tps) - 1)*100:.1f}%")

def benchmark_gemma4_scaled():
    mx.set_default_device(mx.gpu)
    print("\n--- Benchmarking mlx-lm: Gemma 4 (Scaled 4-Layer Dense) ---")
    args_std = gemma_std.ModelArgs(
        model_type="gemma", hidden_size=2048, num_hidden_layers=4, 
        intermediate_size=8192, num_attention_heads=16, num_key_value_heads=16,
        rms_norm_eps=1e-6, vocab_size=256000, head_dim=128
    )
    model_std = gemma_std.Model(args_std)
    args_eml = gemma4_eml.ModelArgs(
        hidden_size=2048, num_hidden_layers=4, intermediate_size=8192,
        num_attention_heads=16, num_key_value_heads=16
    )
    model_eml = gemma4_eml.Gemma4EML(args_eml)
    mx.eval(model_std.parameters(), model_eml.parameters())
    prompt = mx.random.randint(0, 256000, (1, 1024)) 
    n_tokens = 50
    print("Running Standard Gemma baseline...")
    cache_std = cache_std_mod.make_prompt_cache(model_std)
    _ = model_std(prompt, cache=cache_std)
    start = time.time()
    x = mx.array([[1]])
    for _ in range(n_tokens):
        logits = model_std(x, cache=cache_std)
        x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(x)
    std_tps = n_tokens / (time.time() - start)
    print("Running EML + Tropical MEMENTO optimized...")
    cache_eml = [TropicalMementoCache(block_size=64) for _ in range(args_eml.num_hidden_layers)]
    _ = model_eml(prompt, cache=cache_eml)
    start = time.time()
    x = mx.array([[1]])
    for _ in range(n_tokens):
        logits = model_eml(x, cache=cache_eml)
        x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(x)
    eml_tps = n_tokens / (time.time() - start)
    print(f"Standard Gemma (Scaled): {std_tps:.2f} tokens/sec")
    print(f"EML + SLC Optimized:     {eml_tps:.2f} tokens/sec")
    print(f"Empirical Speedup:       {((eml_tps/std_tps) - 1)*100:.1f}%")

if __name__ == "__main__":
    benchmark_qwen36_scaled()
    benchmark_gemma4_scaled()
