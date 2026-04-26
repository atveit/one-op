import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/Users/amund/research/third_party/mlx-lm")

from mlx_lm.models import gpt2 as gpt2_std
from mlx_lm.models import cache as cache_std_mod
import gpt2 as gpt2_eml_mod
from cache_eml import TropicalMementoCache

def benchmark_mlx_lm():
    print("\n--- Benchmarking mlx-lm: GPT-2 Medium (355M) ---")
    args = gpt2_eml_mod.ModelArgs(n_layer=24, n_embd=1024, n_head=16) # GPT-2 Medium spec
    
    model_std = gpt2_std.Model(gpt2_std.ModelArgs(
        model_type="gpt2", n_ctx=2048, n_embd=1024, n_head=16, n_layer=24, 
        n_positions=2048, layer_norm_epsilon=1e-5, vocab_size=50257
    ))
    mx.eval(model_std.parameters())

    model_eml = gpt2_eml_mod.GPT2EML(args)
    mx.eval(model_eml.parameters())

    prompt = mx.random.randint(0, 50257, (1, 1024)) # Long prompt to fill cache
    n_tokens = 100
    
    # --- Standard Benchmark ---
    print("Running Standard mlx-lm generation...")
    cache_std = cache_std_mod.make_prompt_cache(model_std)
    _ = model_std(prompt, cache=cache_std)
    
    start = time.time()
    x = mx.array([[1]])
    for _ in range(n_tokens):
        logits = model_std(x, cache=cache_std)
        x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(x)
    std_time = time.time() - start
    std_tps = n_tokens / std_time

    # --- EML + SLC Benchmark ---
    print("Running EML + Tropical MEMENTO generation...")
    cache_eml = [TropicalMementoCache(block_size=64) for _ in range(args.n_layer)]
    _ = model_eml(prompt, cache=cache_eml)
    
    start = time.time()
    x = mx.array([[1]])
    for _ in range(n_tokens):
        logits = model_eml(x, cache=cache_eml)
        x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(x)
    eml_time = time.time() - start
    eml_tps = n_tokens / eml_time

    print(f"\nStandard mlx-lm: {std_tps:.2f} tokens/sec")
    print(f"EML + SLC optimized: {eml_tps:.2f} tokens/sec")
    print(f"Speedup: {((eml_tps/std_tps) - 1)*100:.1f}%")

if __name__ == "__main__":
    benchmark_mlx_lm()
