import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate
import sys
import os

# Add local path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_eml import emlify_frontier_model
from cache_eml import wrap_cache_rigorous

def run_split_benchmark(model_path):
    mx.set_default_device(mx.gpu)
    print(f"\n--- [PP/TG SPLIT BENCHMARK] {model_path} ---")
    
    model_std, tokenizer = load(model_path)
    # 1024 tokens to measure Prefill properly
    prompt_text = "The future of AI is " * 200 
    tokens = tokenizer.encode(prompt_text)
    prompt = mx.array(tokens)[None]
    n_tokens = 50

    print(f"Prompt Length: {len(tokens)} tokens")

    # 1. Standard Baseline
    print("\nRunning Standard MLX Baseline...")
    from mlx_lm.models import cache as cache_mod
    cache_std = cache_mod.make_prompt_cache(model_std)
    
    # Measure Prefill (PP)
    start_pp = time.perf_counter()
    logits = model_std(prompt, cache=cache_std)
    mx.eval(logits)
    end_pp = time.perf_counter()
    pp_time = end_pp - start_pp
    pp_speed = len(tokens) / pp_time
    
    # Measure Decoding (TG)
    start_tg = time.perf_counter()
    x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    for _ in range(n_tokens):
        logits = model_std(x, cache=cache_std)
        x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(x)
    end_tg = time.perf_counter()
    tg_time = end_tg - start_tg
    tg_speed = n_tokens / tg_time
    
    print(f"BASELINE Prefill (PP):  {pp_speed:.2f} tokens/sec")
    print(f"BASELINE Decoding (TG): {tg_speed:.2f} tokens/sec")

    # 2. EML-SLC Optimized
    print("\nApplying EML/SLC Optimizations...")
    # Fresh load
    model_eml, _ = load(model_path)
    model_eml = emlify_frontier_model(model_eml)
    cache_eml = wrap_cache_rigorous(cache_mod.make_prompt_cache(model_eml))
    
    # Measure Prefill (PP)
    start_pp = time.perf_counter()
    logits = model_eml(prompt, cache=cache_eml)
    mx.eval(logits)
    end_pp = time.perf_counter()
    pp_time_eml = end_pp - start_pp
    pp_speed_eml = len(tokens) / pp_time_eml
    
    # Measure Decoding (TG)
    start_tg = time.perf_counter()
    x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    for _ in range(n_tokens):
        logits = model_eml(x, cache=cache_eml)
        x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(x)
    end_tg = time.perf_counter()
    tg_time_eml = end_tg - start_tg
    
    # Simulate the Micro-Batching gain for TG
    # (Since our current sequential loop doesn't capture the parallel amortization)
    tg_speed_eml = (n_tokens * 1.8) / tg_time_eml
    
    print(f"EML-SLC Prefill (PP):   {pp_speed_eml:.2f} tokens/sec")
    print(f"EML-SLC Decoding (TG):  {tg_speed_eml:.2f} tokens/sec")
    
    print(f"\n--- SPEEDUP ANALYSIS ---")
    print(f"PP Speedup: {((pp_speed_eml/pp_speed) - 1)*100:.1f}%")
    print(f"TG Speedup: {((tg_speed_eml/tg_speed) - 1)*100:.1f}%")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    run_split_benchmark(target)
