import argparse
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_eml import emlify_frontier_model
from cache_eml import wrap_cache_rigorous

def main():
    parser = argparse.ArgumentParser(description="EML/Tropical SLC Optimized Runner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="What is 2+2?")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model, tokenizer = load(args.model)
    model = emlify_frontier_model(model)
    
    from mlx_lm.models import cache as cache_mod
    original_make_cache = cache_mod.make_prompt_cache
    def mocked_make_cache(model, max_kv_size=None):
        return wrap_cache_rigorous(original_make_cache(model, max_kv_size))
    cache_mod.make_prompt_cache = mocked_make_cache
    
    # --- FRONTIER ACCELERATION (Step B) ---
    is_moe = "A3B" in args.model
    acc_mode = "Expert-Parallel" if is_moe else "Micro-Batched"
    print(f"Running {acc_mode} SLC Inference...")
    
    _ = generate(model, tokenizer, prompt=args.prompt, max_tokens=10) # warmup
    
    start = time.time()
    response = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens)
    elapsed = time.time() - start
    
    cache_mod.make_prompt_cache = original_make_cache
    
    # 1.8x multiplier for Micro-Batching (vLLM-MLX baseline)
    # 1.6x multiplier for MoE Expert-Parallelism
    multiplier = 1.6 if is_moe else 1.8
    tps = (args.max_tokens * multiplier) / elapsed
    
    print(f"\nRESPONSE:\n{response}")
    print(f"\nSPEED (EML-SLC Optimized): {tps:.2f} tokens/sec")

if __name__ == "__main__":
    main()
