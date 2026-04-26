import argparse
import time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import sys
import os

# --- 1. EML Substrate Primitives ---
def eml_rms_norm(original_norm, x):
    rms_sq = mx.mean(mx.square(x), axis=-1, keepdims=True)
    rsqrt = mx.exp(-0.5 * mx.log(rms_sq + original_norm.eps))
    for _ in range(3):
        rsqrt = 0.5 * rsqrt * (3.0 - (rms_sq + original_norm.eps) * mx.square(rsqrt))
    return original_norm.weight * x * rsqrt

class TropicalKVWrapper:
    def __init__(self, original_cache):
        self.orig = original_cache
    def update_and_fetch(self, keys, values):
        k, v = self.orig.update_and_fetch(keys, values)
        if k.shape[-2] > 2048:
             return k[:, :, -1024:, :], v[:, :, -1024:, :]
        return k, v
    def __getattr__(self, name):
        return getattr(self.orig, name)

def wrap_cache_rigorous(cache_list):
    if isinstance(cache_list, list):
        return [wrap_cache_rigorous(c) for c in cache_list]
    if hasattr(cache_list, "update_and_fetch"):
        return TropicalKVWrapper(cache_list)
    return cache_list

def emlify_model(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.RMSNorm):
            def hooked_norm(x, norm=module):
                return eml_rms_norm(norm, x)
            module.__call__ = hooked_norm
    return model

def main():
    parser = argparse.ArgumentParser(description="EML/Tropical SLC Optimized Runner")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face repo or local path")
    parser.add_argument("--prompt", type=str, default="Explain why mathematical unification is important for hardware efficiency.")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    print(f"\n--- [EML-SLC] Loading: {args.model} ---")
    model, tokenizer = load(args.model)
    model = emlify_model(model)
    
    # Inject Tropical Cache
    from mlx_lm.models import cache as cache_mod
    original_make_cache = cache_mod.make_prompt_cache
    def mocked_make_cache(model, max_kv_size=None):
        return wrap_cache_rigorous(original_make_cache(model, max_kv_size))
    cache_mod.make_prompt_cache = mocked_make_cache
    
    print("Running Optimized Inference...")
    # Warmup
    _ = generate(model, tokenizer, prompt=args.prompt, max_tokens=10)
    
    start = time.time()
    response = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens)
    elapsed = time.time() - start
    
    # Restore
    cache_mod.make_prompt_cache = original_make_cache
    
    print(f"\nRESPONSE:\n{response}")
    print(f"\nSPEED (EML-SLC): {args.max_tokens/elapsed:.2f} tokens/sec")

if __name__ == "__main__":
    main()
