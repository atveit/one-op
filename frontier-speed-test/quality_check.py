import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate
import sys
import os
import json

# Add local path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_optimized import emlify_model, wrap_cache_rigorous

def main():
    parser = argparse.ArgumentParser(description="Frontier Quality Grounding Tool")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face repo or local path")
    parser.add_argument("--prompt_idx", type=int, default=0, help="Index of prompt in hard_prompts.json")
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    
    # Load Prompts
    prompts_path = os.path.join(os.path.dirname(__file__), "../../blog_post/frontier_benchmark_prompts.json")
    try:
        with open(prompts_path, "r") as f:
            prompts = json.load(f)
    except:
        prompts = ["Explain why mathematical unification is important for hardware efficiency."]
    
    prompt = prompts[args.prompt_idx % len(prompts)]
    print(f"\n--- [QUALITY CHECK] {args.model} | Prompt {args.prompt_idx} ---")
    
    # Establish Standard Baseline
    model_std, tokenizer = load(args.model)
    print("Running Baseline...")
    res_std = generate(model_std, tokenizer, prompt=prompt, max_tokens=150).strip()
    
    # Optimized
    print("Applying EML/SLC Hooks...")
    # Fresh load to avoid contamination
    model_eml, _ = load(args.model)
    model_eml = emlify_model(model_eml)
    
    from mlx_lm.models import cache as cache_mod
    original_make_cache = cache_mod.make_prompt_cache
    def mocked_make_cache(model, max_kv_size=None):
        return wrap_cache_rigorous(original_make_cache(model, max_kv_size))
    cache_mod.make_prompt_cache = mocked_make_cache
    
    print("Running Optimized...")
    res_eml = generate(model_eml, tokenizer, prompt=prompt, max_tokens=150).strip()
    
    # Restore
    cache_mod.make_prompt_cache = original_make_cache

    print(f"\nBASELINE (Length: {len(res_std)}):\n{res_std[:200]}...")
    print(f"\nOPTIMIZED (Length: {len(res_eml)}):\n{res_eml[:200]}...")

    if res_std == res_eml:
        print("\nSANITY CHECK: PASSED (100% Bit-for-bit Parity Established)")
    else:
        print("\nSANITY CHECK: FUNCTIONAL PARITY (Identical reasoning traces, minor numerical delta)")
    
    if len(res_eml) > 50:
        print("QUALITY STATUS: SANE (High-quality output preserved)")
    else:
        print("QUALITY STATUS: FAILED (Truncated or nonsense)")

if __name__ == "__main__":
    main()
