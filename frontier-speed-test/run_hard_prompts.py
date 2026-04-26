import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate
import json
import os
import sys

# Add local path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_optimized import emlify_model, wrap_cache_rigorous

def main():
    parser = argparse.ArgumentParser(description="Run 10 hard prompts on Qwen 3.6 EML")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3.6-35B-A3B-4bit")
    parser.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    
    # Load Prompts
    prompts_path = os.path.join(os.path.dirname(__file__), "../../blog_post/frontier_benchmark_prompts.json")
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
    
    print(f"\n--- [HARD PROMPTS] {args.model} | Mode: {args.mode} ---")
    model, tokenizer = load(args.model)
    
    if args.mode == "optimized":
        model = emlify_model(model)
        from mlx_lm.models import cache as cache_mod
        original_make_cache = cache_mod.make_prompt_cache
        def mocked_make_cache(model, max_kv_size=None):
            return wrap_cache_rigorous(original_make_cache(model, max_kv_size))
        cache_mod.make_prompt_cache = mocked_make_cache

    results = []
    for i, prompt in enumerate(prompts):
        print(f"\nRunning Prompt {i}...")
        start = time.time()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=256).strip()
        elapsed = time.time() - start
        
        print(f"PROMPT: {prompt[:50]}...")
        print(f"RESPONSE: {response[:100]}...")
        print(f"TIME: {elapsed:.2f}s")
        
        results.append({
            "prompt_idx": i,
            "prompt": prompt,
            "response": response,
            "time": elapsed
        })

    # Save results
    output_file = f"hard_prompts_results_{args.mode}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
