import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate
import json
import os
import sys

# Add local path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_eml import emlify_frontier_model
from cache_eml import wrap_cache_rigorous

def main():
    parser = argparse.ArgumentParser(description="Run 10 hard prompts on Frontier EML")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", choices=["baseline", "optimized"], required=True)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    
    # Load Prompts
    prompts_path = os.path.join(os.path.dirname(__file__), "../../blog_post/frontier_benchmark_prompts.json")
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
    
    print(f"\n--- [HARD PROMPTS] {args.model} | Mode: {args.mode} ---")
    model, tokenizer = load(args.model)
    
    if args.mode == "optimized":
        print("Applying EML/SLC Frontier Optimizations...")
        model = emlify_frontier_model(model)
        from mlx_lm.models import cache as cache_mod
        original_make_cache = cache_mod.make_prompt_cache
        def mocked_make_cache(model, max_kv_size=None):
            return wrap_cache_rigorous(original_make_cache(model, max_kv_size))
        cache_mod.make_prompt_cache = mocked_make_cache

    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[PROMPT {i}] Running...")
        start = time.time()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=150).strip()
        elapsed = time.time() - start
        
        print(f"RESPONSE:\n{response[:200]}...")
        print(f"TIME: {elapsed:.2f}s | SPEED: {150/elapsed:.2f} tok/s")
        
        results.append({
            "prompt_idx": i,
            "prompt": prompt,
            "response": response,
            "time": elapsed
        })

    # Save results
    output_file = f"hard_prompts_results_{args.model.split('/')[-1]}_{args.mode}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
