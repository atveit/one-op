import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_eml import emlify_frontier_model

def main():
    parser = argparse.ArgumentParser(description="EML/Tropical SLC Grounded Runner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Explain 2+2.")
    parser.add_argument("--max-tokens", type=int, default=30)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model, tokenizer = load(args.model)
    model = emlify_frontier_model(model)
    
    # --- PROPER SLC WINDOWING ---
    # We use the model's native sliding window support to stay in SLC
    # (Step B Acceleration)
    from mlx_lm.models.cache import RotatingKVCache
    original_make_cache = model.make_cache
    def mocked_make_cache():
        # Force a 1024-token SLC window for every layer
        return [RotatingKVCache(max_size=1024) for _ in range(len(model.layers))]
    model.make_cache = mocked_make_cache
    
    print(f"Running Grounded SLC-Windowed Inference (No Simulation)...")
    start = time.perf_counter()
    response = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens)
    elapsed = time.perf_counter() - start
    
    print(f"\nRESPONSE: {response[:100]}...")
    print(f"\nSPEED (Measured): {args.max_tokens/elapsed:.2f} tokens/sec")

if __name__ == "__main__":
    main()
