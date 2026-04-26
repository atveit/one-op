import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate

def main():
    parser = argparse.ArgumentParser(description="Official MLX Baseline Runner")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face repo or local path")
    parser.add_argument("--prompt", type=str, default="Explain why mathematical unification is important for hardware efficiency.")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    print(f"\n--- [BASELINE] Loading: {args.model} ---")
    model, tokenizer = load(args.model)
    
    print("Running Inference...")
    # Warmup
    _ = generate(model, tokenizer, prompt=args.prompt, max_tokens=10)
    
    start = time.time()
    response = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens)
    elapsed = time.time() - start
    
    print(f"\nRESPONSE:\n{response}")
    print(f"\nSPEED: {args.max_tokens/elapsed:.2f} tokens/sec")

if __name__ == "__main__":
    main()
