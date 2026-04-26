import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate

def main():
    parser = argparse.ArgumentParser(description="Official MLX Baseline Runner")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="A " * 3000 + "What is 2+2?")
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model, tokenizer = load(args.model)
    
    # Warmup
    _ = generate(model, tokenizer, prompt=args.prompt, max_tokens=10)
    
    start = time.time()
    response = generate(model, tokenizer, prompt=args.prompt, max_tokens=args.max_tokens)
    elapsed = time.time() - start
    
    print(f"\nSPEED: {args.max_tokens/elapsed:.2f} tokens/sec")

if __name__ == "__main__":
    main()
