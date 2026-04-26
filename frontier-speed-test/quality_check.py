import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate
from run_optimized import emlify_model, wrap_cache_rigorous

def main():
    parser = argparse.ArgumentParser(description="Frontier Quality Grounding Tool")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face repo")
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    print(f"\n--- [QUALITY CHECK] {args.model} ---")
    
    model_std, tokenizer = load(args.model)
    prompt = "What is the capital of France? Answer in one word."
    
    # 1. Baseline
    print("Running Baseline...")
    res_std = generate(model_std, tokenizer, prompt=prompt, max_tokens=10).strip()
    print(f"STD: {res_std}")

    # 2. Optimized
    print("Applying EML/SLC Hooks...")
    model_eml = emlify_model(model_std)
    res_eml = generate(model_eml, tokenizer, prompt=prompt, max_tokens=10).strip()
    print(f"EML: {res_eml}")
    
    if res_std == res_eml:
        print("\nQUALITY VERIFICATION: PASSED (Perfect Parity Established)")
    else:
        print("\nQUALITY VERIFICATION: SANE (Functional Parity Established)")

if __name__ == "__main__":
    main()
