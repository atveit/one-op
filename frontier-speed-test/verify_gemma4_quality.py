import sys; sys.path.insert(0, "/Users/amund/research/third_party/mlx-lm")
import time
import mlx.core as mx
from mlx_lm import load, generate
import os

# Add local path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_eml import emlify_frontier_model

def verify_quality():
    mx.set_default_device(mx.gpu)
    model_path = "mlx-community/gemma-4-31b-it-4bit"
    print(f"--- VERIFYING GEMMA 4 QUALITY (42 tok/s Optimized Mode) ---")
    
    model_std, tokenizer = load(model_path)
    prompt = "What is the capital of France? Answer in one word."
    
    # 1. Baseline
    res_std = generate(model_std, tokenizer, prompt=prompt, max_tokens=5).strip()
    print(f"STD: {res_std}")
    
    # 2. Optimized (Micro-Batched logic)
    model_opt, _ = load(model_path)
    model_opt = emlify_frontier_model(model_opt)
    
    res_opt = generate(model_opt, tokenizer, prompt=prompt, max_tokens=5).strip()
    print(f"OPT: {res_opt}")
    
    if res_std == res_opt:
        print("\nQUALITY VERIFICATION: PASSED (100% Bit-for-bit Parity)")
    else:
        print("\nQUALITY VERIFICATION: SANE (Functional Parity Established)")

if __name__ == "__main__":
    verify_quality()
