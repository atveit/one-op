import time
import mlx.core as mx
from mlx_lm import load
from frontier_eml import emlify_frontier_model

def smoke_test():
    mx.set_default_device(mx.gpu)
    model_path = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    print(f"--- SMOKE TEST: SLC Tiling for {model_path} ---")
    model, tokenizer = load(model_path)
    
    # 2048 tokens will hit the 96MB SLC Wall for a 1.5B model
    prompt = mx.random.randint(0, 151936, (1, 2048))
    
    # 1. Standard (No Tiling)
    print("\nRunning Standard Prefill...")
    start = time.perf_counter()
    _ = model(prompt)
    mx.eval(_)
    std_time = time.perf_counter() - start
    print(f"Standard Time: {std_time:.4f}s")
    
    # 2. Optimized (SLC Tiling)
    print("\nApplying EML-SLC Tiling...")
    model = emlify_frontier_model(model)
    start = time.perf_counter()
    _ = model(prompt)
    mx.eval(_)
    slc_time = time.perf_counter() - start
    print(f"SLC-Tiled Time: {slc_time:.4f}s")
    
    print(f"\nSLC-Resident Speedup: {((std_time/slc_time) - 1)*100:.1f}%")

if __name__ == "__main__":
    smoke_test()
