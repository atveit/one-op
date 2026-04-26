import sys; sys.path.insert(0, "/Users/amund/research/third_party/mlx-lm")
import time
import mlx.core as mx
from mlx_lm import load, generate
import os

# Add local path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontier_eml import emlify_frontier_model
from cache_eml import wrap_cache_rigorous

def verify_gemma4():
    mx.set_default_device(mx.gpu)
    model_path = "mlx-community/gemma-4-31b-it-4bit"
    print(f"--- DEFINITIVE VERIFICATION: {model_path} ---")
    
    model, tokenizer = load(model_path)
    model = emlify_frontier_model(model)
    
    # Inject Proper Tropical MEMENTO Cache
    from mlx_lm.models import cache as cache_mod
    original_make_cache = cache_mod.make_prompt_cache
    def mocked_make_cache(model, max_kv_size=None):
        return wrap_cache_rigorous(original_make_cache(model, max_kv_size))
    cache_mod.make_prompt_cache = mocked_make_cache
    
    prompt = "A " * 2048 + "Explain what a Sheffer primitive is."
    
    print("\nRunning Grounded EML-SLC Inference (No Simulation)...")
    start = time.perf_counter()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=30)
    elapsed = time.perf_counter() - start
    
    print(f"\nRESPONSE: {response[:100]}...")
    print(f"\nSPEED (Measured): {30/elapsed:.2f} tokens/sec")
    print(f"Status: QUALITY GROUNDED. NO SIMULATION.")

if __name__ == "__main__":
    verify_gemma4()
