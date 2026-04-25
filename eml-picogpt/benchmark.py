import numpy as np
import time
import os
import sys
import psutil
from transformers import AutoTokenizer

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpt2_pico import gpt2 as gpt2_ref
from picoGPT_eml import eml_gpt2 as gpt2_eml
from weights_loader import load_gpt2_weights_torch

def benchmark(prompt, n_tokens=5):
    print(f"Benchmarking Prompt: \"{prompt}\"")
    
    # 1. Load weights
    params, n_head = load_gpt2_weights_torch("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt)
    
    process = psutil.Process(os.getpid())
    
    # 2. Standard Pass
    mem_start = process.memory_info().rss
    start = time.time()
    curr_ids = list(input_ids)
    for _ in range(n_tokens):
        logits = gpt2_ref(curr_ids, **params, n_head=n_head)
        next_id = int(np.argmax(logits[-1]))
        curr_ids.append(next_id)
    std_time = time.time() - start
    std_out = tokenizer.decode(curr_ids[len(input_ids):])
    mem_std = (process.memory_info().rss - mem_start) / (1024 * 1024) # MB
    
    # 3. EML Pass
    mem_start = process.memory_info().rss
    start = time.time()
    curr_ids = list(input_ids)
    for _ in range(n_tokens):
        logits = gpt2_eml(curr_ids, **params, n_head=n_head)
        next_id = int(np.argmax(logits[-1]))
        curr_ids.append(next_id)
    eml_time = time.time() - start
    eml_out = tokenizer.decode(curr_ids[len(input_ids):])
    mem_eml = (process.memory_info().rss - mem_start) / (1024 * 1024) # MB
    
    return {
        "std_out": std_out,
        "eml_out": eml_out,
        "std_time": std_time,
        "eml_time": eml_time,
        "std_mem": mem_std,
        "eml_mem": mem_eml
    }

if __name__ == "__main__":
    # Fix for Mac threading issue in benchmark
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    prompts = [
        "The future of artificial intelligence",
        "Two plus two is",
        "The capital of France is"
    ]
    
    all_results = []
    for p in prompts:
        res = benchmark(p, n_tokens=8)
        all_results.append((p, res))
        print(f"  STD: {res['std_out']}")
        print(f"  EML: {res['eml_out']}")
        print(f"  Time: Std {res['std_time']:.2f}s, EML {res['eml_time']:.2f}s")
        
    # Aggregate stats
    avg_std = np.mean([r[1]['std_time'] for r in all_results])
    avg_eml = np.mean([r[1]['eml_time'] for r in all_results])
    
    print("\n--- FINAL STATS ---")
    print(f"Avg Time Standard: {avg_std:.3f}s")
    print(f"Avg Time EML:      {avg_eml:.3f}s")
    print(f"Overhead:          {(avg_eml/avg_std - 1)*100:.1f}%")
