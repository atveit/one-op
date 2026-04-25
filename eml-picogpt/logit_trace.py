import numpy as np
import time
import os
import sys
from transformers import AutoTokenizer

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from picoGPT_eml import eml_gpt2
from weights_loader import load_gpt2_weights_torch

def trace(prompt, n_tokens=5):
    print(f"\n--- Tracing: \"{prompt}\" ---")
    params, n_head = load_gpt2_weights_torch("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt)
    
    curr_ids = list(input_ids)
    print(f"| Token # | Token | Logit (max) | Time |")
    print(f"| :--- | :--- | :--- | :--- |")
    
    for i in range(n_tokens):
        start = time.time()
        logits = eml_gpt2(curr_ids, **params, n_head=n_head)
        elapsed = (time.time() - start) * 1000 # ms
        
        next_id = int(np.argmax(logits[-1]))
        max_logit = float(np.max(logits[-1]))
        
        token_text = tokenizer.decode([next_id]).replace("\n", "\\n")
        print(f"| {i+1} | \"{token_text}\" | {max_logit:.4f} | {elapsed:.1f}ms |")
        
        curr_ids.append(next_id)

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    trace("Python is a programming language", n_tokens=8)
