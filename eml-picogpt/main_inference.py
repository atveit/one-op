import numpy as np
import os
import sys
from transformers import AutoTokenizer

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from picoGPT_eml import eml_gpt2
from load_weights_torch import load_gpt2_weights_torch

def run_real_inference(prompt, n_tokens=10):
    params, n_head = load_gpt2_weights_torch("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt)
    curr_ids = list(input_ids)
    for i in range(n_tokens):
        logits = eml_gpt2(curr_ids, **params, n_head=n_head)
        next_id = int(np.argmax(logits[-1]))
        curr_ids.append(next_id)
    return tokenizer.decode(curr_ids)

if __name__ == "__main__":
    prompts = [
        "The future of artificial intelligence",
        "Python is a programming language",
        "The capital of France is"
    ]
    for p in prompts:
        print(f"\nPROMPT: {p}")
        out = run_real_inference(p, n_tokens=10)
        print(f"EML-OUTPUT: {out}")
