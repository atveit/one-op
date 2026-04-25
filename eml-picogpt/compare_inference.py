import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpt2_pico import gpt2 as gpt2_ref
from picoGPT_eml import eml_gpt2 as gpt2_eml
from utils import load_encoder_hparams_and_params

def compare(prompt, n_tokens=10):
    print(f"\n--- Comparing Prompt: \"{prompt}\" ---")
    
    # We use a mock/small setup if weights aren't available to ensure script runs
    # But here we try to load the real ones
    try:
        encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
        input_ids = encoder.encode(prompt)
        
        # 1. Standard Pass
        def generate(model_fn, ids):
            curr_ids = list(ids)
            for _ in range(n_tokens):
                logits = model_fn(curr_ids, **params, n_head=hparams["n_head"])
                next_id = np.argmax(logits[-1])
                curr_ids.append(int(next_id))
            return encoder.decode(curr_ids[len(ids):])

        ref_out = generate(gpt2_ref, input_ids)
        eml_out = generate(gpt2_eml, input_ids)
        
        print(f"Standard picoGPT: {ref_out}")
        print(f"EML-native:       {eml_out}")
        
    except Exception as e:
        print(f"Skipping real weight comparison due to loading error (likely TF/Hardware issue).")
        print("Running mathematical parity check on random tensors instead...")
        
        # Math Parity Check
        n_vocab, n_ctx, n_embd, n_head, n_layer = 100, 10, 64, 2, 2
        mock_params = {
            "wte": np.random.randn(n_vocab, n_embd),
            "wpe": np.random.randn(n_ctx, n_embd),
            "blocks": [{
                "ln_1": {"g": np.random.randn(n_embd), "b": np.random.randn(n_embd)},
                "ln_2": {"g": np.random.randn(n_embd), "b": np.random.randn(n_embd)},
                "attn": {
                    "c_attn": {"w": np.random.randn(n_embd, 3*n_embd), "b": np.random.randn(3*n_embd)},
                    "c_proj": {"w": np.random.randn(n_embd, n_embd), "b": np.random.randn(n_embd)},
                },
                "mlp": {
                    "c_fc": {"w": np.random.randn(n_embd, 4*n_embd), "b": np.random.randn(4*n_embd)},
                    "c_proj": {"w": np.random.randn(4*n_embd, n_embd), "b": np.random.randn(n_embd)},
                }
            } for _ in range(n_layer)],
            "ln_f": {"g": np.random.randn(n_embd), "b": np.random.randn(n_embd)}
        }
        
        test_ids = [1, 5, 20, 42]
        logits_ref = gpt2_ref(test_ids, **mock_params, n_head=n_head)
        logits_eml = gpt2_eml(test_ids, **mock_params, n_head=n_head)
        
        diff = np.abs(logits_ref - logits_eml).max()
        print(f"Max Logit Difference: {diff:.2e}")
        if diff < 1e-5:
            print("SUCCESS: EML circuit is mathematically identical to standard picoGPT.")
        else:
            print("FAILURE: Mathematical divergence detected.")

if __name__ == "__main__":
    prompts = [
        "The EML operator is",
        "Deep learning was always",
        "To prove that exp minus log is"
    ]
    for p in prompts:
        compare(p, n_tokens=10)
