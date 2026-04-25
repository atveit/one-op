import torch
import numpy as np
from transformers import GPT2LMHeadModel

def load_gpt2_weights_torch(model_size="gpt2"):
    print(f"Loading {model_size} weights via PyTorch/Transformers...")
    model = GPT2LMHeadModel.from_pretrained(model_size)
    sd = model.state_dict()
    
    # Map PyTorch state dict to picoGPT style params
    params = {"blocks": []}
    
    # Embeddings
    params["wte"] = sd["transformer.wte.weight"].numpy()
    params["wpe"] = sd["transformer.wpe.weight"].numpy()
    
    # Blocks
    n_layer = model.config.n_layer
    for i in range(n_layer):
        block = {
            "ln_1": {
                "g": sd[f"transformer.h.{i}.ln_1.weight"].numpy(),
                "b": sd[f"transformer.h.{i}.ln_1.bias"].numpy(),
            },
            "ln_2": {
                "g": sd[f"transformer.h.{i}.ln_2.weight"].numpy(),
                "b": sd[f"transformer.h.{i}.ln_2.bias"].numpy(),
            },
            "attn": {
                "c_attn": {
                    "w": sd[f"transformer.h.{i}.attn.c_attn.weight"].numpy(),
                    "b": sd[f"transformer.h.{i}.attn.c_attn.bias"].numpy(),
                },
                "c_proj": {
                    "w": sd[f"transformer.h.{i}.attn.c_proj.weight"].numpy(),
                    "b": sd[f"transformer.h.{i}.attn.c_proj.bias"].numpy(),
                },
            },
            "mlp": {
                "c_fc": {
                    "w": sd[f"transformer.h.{i}.mlp.c_fc.weight"].numpy(),
                    "b": sd[f"transformer.h.{i}.mlp.c_fc.bias"].numpy(),
                },
                "c_proj": {
                    "w": sd[f"transformer.h.{i}.mlp.c_proj.weight"].numpy(),
                    "b": sd[f"transformer.h.{i}.mlp.c_proj.bias"].numpy(),
                },
            },
        }
        params["blocks"].append(block)
    
    # Final LayerNorm
    params["ln_f"] = {
        "g": sd["transformer.ln_f.weight"].numpy(),
        "b": sd["transformer.ln_f.bias"].numpy(),
    }
    
    return params, model.config.n_head

if __name__ == "__main__":
    params, n_head = load_gpt2_weights_torch()
    print("Success! Weights loaded into NumPy.")
