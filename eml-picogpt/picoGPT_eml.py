import numpy as np
import os
import sys

# Add current directory to path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# EML Operator Primitives (Single-Operator Building Blocks)
# ---------------------------------------------------------------------------

def eml(x, y):
    """The continuous Sheffer primitive: Exp Minus Log."""
    return np.exp(x) - np.log(y)

def eml_exp(x):
    return eml(x, 1.0)

def eml_ln(z):
    # Safety clamp for log domain
    return eml(1.0, eml(eml(1.0, np.maximum(z, 1e-10)), 1.0))

def eml_softmax(x):
    # Stabilized via Min-Plus (Log-domain) subtraction
    # x - max(x) is standard trick for stability, also EML-expressible
    x_max = np.max(x, axis=-1, keepdims=True)
    logits = x - x_max
    lse = np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    return eml_exp(logits - lse)

def eml_layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    # Using EML rsqrt
    rsqrt = eml_exp(-0.5 * eml_ln(variance + eps))
    return g * (x - mean) * rsqrt + b

def eml_gelu(x):
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def eml_linear(x, w, b):
    return x @ w + b

def eml_ffn(x, c_fc, c_proj):
    return eml_linear(eml_gelu(eml_linear(x, **c_fc)), **c_proj)

def eml_attention(q, k, v, mask):
    logits = q @ k.T / np.sqrt(q.shape[-1]) + mask
    return eml_softmax(logits) @ v

def eml_mha(x, c_attn, c_proj, n_head):
    x = eml_linear(x, **c_attn)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1)))
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [eml_attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = eml_linear(np.hstack(out_heads), **c_proj)
    return x

def eml_transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + eml_mha(eml_layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + eml_ffn(eml_layer_norm(x, **ln_2), **mlp)
    return x

def eml_gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = eml_transformer_block(x, **block, n_head=n_head)
    return eml_layer_norm(x, **ln_f) @ wte.T

# ---------------------------------------------------------------------------
# Inference Wrapper
# ---------------------------------------------------------------------------

def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_to_generate), "generating (EML-native)"):
        logits = eml_gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate :]

def test_forward():
    print("Testing EML-native picoGPT forward pass...")
    n_vocab, n_ctx, n_embd, n_head, n_layer = 100, 10, 64, 2, 2
    
    # Mock params
    params = {
        "wte": np.random.randn(n_vocab, n_embd),
        "wpe": np.random.randn(n_ctx, n_embd),
        "blocks": [{
            "ln_1": {"g": np.ones(n_embd), "b": np.zeros(n_embd)},
            "ln_2": {"g": np.ones(n_embd), "b": np.zeros(n_embd)},
            "attn": {
                "c_attn": {"w": np.random.randn(n_embd, 3*n_embd), "b": np.zeros(3*n_embd)},
                "c_proj": {"w": np.random.randn(n_embd, n_embd), "b": np.zeros(n_embd)},
            },
            "mlp": {
                "c_fc": {"w": np.random.randn(n_embd, 4*n_embd), "b": np.zeros(4*n_embd)},
                "c_proj": {"w": np.random.randn(4*n_embd, n_embd), "b": np.zeros(n_embd)},
            }
        } for _ in range(n_layer)],
        "ln_f": {"g": np.ones(n_embd), "b": np.zeros(n_embd)}
    }
    
    inputs = [1, 2, 3, 4]
    logits = eml_gpt2(inputs, **params, n_head=n_head)
    print(f"Success! Output Logits Shape: {logits.shape}")
    assert logits.shape == (len(inputs), n_vocab)
    print("EML-native forward pass verified.")

def main(prompt: str = None, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models", test: bool = False):
    if test:
        test_forward()
        return

    if prompt is None:
        print("Usage: python3 picoGPT_eml.py \"Your prompt\"")
        return

    try:
        from utils import load_encoder_hparams_and_params
        encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
        input_ids = encoder.encode(prompt)
        output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
        print("\nPROMPT:", prompt)
        print("RESPONSE:", encoder.decode(output_ids))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Running forward pass test instead...")
        test_forward()

if __name__ == "__main__":
    import fire
    fire.Fire(main)
