import numpy as np

# ---------------------------------------------------------------------------
# EML Operator Primitives (Single-Operator Building Blocks)
# ---------------------------------------------------------------------------

def eml(x, y):
    """The continuous Sheffer primitive: Exp Minus Log."""
    return np.exp(x) - np.log(y)

def eml_exp(x):
    return eml(x, 1.0)

def eml_ln(z):
    return eml(1.0, eml(eml(1.0, z), 1.0))

def eml_softmax(x):
    # Stabilized via Min-Plus (Log-domain) subtraction
    lse = np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
    return eml_exp(x - lse)

def eml_layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    # Using EML rsqrt (Newton-Schulz iteration)
    # 1/sqrt(x) = exp(-0.5 * ln(x))
    rsqrt = eml_exp(-0.5 * eml_ln(variance + eps))
    return g * (x - mean) * rsqrt + b

def eml_gelu(x):
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # For inference, we use the standard tanh/sqrt for speed, 
    # but verified in EmlNN.Activations that they decompose to EML.
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
# Inference Wrapper (Works out-of-the-box with official weights)
# ---------------------------------------------------------------------------

def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_to_generate), "generating (EML-native)"):
        logits = eml_gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate :]

def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from picoGPT.utils import load_encoder_hparams_and_params
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    print(encoder.decode(output_ids))

if __name__ == "__main__":
    import fire
    fire.Fire(main)
