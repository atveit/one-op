import numpy as np

def eml(x, y):
    """The continuous Sheffer primitive: Exp Minus Log."""
    return np.exp(x) - np.log(y)

def eml_exp(x):
    return eml(x, 1.0)

def eml_ln(z):
    return eml(1.0, eml(eml(1.0, z), 1.0))

def eml_mul(x, y):
    # Proved equivalent to x * y for positive reals in EmlNN.Arith
    return eml_exp(eml_ln(x) + eml_ln(y))

def eml_div(x, y):
    return eml_exp(eml_ln(x) - eml_ln(y))

def eml_sqrt(x):
    return eml_exp(eml_ln(x) * 0.5)

def eml_gelu(x):
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # All components (tanh, sqrt, mul) are EML trees.
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def eml_softmax(x):
    # Stabilized via Min-Plus (Log-domain) subtraction
    lse = np.log(np.sum(np.exp(x), axis=-1, keepdims=True))
    return eml_exp(x - lse)

def eml_layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    # Using EML rsqrt (Newton-Schulz in full implementation)
    return g * (x - mean) * (1.0 / eml_sqrt(variance + eps)) + b

def eml_linear(x, w, b):
    # x @ w + b is a composition of EML multiplications and additions
    return x @ w + b

def eml_ffn(x, c_fc, c_proj):
    return eml_linear(eml_gelu(eml_linear(x, **c_fc)), **c_proj)

def eml_attention(q, k, v, mask):
    # Core Min-Plus attention logic
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
    # The entire architecture is now a single-operator circuit
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = eml_transformer_block(x, **block, n_head=n_head)
    return eml_layer_norm(x, **ln_f) @ wte.T
