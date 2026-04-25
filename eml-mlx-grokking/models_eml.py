import mlx.nn as nn
import mlx.core as mx
import numpy as np

# ---------------------------------------------------------------------------
# EML Dual-Space Primitives
# ---------------------------------------------------------------------------

def eml(x, y):
    """The continuous Sheffer primitive: Exp Minus Log."""
    return mx.exp(x) - mx.log(y)

def eml_exp(x):
    return eml(x, 1.0)

def eml_ln(z):
    # Safety clamp for log domain
    return eml(1.0, eml(eml(1.0, mx.maximum(z, 1e-10)), 1.0))

def eml_rsqrt_ns(x, eps=1e-8, iterations=3):
    """Newton-Schulz Iterative Refinement for 1/sqrt(x)"""
    xe = x + eps
    # Seed
    y = mx.array(1.0) / mx.exp(0.5 * eml_ln(xe))
    three = mx.array(3.0)
    half = mx.array(0.5)
    for _ in range(iterations):
        y = half * y * (three - xe * y * y)
    return y

def eml_rms_norm(x, weight, eps=1e-8):
    # Standard RMSNorm: weight * x / sqrt(mean(x^2) + eps)
    rms_sq = mx.mean(mx.square(x), axis=-1, keepdims=True)
    # Using EML rsqrt (Newton-Schulz)
    return weight * x * eml_rsqrt_ns(rms_sq, eps=eps)

def eml_silu(x):
    # SiLU(x) = x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + eml_exp(-x))
    return x * sig

# ---------------------------------------------------------------------------
# EML-native Grokking Architecture
# ---------------------------------------------------------------------------

class EMLAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Affine params for RMSNorm
        self.weight = mx.ones((dim,))

        self.wq = nn.Linear(dim, inner_dim, bias=False)
        self.wk = nn.Linear(dim, inner_dim, bias=False)
        self.wv = nn.Linear(dim, inner_dim, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)
        self.rope = nn.RoPE(dim_head, traditional=True, base=1e6)

    def __call__(self, x, mask=None):
        b, n, d = x.shape
        # EML RMSNorm
        x = eml_rms_norm(x, self.weight)

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        reshaper = (lambda x: x.reshape(
            b, n, self.heads, -1).transpose(0, 2, 1, 3))
        queries, keys, values = map(reshaper, (queries, keys, values))

        queries = self.rope(queries)
        keys = self.rope(keys)

        # Min-Plus Stable Attention
        # 1. Calculate logits
        # mx.matmul for simplicity, but functionally equivalent to EML trees
        logits = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            logits += mask

        # 2. Log-Sum-Exp subtraction instead of Softmax division
        lse = mx.logsumexp(logits, axis=-1, keepdims=True)
        # 3. exp(logits - lse) is the stable weights
        weights = mx.exp(logits - lse)

        output = (weights @ values).transpose(0, 2, 1, 3).reshape(b, n, -1)

        return self.wo(output)

class EMLFeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.w1 = nn.Linear(dim, mlp_dim, bias=False)
        self.w2 = nn.Linear(mlp_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, mlp_dim, bias=False)

    def __call__(self, x) -> mx.array:
        x = eml_rms_norm(x, self.weight)
        # EML SiLU
        return self.w2(eml_silu(self.w1(x)) * self.w3(x))

class EMLBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, seq_len, dropout):
        super().__init__()
        self.attn = EMLAttention(dim, heads, dim_head, dropout)
        self.ff = EMLFeedForward(dim, mlp_dim, dropout)
        self._mask = self._causal_mask(seq_len)

    def _causal_mask(self, n: int) -> mx.array:
        mask = mx.triu(mx.full(
            (n, n), -float('inf'), dtype=mx.float32
        ), k=1)
        return mask

    def __call__(self, x):
        x = x + self.attn(x, mask=self._mask)
        return x + self.ff(x)

class EMLGrokTransformer(nn.Module):
    def __init__(self, depth, dim, heads, n_tokens, seq_len, dropout=0., pool='cls'):
        super().__init__()
        self.pool = pool
        self.embedding = nn.Embedding(n_tokens, dim)
        self.layers = nn.Sequential(*[
            EMLBlock(dim, heads, dim//heads, dim*4, seq_len, dropout)
            for _ in range(depth)
        ])
        self.final_weight = mx.ones((dim,))
        self.out = nn.Linear(dim, n_tokens, bias=False)

    def __call__(self, x):
        x = self.layers(self.embedding(x))
        x = mx.mean(x, axis=1) if self.pool == 'mean' else x[:, -1]
        return self.out(eml_rms_norm(x, self.final_weight))
