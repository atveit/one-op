"""Auto-generated LaTeX for Table 1 of the paper.

For every row of ``ComponentCoverage.md`` with a concrete arithmetic EML
construction, render the SymPy expression to LaTeX. Rows with status ``N/A``
or ``hard`` are skipped with a note.

Outputs:
- ``paper/table1_constructions.tex`` -- one ``\\textbf{Primitive}`` entry per
  row, suitable for \\input{} into the paper.
- ``paper/table1_constructions.md`` -- mirror in Markdown for review.
"""
from __future__ import annotations

from pathlib import Path

import sympy as sp


# ---------------------------------------------------------------------------
# Row definitions
# ---------------------------------------------------------------------------

# Common symbols
x, y, z, t, p, q = sp.symbols("x y z t p q", real=True)
beta, eta, eps, gamma_, beta_n = sp.symbols(
    "beta eta epsilon gamma beta_n", real=True
)
beta1, beta2 = sp.symbols("beta_1 beta_2", positive=True)
mu, sigma2 = sp.symbols("mu sigma^2", real=True)
n = sp.Symbol("n", integer=True, positive=True)
d_model = sp.Symbol("d", positive=True)
theta, g, m_prev, v_prev = sp.symbols("theta g m_prev v_prev", real=True)
tau = sp.Symbol("tau", positive=True)
# IndexedBases for vector-valued rows
v_vec = sp.IndexedBase("v")
Q = sp.IndexedBase("Q")
K = sp.IndexedBase("K")
V = sp.IndexedBase("V")
W_g = sp.IndexedBase("W_g")
W_v = sp.IndexedBase("W_v")
W_o = sp.IndexedBase("W_o")
W_r = sp.IndexedBase("W_r")
i_idx, j_idx, k_idx = sp.symbols("i j k", integer=True, nonnegative=True)

# Small helpers
_exp = sp.exp
_log = sp.log


def _sigmoid(a):
    return 1 / (1 + _exp(-a))


def _softplus(a):
    return _log(1 + _exp(a))


def _softmax_sum():
    return sp.Sum(_exp(v_vec[j_idx]), (j_idx, 0, n - 1))


def _lse():
    return _log(_softmax_sum())


def _attention_row():
    # softmax(Q K^T / sqrt(d)) V   -- per output index i
    s = sp.IndexedBase("s")  # s_j = Q_i . K_j / sqrt(d)
    soft_sum = sp.Sum(_exp(s[j_idx]), (j_idx, 0, n - 1))
    return sp.Sum(_exp(s[j_idx]) / soft_sum * V[j_idx], (j_idx, 0, n - 1))


# Each row: (section, name, expr_or_text, status, lean_thm, skip_reason)
# ``expr_or_text`` is either a SymPy expression (rendered via ``sp.latex``)
# or a tuple ``("text", "raw LaTeX string")``.

def _rows():
    R = []

    # --- Activations ---
    R.append(("Activations", "Sigmoid", _sigmoid(x), "done", "sigmoid_from_eml", None))
    R.append(("Activations", "Tanh", (_exp(x) - _exp(-x)) / (_exp(x) + _exp(-x)), "done", "tanh_from_exp", None))
    R.append(("Activations", "ReLU (exact)", sp.Rational(1, 2) * (x + sp.sqrt(x ** 2)), "done", "relu_exact_from_sqrt", None))
    R.append(("Activations", "ReLU (smooth)", _softplus(beta * x) / beta, "done", "relu_smooth_def", None))
    R.append(("Activations", "Softplus", _softplus(x), "done", "softplus_from_eml", None))
    R.append(("Activations", "GELU (tanh approx)",
              sp.Rational(1, 2) * x * (1 + sp.tanh(sp.sqrt(2 / sp.pi) * (x + sp.Rational(4471, 100000) * x ** 3))),
              "done", "gelu_def", None))
    R.append(("Activations", "GELU (erf exact)",
              sp.Rational(1, 2) * x * (1 + sp.erf(x / sp.sqrt(2))),
              "medium", "gelu_erf", None))
    R.append(("Activations", "SiLU / Swish", x * _sigmoid(x), "trivial", "silu_def", None))
    R.append(("Activations", "Mish", x * sp.tanh(_softplus(x)), "trivial", "mish_def", None))
    R.append(("Activations", "SwiGLU",
              ("text", r"\mathrm{SiLU}(x W_g) \odot (x W_v) W_o"),
              "trivial", "swiglu_def", None))
    R.append(("Activations", "GeGLU",
              ("text", r"\mathrm{GELU}(x W_g) \odot (x W_v) W_o"),
              "trivial", "geglu_def", None))
    R.append(("Activations", "Softmax",
              _exp(v_vec[i_idx]) / _softmax_sum(), "done",
              "softmax_eq_exp_div_sum", None))
    R.append(("Activations", "Log-softmax", v_vec[i_idx] - _lse(),
              "done", "log_softmax_def", None))

    # --- Normalization ---
    R.append(("Normalization", "LayerNorm",
              gamma_ * (x - mu) / sp.sqrt(sigma2 + eps) + beta_n,
              "done", "layer_norm_def", None))
    R.append(("Normalization", "RMSNorm",
              gamma_ * x / sp.sqrt(sigma2 + eps), "done", "rms_norm_def", None))
    R.append(("Normalization", "GroupNorm",
              gamma_ * (x - mu) / sp.sqrt(sigma2 + eps) + beta_n,
              "trivial", "group_norm_def", None))
    R.append(("Normalization", "InstanceNorm",
              gamma_ * (x - mu) / sp.sqrt(sigma2 + eps) + beta_n,
              "trivial", "instance_norm_def", None))
    R.append(("Normalization", "BatchNorm (train)",
              gamma_ * (x - mu) / sp.sqrt(sigma2 + eps) + beta_n,
              "trivial", "batchnorm_def", None))
    R.append(("Normalization", "BatchNorm (inference)",
              gamma_ * (x - mu) / sp.sqrt(sigma2 + eps) + beta_n,
              "trivial", "batchnorm_inference_def", None))

    # --- Linear / embedding ---
    R.append(("Linear", "Linear (Dense)",
              ("text", r"x W^{\top} + b"),
              "done", "matmul_assoc", None))
    R.append(("Linear", "Token Embedding", None, "N/A", None, "data movement"))
    R.append(("Linear", "Positional Embedding (learned)", None, "N/A", None, "data movement + add"))
    R.append(("Linear", "Positional Embedding (sinusoidal)",
              sp.sin(k_idx / 10000 ** (2 * i_idx / d_model)),
              "medium", "sin_cos_from_complex_exp", None))
    R.append(("Linear", "RoPE",
              ("text", r"\mathrm{rot}(\theta)\,(x_{2i},\,x_{2i+1})"),
              "medium", "rope_def", None))
    R.append(("Linear", "ALiBi", ("text", r"s_{ij} + m \cdot (j - i)"),
              "trivial", None, None))
    R.append(("Linear", "Layer Scale", gamma_ * x, "trivial", None, None))
    R.append(("Linear", "Patch Embedding",
              ("text", r"\mathrm{Conv2D}_{k=16,\,s=16}(x)"),
              "done", "patch_embedding_def", None))

    # --- Convolutions ---
    R.append(("Convolutions", "Conv1D",
              ("text", r"\mathrm{im2col}(x) \cdot W^{\top}"),
              "trivial", None, None))
    R.append(("Convolutions", "Conv2D",
              ("text", r"\mathrm{im2col}(x) \cdot W^{\top}"),
              "done", "conv2d_eq_matmul", None))
    R.append(("Convolutions", "Conv3D",
              ("text", r"\mathrm{im2col}(x) \cdot W^{\top}"),
              "trivial", None, None))
    R.append(("Convolutions", "ConvTranspose",
              ("text", r"\mathrm{Conv}(\mathrm{upsample}(x))"),
              "trivial", None, None))
    R.append(("Convolutions", "DepthwiseConv",
              ("text", r"\mathrm{Conv}_{\mathrm{per\text{-}channel}}(x)"),
              "trivial", None, None))
    R.append(("Convolutions", "MaxPool", None, "N/A", None, "discrete reduction"))
    R.append(("Convolutions", "AvgPool",
              ("text", r"\tfrac{1}{k} \sum_{i} x_i"),
              "trivial", None, None))

    # --- Attention family ---
    R.append(("Attention", "Multi-Head Attention",
              ("text", r"\mathrm{softmax}\!\left(\tfrac{Q K^{\top}}{\sqrt{d}}\right) V"),
              "done", "attention_from_eml", None))
    R.append(("Attention", "Log-domain attention",
              ("text", r"\sum_j \exp\!\left(s_{ij} - \mathrm{LSE}_j(s_i)\right) V_j"),
              "in progress", "log_domain_attention_eq_attention", None))
    R.append(("Attention", "Flash Attention",
              ("text", r"\text{tiled LSE + online softmax}"),
              "trivial", "flash_eq_log_domain_attention", None))
    R.append(("Attention", "Grouped-Query",
              ("text", r"\mathrm{MHA}\; \text{with fewer KV heads}"),
              "trivial", None, None))
    R.append(("Attention", "Multi-Query",
              ("text", r"\mathrm{GQA}\; \text{with 1 KV head}"),
              "trivial", None, None))
    R.append(("Attention", "Sliding-Window Attn",
              ("text", r"\mathrm{MHA} \; + \; \text{banded mask}"),
              "trivial", None, None))
    R.append(("Attention", "Cross-Attention",
              ("text", r"\mathrm{MHA}(Q_x, K_y, V_y)"),
              "trivial", None, None))
    R.append(("Attention", "Causal Mask",
              ("text", r"s_{ij} + M_{ij},\ \ M_{ij} = -\infty \text{ if } j > i"),
              "trivial", None, None))
    R.append(("Attention", "MLA (DeepSeek)",
              ("text", r"\mathrm{MHA}(Q,\; K_{\mathrm{latent}},\; V_{\mathrm{latent}})"),
              "trivial", None, None))
    R.append(("Attention", "Deformable Attention", None, "hard", None, "learned offset gather"))
    R.append(("Attention", "Shifted-Window (Swin)",
              ("text", r"\mathrm{MHA}_{\mathrm{window}}(x^{\mathrm{shift}})"),
              "trivial", None, None))
    R.append(("Attention", "Ragged Attention",
              ("text", r"\mathrm{MHA}(\text{packed seq})"),
              "trivial", None, None))
    R.append(("Attention", "Linear/Performer",
              ("text", r"\phi(Q) \,(\phi(K)^{\top} V)"),
              "trivial", None, None))
    R.append(("Attention", "Max-Plus attention",
              ("text", r"V_{\arg\max_j (Q K^{\top}/\sqrt{d})_{ij}}"),
              "medium", "max_plus_attention_def", None))

    # --- FFN ---
    R.append(("FFN", "Vanilla FFN",
              ("text", r"\mathrm{Linear}_2(\mathrm{GELU}(\mathrm{Linear}_1(x)))"),
              "trivial", None, None))
    R.append(("FFN", "SwiGLU FFN",
              ("text", r"(\mathrm{SiLU}(x W_g) \odot (x W_v)) W_o"),
              "trivial", None, None))
    R.append(("FFN", "GeGLU FFN",
              ("text", r"(\mathrm{GELU}(x W_g) \odot (x W_v)) W_o"),
              "trivial", None, None))

    # --- Mixture of Experts ---
    R.append(("MoE", "Top-K Router", None, "hard", None, "discrete top-k"))
    R.append(("MoE", "Expert FFN",
              ("text", r"\mathrm{SwiGLU}(x)\;\text{per expert}"),
              "trivial", None, None))
    R.append(("MoE", "Load-Balancing Aux Loss",
              ("text", r"\alpha \cdot n \sum_e f_e p_e"),
              "trivial", None, None))
    R.append(("MoE", "Token Dispatch", None, "N/A", None, "data movement"))
    R.append(("MoE", "Capacity Factor", None, "hard", None, "compare/truncate"))

    # --- Regularization ---
    R.append(("Regularization", "Dropout",
              ("text", r"x \cdot \mathrm{Bern}(1-p) / (1-p)"),
              "trivial", None, None))
    R.append(("Regularization", "DropPath",
              ("text", r"x \cdot \mathrm{Bern}(1-p)"),
              "trivial", None, None))
    R.append(("Regularization", "L1 Penalty",
              sp.Sum(sp.sqrt(theta ** 2), (i_idx, 0, n - 1)),
              "trivial", None, None))
    R.append(("Regularization", "L2 Penalty",
              sp.Sum(theta ** 2, (i_idx, 0, n - 1)),
              "trivial", None, None))
    R.append(("Regularization", "Weight Decay",
              theta - eta * sp.Symbol("lambda") * theta,
              "trivial", None, None))

    # --- Losses ---
    R.append(("Losses", "MSE", (x - y) ** 2, "trivial", None, None))
    R.append(("Losses", "L1", sp.sqrt((x - y) ** 2), "trivial", None, None))
    R.append(("Losses", "Huber",
              ("text", r"\tfrac{1}{2}(x - y)^2 \; \text{or}\; \delta(\lvert x - y\rvert - \tfrac{\delta}{2})"),
              "trivial", None, None))
    R.append(("Losses", "BCE",
              -(y * _log(p) + (1 - y) * _log(1 - p)),
              "trivial", None, None))
    R.append(("Losses", "Cross-Entropy",
              _lse() - v_vec[i_idx], "done", "ce_eq_lse_minus_target_logit", None))
    R.append(("Losses", "KL Divergence",
              sp.Sum(p * _log(p / q), (i_idx, 0, n - 1)),
              "trivial", None, None))
    R.append(("Losses", "Focal Loss",
              -(1 - p) ** sp.Symbol("gamma_f") * _log(p),
              "trivial", None, None))
    R.append(("Losses", "InfoNCE / NT-Xent",
              -_log(_exp(sp.Symbol("s_+") / tau) /
                    sp.Sum(_exp(sp.Symbol("s_j") / tau), (j_idx, 0, n - 1))),
              "trivial", None, None))
    R.append(("Losses", "Triplet Loss",
              sp.Rational(1, 2) * (sp.Symbol("d_p") - sp.Symbol("d_n") + sp.Symbol("m")
                                   + sp.sqrt((sp.Symbol("d_p") - sp.Symbol("d_n") + sp.Symbol("m")) ** 2)),
              "trivial", None, None))
    R.append(("Losses", "SigLIP Loss",
              _log(1 + _exp(-y * (sp.Symbol("s") / tau + sp.Symbol("b")))),
              "trivial", None, None))

    # --- Optimizers ---
    R.append(("Optimizers", "SGD", theta - eta * g, "trivial", None, None))
    R.append(("Optimizers", "SGD + Momentum",
              theta - eta * (sp.Symbol("mu") * v_prev + g),
              "trivial", None, None))
    R.append(("Optimizers", "Nesterov",
              theta - eta * (sp.Symbol("mu") * (sp.Symbol("mu") * v_prev + g) + g),
              "trivial", None, None))
    R.append(("Optimizers", "Adam",
              ("text",
               r"\theta \leftarrow \theta - \eta \cdot \frac{m / (1 - e^{t \ln \beta_1})}"
               r"{\sqrt{v / (1 - e^{t \ln \beta_2})} + \epsilon}"),
              "medium", "adam_update_def", None))
    R.append(("Optimizers", "AdamW",
              ("text", r"\text{Adam step} \; - \; \eta \lambda \theta"),
              "trivial", None, None))
    R.append(("Optimizers", "RMSProp",
              theta - eta * g / sp.sqrt(sp.Symbol("v") + eps),
              "trivial", None, None))
    R.append(("Optimizers", "Adagrad",
              theta - eta * g / sp.sqrt(sp.Symbol("G") + eps),
              "trivial", None, None))
    R.append(("Optimizers", "Lion",
              theta - eta * sp.sqrt(g ** 2) / g,
              "trivial", None, None))
    R.append(("Optimizers", "AdaBelief",
              ("text", r"\theta - \eta m / (\sqrt{s} + \epsilon)"),
              "trivial", None, None))
    R.append(("Optimizers", "Muon",
              ("text", r"\theta - \eta \, \mathrm{NewtonSchulz}(m)"),
              "medium", "newton_schulz_eq_polar_factor", None))
    R.append(("Optimizers", "Shampoo",
              ("text", r"\theta - \eta L^{-1/4} G R^{-1/4}"),
              "trivial", None, None))

    # --- Schedulers ---
    R.append(("Schedulers", "Constant", sp.Symbol("lr"), "trivial", None, None))
    R.append(("Schedulers", "Linear warmup",
              sp.Symbol("lr") * t / sp.Symbol("T_w"),
              "trivial", None, None))
    R.append(("Schedulers", "Exponential decay",
              sp.Symbol("lr_0") * _exp(-sp.Symbol("k") * t),
              "trivial", None, None))
    R.append(("Schedulers", "Polynomial decay",
              sp.Symbol("lr_0") * (1 - t / sp.Symbol("T")) ** sp.Symbol("p_s"),
              "trivial", None, None))
    R.append(("Schedulers", "Cosine schedule",
              sp.Symbol("lr_0") * sp.Rational(1, 2)
              * (1 + sp.cos(sp.pi * t / sp.Symbol("T"))),
              "medium", None, None))
    R.append(("Schedulers", "Cosine with warmup",
              ("text",
               r"\mathrm{piecewise}\{\mathrm{linear\ warmup},\ \mathrm{cosine}\}"),
              "medium", None, None))
    R.append(("Schedulers", "1cycle",
              ("text", r"\mathrm{piecewise\ super\text{-}convergence}"),
              "trivial", None, None))

    # --- Training meta ---
    R.append(("Training", "Gradient clipping (norm)",
              g * sp.Min(1, tau / sp.sqrt(sp.Symbol("\\lVert g \\rVert^2"))),
              "trivial", None, None))
    R.append(("Training", "Gradient clipping (value)",
              ("text", r"\mathrm{clip}(g, -c, c)"),
              "trivial", None, None))
    R.append(("Training", "EMA of weights",
              sp.Symbol("alpha") * sp.Symbol("theta_ema") + (1 - sp.Symbol("alpha")) * theta,
              "trivial", None, None))
    R.append(("Training", "Checkpoint", None, "N/A", None, "pure I/O"))

    # --- Vision extras ---
    R.append(("Vision", "Class Token", None, "N/A", None, "concat"))
    R.append(("Vision", "2D Sinusoidal PE",
              ("text", r"\sin(h / 10000^{2i/d}),\; \cos(w / 10000^{2i/d})"),
              "medium", None, None))
    R.append(("Vision", "Interpolated PE",
              ("text", r"\mathrm{bilinear}(PE_{\mathrm{table}})"),
              "trivial", None, None))
    R.append(("Vision", "Image standardization",
              (x - mu) / sp.Symbol("sigma"), "trivial", None, None))
    R.append(("Vision", "CutMix / MixUp",
              sp.Symbol("lambda_mix") * x + (1 - sp.Symbol("lambda_mix")) * y,
              "trivial", None, None))

    # --- Quantization ---
    R.append(("Quantization", "Fake-quant (forward)", None, "hard", None, "discrete round"))
    R.append(("Quantization", "Straight-through estimator",
              ("text", r"\partial / \partial x = \mathbb{1}"),
              "trivial", None, None))
    R.append(("Quantization", "bf16/fp16/fp8 storage", None, "N/A", None, "cast"))

    return R


def _render_row(row):
    section, name, expr, status, thm, skip_reason = row
    if status in ("N/A", "hard"):
        return None  # skip
    if expr is None:
        return None
    if isinstance(expr, tuple) and expr[0] == "text":
        latex_str = expr[1]
    else:
        latex_str = sp.latex(expr)
    return (section, name, latex_str, status, thm)


def main():
    out_tex = Path("paper/table1_constructions.tex")
    out_md = Path("paper/table1_constructions.md")
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    rows = _rows()
    rendered = [r for r in (_render_row(r) for r in rows) if r is not None]

    skipped = [r for r in rows if r[3] in ("N/A", "hard")]

    tex_lines = [
        "% Auto-generated by eml_nn/sympy/latex_table.py",
        "% Each entry: \\textbf{Primitive} \\quad $latex expression$",
        "",
        "\\begin{description}",
    ]
    for section, name, latex_str, status, thm in rendered:
        tex_lines.append(
            f"  \\item[\\textbf{{{name}}}] ${latex_str}$"
            + (f" \\hfill\\textit{{{thm}}}" if thm else "")
        )
    tex_lines.append("\\end{description}")
    out_tex.write_text("\n".join(tex_lines) + "\n")

    md_lines = [
        "# Table 1 constructions (auto-generated)",
        "",
        f"Generated rows: {len(rendered)}. Skipped (N/A or hard): {len(skipped)}.",
        "",
        "| Section | Primitive | LaTeX | Status | Lean theorem |",
        "|---------|-----------|-------|--------|--------------|",
    ]
    for section, name, latex_str, status, thm in rendered:
        thm_s = thm if thm else "-"
        md_lines.append(
            f"| {section} | {name} | `${latex_str}$` | {status} | {thm_s} |"
        )
    md_lines.append("")
    md_lines.append("## Skipped rows")
    md_lines.append("")
    for section, name, _, status, _, reason in skipped:
        md_lines.append(f"- **{name}** ({section}, status `{status}`): {reason or '—'}")
    out_md.write_text("\n".join(md_lines) + "\n")

    print(f"[latex_table] wrote {out_tex} ({len(rendered)} rows, {len(skipped)} skipped)")
    print(f"[latex_table] wrote {out_md}")


if __name__ == "__main__":
    main()
