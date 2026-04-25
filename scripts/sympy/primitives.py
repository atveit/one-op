"""Symbolic EML primitives mirroring ``eml_nn/arith.py``.

The base operator is ``eml(x, y) = exp(x) - log(y)``. Every derived primitive
is expressed as nested ``exp(_) - log(_)`` trees and then simplified via
SymPy. These expressions are *symbolic*, not numerical -- they are used to
feed ``tree_search.py``, ``latex_table.py``, and
``derivative_supplement.py``.
"""
from __future__ import annotations

import sympy as sp

# Symbols used across the module. All primitives are expressed in terms of
# these (or the integer constant 1 via ``sp.Integer(1)``).
x, y, z = sp.symbols("x y z", real=True)


def eml(a, b):
    """Symbolic ``eml(a, b) = exp(a) - log(b)``.

    ``b`` must be positive where ``log`` is applied; SymPy does not enforce
    this, we simply note it (Lean is the final arbiter).
    """
    return sp.exp(a) - sp.log(b)


# ---------------------------------------------------------------------------
# Stage 2: arithmetic primitives
# ---------------------------------------------------------------------------

def eml_exp_sympy(a):
    """exp(a) = eml(a, 1). Depth 1."""
    return eml(a, sp.Integer(1))


def eml_ln_sympy(a):
    """ln(a) = eml(1, eml(eml(1, a), 1)). Depth 3 (paper eq. 5)."""
    return eml(sp.Integer(1), eml(eml(sp.Integer(1), a), sp.Integer(1)))


def e_const_sympy():
    """Euler's e = eml(1, 1). Depth 1."""
    return eml(sp.Integer(1), sp.Integer(1))


def eml_neg_ln_sympy(a):
    """-ln(a) = eml(0, e*a) = 1 - ln(e*a)."""
    return eml(sp.Integer(0), e_const_sympy() * a)


def eml_recip_sympy(a):
    """1/a = exp(-ln a) = eml(-ln a, 1)."""
    return eml(eml_neg_ln_sympy(a), sp.Integer(1))


def eml_neg_sympy(a):
    """-a = ln(1/exp(a))."""
    return eml_ln_sympy(eml_recip_sympy(eml_exp_sympy(a)))


def eml_mul_sympy(a, b):
    """a*b for a>0, b>0 = exp(ln a + ln b)."""
    return eml_exp_sympy(eml_ln_sympy(a) + eml_ln_sympy(b))


def eml_div_sympy(a, b):
    """a/b for a>0, b>0 = exp(ln a - ln b)."""
    return eml_exp_sympy(eml_ln_sympy(a) - eml_ln_sympy(b))


def eml_add_sympy(a, b):
    """a+b = ln(exp(a)*exp(b))."""
    return eml_ln_sympy(eml_exp_sympy(a) * eml_exp_sympy(b))


def eml_sub_sympy(a, b):
    """a-b = a + (-b)."""
    return eml_add_sympy(a, eml_neg_sympy(b))


def eml_sqrt_sympy(a):
    """sqrt(a) = exp((1/2) ln a)."""
    return eml_exp_sympy(sp.Rational(1, 2) * eml_ln_sympy(a))


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def eml_sigmoid_sympy(a):
    """sigmoid(a) = 1/(1+exp(-a))."""
    return 1 / (1 + sp.exp(-a))


def eml_tanh_sympy(a):
    """tanh(a) = (exp(a)-exp(-a)) / (exp(a)+exp(-a))."""
    return (sp.exp(a) - sp.exp(-a)) / (sp.exp(a) + sp.exp(-a))


def eml_softplus_sympy(a):
    """softplus(a) = log(1+exp(a))."""
    return sp.log(1 + sp.exp(a))


def eml_relu_exact_sympy(a):
    """ReLU via 1/2 (x + sqrt(x^2))."""
    return sp.Rational(1, 2) * (a + sp.sqrt(a ** 2))


def eml_relu_smooth_sympy(a, beta=sp.Symbol("beta", positive=True)):
    """Smooth ReLU: softplus(beta x) / beta."""
    return eml_softplus_sympy(beta * a) / beta


def eml_silu_sympy(a):
    """SiLU / Swish: x * sigmoid(x)."""
    return a * eml_sigmoid_sympy(a)


def eml_mish_sympy(a):
    """Mish: x * tanh(softplus(x))."""
    return a * sp.tanh(eml_softplus_sympy(a))


def eml_gelu_tanh_sympy(a):
    """GELU (tanh approximation)."""
    c = sp.sqrt(2 / sp.pi)
    return sp.Rational(1, 2) * a * (1 + sp.tanh(c * (a + sp.Rational(4471, 100000) * a ** 3)))


def eml_gelu_erf_sympy(a):
    """GELU (erf exact)."""
    return sp.Rational(1, 2) * a * (1 + sp.erf(a / sp.sqrt(2)))


def eml_softmax_sympy(v, i, n):
    """softmax(v)_i = exp(v_i) / sum_j exp(v_j).

    ``v`` should be an ``IndexedBase``; ``i`` and ``n`` are summation indices.
    """
    j = sp.symbols("j", integer=True)
    return sp.exp(v[i]) / sp.Sum(sp.exp(v[j]), (j, 0, n - 1))


def eml_log_softmax_sympy(v, i, n):
    """log-softmax(v)_i = v_i - LSE(v)."""
    j = sp.symbols("j", integer=True)
    return v[i] - sp.log(sp.Sum(sp.exp(v[j]), (j, 0, n - 1)))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def eml_layer_norm_sympy(x_i, mu, sigma_sq, eps, gamma, beta):
    """LayerNorm: gamma * (x - mu) / sqrt(sigma^2 + eps) + beta."""
    return gamma * (x_i - mu) / sp.sqrt(sigma_sq + eps) + beta


def eml_rms_norm_sympy(x_i, mean_sq, eps, gamma):
    """RMSNorm: gamma * x / sqrt(mean(x^2) + eps)."""
    return gamma * x_i / sp.sqrt(mean_sq + eps)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def eml_adam_step_sympy():
    """Single-parameter Adam update. Returns the symbolic updated theta and
    helper quantities as a dict, for use by ``derivative_supplement``.
    """
    theta, g, m_prev, v_prev = sp.symbols("theta g m_prev v_prev", real=True)
    beta1, beta2, eps, eta, t = sp.symbols(
        "beta_1 beta_2 epsilon eta t", positive=True
    )
    m = beta1 * m_prev + (1 - beta1) * g
    v = beta2 * v_prev + (1 - beta2) * g ** 2
    # beta^t = exp(t log beta) -- thematic EML bow.
    m_hat = m / (1 - sp.exp(t * sp.log(beta1)))
    v_hat = v / (1 - sp.exp(t * sp.log(beta2)))
    theta_new = theta - eta * m_hat / (sp.sqrt(v_hat) + eps)
    return {
        "theta": theta,
        "g": g,
        "m_prev": m_prev,
        "v_prev": v_prev,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "eta": eta,
        "t": t,
        "m": m,
        "v": v,
        "m_hat": m_hat,
        "v_hat": v_hat,
        "theta_new": theta_new,
    }


# ---------------------------------------------------------------------------
# Log-domain attention weight (for the derivative supplement)
# ---------------------------------------------------------------------------

def eml_log_attention_weight_sympy(n=None):
    """Returns ``(log_w_i, w_i)`` for log-domain attention weight at index i.

    log_w_i = s_i - LSE(s),  w_i = exp(log_w_i).
    """
    n_sym = n if n is not None else sp.Symbol("n", integer=True, positive=True)
    s = sp.IndexedBase("s")
    i, j = sp.symbols("i j", integer=True, nonnegative=True)
    lse = sp.log(sp.Sum(sp.exp(s[j]), (j, 0, n_sym - 1)))
    log_w = s[i] - lse
    w = sp.exp(log_w)
    return log_w, w, s, i, j, n_sym
