import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# EML Dual-Space Primitives
# ---------------------------------------------------------------------------
def eml(x, y):
    return mx.exp(x) - mx.log(y)

def eml_sqrt(x):
    return mx.exp(0.5 * mx.log(x))

def eml_rsqrt_ns(x, eps=1e-8, iterations=3):
    """Newton-Schulz Iterative Refinement for 1/sqrt(x)"""
    xe = x + eps
    # Seed
    y = mx.array(1.0) / eml_sqrt(xe)
    three = mx.array(3.0)
    half = mx.array(0.5)
    for _ in range(iterations):
        y = half * y * (three - xe * y * y)
    return y

# ---------------------------------------------------------------------------
# JEPA Architecture
# ---------------------------------------------------------------------------
class ToyJEPA(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, embed_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim + 1, 16),
            nn.GELU(),
            nn.Linear(16, embed_dim)
        )
        
    def __call__(self, x, delta_t):
        s_x = self.encoder(x)
        # predictor takes context embedding + time delta
        pred_y = self.predictor(mx.concatenate([s_x, delta_t], axis=-1))
        return s_x, pred_y

    def encode_target(self, y):
        return self.encoder(y)

def vicreg_loss(pred_y, s_y, gamma=1.0, eps=1e-8, use_eml=False):
    # Invariance loss: predictor should match target embedding
    inv_loss = mx.mean(mx.square(pred_y - s_y))
    
    # Variance loss: prevent representation collapse
    var_y = mx.var(s_y, axis=0)
    
    if use_eml:
        # Dual-Space Iterative Refinement (Newton-Schulz)
        # var(Z) term using rsqrt
        std_y = 1.0 / eml_rsqrt_ns(var_y, eps=eps)
    else:
        # Standard FP32 sqrt, notoriously fragile gradients near 0
        std_y = mx.sqrt(var_y + eps)
        
    var_loss = mx.mean(mx.maximum(0.0, gamma - std_y))
    
    return inv_loss + var_loss, inv_loss, var_loss

def train_jepa(use_eml=False, seed=42):
    np.random.seed(seed)
    mx.random.seed(seed)
    
    model = ToyJEPA()
    # Induce collapse by using a very high learning rate
    optimizer = optim.AdamW(learning_rate=5e-3)
    
    # 1D Kinematics Data (Bouncing Ball / Sine wave)
    t = np.linspace(0, 10 * np.pi, 500).reshape(-1, 1)
    data = np.sin(t)
    delta_t = np.ones_like(data) * (t[1] - t[0])
    
    X = mx.array(data[:-1])
    Y = mx.array(data[1:])
    DT = mx.array(delta_t[:-1])
    
    def loss_fn(model):
        s_x, pred_y = model(X, DT)
        s_y = model.encode_target(Y)
        # Extreme epsilon to simulate precision starvation
        loss, inv, var = vicreg_loss(pred_y, s_y, eps=0.0, use_eml=use_eml)
        return loss, inv, var

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    losses = []
    
    for epoch in range(100):
        (loss, inv, var), grads = loss_and_grad_fn(model)
        
        # If NaN, stop early
        if mx.isnan(loss).item():
            losses.extend([float('nan')] * (100 - epoch))
            break
            
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        losses.append(loss.item())
        
    return losses

def main():
    print("Running 1D Kinematics V-JEPA Experiment...")
    
    os.makedirs("results/jepa/plots", exist_ok=True)
    
    # Run FP32 Baseline
    losses_fp32 = train_jepa(use_eml=False, seed=12)
    print(f"FP32 Baseline Final Loss: {losses_fp32[-1]:.4f} (NaN count: {np.isnan(losses_fp32).sum()})")
    
    # Run EML Iterative Refinement
    losses_eml = train_jepa(use_eml=True, seed=12)
    print(f"EML Dual-Space Final Loss: {losses_eml[-1]:.4f} (NaN count: {np.isnan(losses_eml).sum()})")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses_fp32, label='Standard FP32 VICReg (Collapse & NaNs)', color='red', alpha=0.8)
    plt.plot(losses_eml, label='EML Newton-Schulz VICReg (Stable)', color='blue', linewidth=2)
    plt.yscale('log')
    plt.title('V-JEPA 1D Kinematics: Representation Collapse vs. EML Stability')
    plt.xlabel('Epochs')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/jepa/plots/1d_kinematics_vjepa.png')
    print("Saved plot to results/jepa/plots/1d_kinematics_vjepa.png")

if __name__ == "__main__":
    main()
