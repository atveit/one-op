import argparse
import numpy as np
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from functools import partial

# Import both models
from models_eml import EMLGrokTransformer
from reference.models import Transformer as StandardTransformer
from reference.data import grokking_data

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--p', type=int, default=97)
parser.add_argument('--train-fraction', type=float, default=0.5)
args = parser.parse_args()

class Trainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        warmup = optim.linear_schedule(0, lr, 10)
        self.optimizer = optim.AdamW(learning_rate=warmup, weight_decay=1.0)
        self.loss_fn = nn.losses.cross_entropy
        
        self.train_acc = []
        self.val_acc = []

    def train_epoch(self, train_data, val_data, batch_size=512):
        X, T = train_data
        inds = mx.array(np.random.permutation(X.shape[0]))
        X, T = X[inds], T[inds]
        
        total_correct = 0
        for i in range(0, X.shape[0], batch_size):
            xb, tb = X[i:i+batch_size], T[i:i+batch_size]
            
            def loss_fn(model):
                y = model(xb)
                return mx.mean(self.loss_fn(y, tb))
            
            loss, grads = nn.value_and_grad(self.model, loss_fn)(self.model)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.state, self.optimizer.state)
            
            y = self.model(xb)
            total_correct += mx.sum(mx.argmax(y, axis=1) == tb).item()
            
        self.train_acc.append(total_correct / X.shape[0])
        
        # Eval
        Xv, Tv = val_data
        yv = self.model(Xv)
        val_correct = mx.sum(mx.argmax(yv, axis=1) == Tv).item()
        self.val_acc.append(val_correct / Xv.shape[0])

def run_experiment():
    mx.random.seed(42)
    np.random.seed(42)
    
    Xtrain, Ttrain, Xtest, Ttest = grokking_data(args.p, op='/', train_fraction=args.train_fraction)
    
    kwargs = {'depth': 2, 'dim': 128, 'heads': 1, 'n_tokens': args.p + 2, 'seq_len': Xtrain.shape[1], 'dropout': 0.2}
    
    print("Training Standard MLX Baseline...")
    std_model = StandardTransformer(**kwargs)
    std_trainer = Trainer(std_model)
    for _ in tqdm(range(args.epochs)):
        std_trainer.train_epoch((Xtrain, Ttrain), (Xtest, Ttest))
        
    print("Training EML-native Model...")
    eml_model = EMLGrokTransformer(**kwargs)
    eml_trainer = Trainer(eml_model)
    for _ in tqdm(range(args.epochs)):
        eml_trainer.train_epoch((Xtrain, Ttrain), (Xtest, Ttest))
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(std_trainer.train_acc)*100, color='#1b9e77', alpha=0.3, label='Standard Train')
    plt.plot(np.array(std_trainer.val_acc)*100, color='#1b9e77', label='Standard Val (Baseline)')
    
    plt.plot(np.array(eml_trainer.train_acc)*100, color='#d95f02', alpha=0.3, label='EML Train')
    plt.plot(np.array(eml_trainer.val_acc)*100, color='#d95f02', linewidth=2, label='EML Val (This work)')
    
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Grokking Phase Transition: Standard MLX vs. EML-native (p={args.p})')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('grokking_comparison.png', dpi=300)
    print("Saved comparison plot to grokking_comparison.png")

if __name__ == '__main__':
    run_experiment()
