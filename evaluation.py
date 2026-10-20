import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from GATr import GATrModel
import torch.nn as nn
import numpy as np
import random

data = torch.load('mp_data.pt', weights_only=False)

ys = torch.stack([d.y.reshape(-1) for d in data])  # shape: [N, n_tasks]
n_tasks = ys.shape[1]
y_mean = ys.mean(dim=0)
y_std = ys.std(dim=0)
y_std = torch.clamp(y_std, min=1e-8)  # avoid divide-by-zero for near-constant targets

for i in range(n_tasks):
    print(
        f"Target {i} | mean: {y_mean[i].item():.3f} | std: {y_std[i].item():.3f} | "
        f"min: {ys[:, i].min().item():.3f} | max: {ys[:, i].max().item():.3f}"
    )

for d in data:
    d.y = ((d.y.reshape(-1) - y_mean) / y_std).reshape(1, -1)

# save stats for later
torch.save({'mean': y_mean, 'std': y_std}, 'mp_y_stats.pt')

# --- shuffle and split ---
random.seed(42)
random.shuffle(data)
n          = len(data)
train_data = data[:int(n * 0.5)]
val_data   = data[int(n * 0.5):]
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# --- model output size matches dataset targets ---
model = GATrModel(mv_channels=4, scalar_channels=8, n_heads=1, n_layers=1, n_tasks=n_tasks)
state_dict = torch.load(
    "full_pretrained_model.pt",
    map_location="cpu"
)

model.load_state_dict(state_dict)

# 4. Set to evaluation mode (essential if using dropout or batch norm)
def evaluate(model, test_data, y_mean, y_std, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(test_data, batch_size=batch_size)

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.node_mvs, batch.x, batch.batch, edge_index=batch.edge_index)
            # collect one row per sample, shape (batch_size, n_tasks)
            preds.append(pred.cpu())
            targets.append(batch.y.reshape(pred.shape).cpu())

    preds   = torch.cat(preds)    # (N, n_tasks)
    targets = torch.cat(targets)  # (N, n_tasks)

    # denormalize to original units — same transform as plot_predictions
    preds   = preds   * y_std + y_mean
    targets = targets * y_std + y_mean

    mse  = nn.MSELoss()(preds, targets).item()
    mae  = nn.L1Loss()(preds, targets).item()
    rmse = mse ** 0.5

    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot).item()

    print(f"Test MSE:  {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R²:   {r2:.4f}")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

# usage
results = evaluate(model, data, y_mean, y_std)

import matplotlib.pyplot as plt


def plot_predictions(model, val_data, y_mean, y_std, device='cuda'):
    model.eval()
    loader = DataLoader(val_data, batch_size=64)
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.node_mvs, batch.x, batch.batch, batch.edge_index)
            preds.append(pred.cpu())
            targets.append(batch.y.reshape(pred.shape).cpu())

    # denormalize — keep as tensors so broadcasting works for any n_tasks
    preds   = (torch.cat(preds)   * y_std + y_mean).numpy()
    targets = (torch.cat(targets) * y_std + y_mean).numpy()

    mae = np.mean(np.abs(preds - targets))
    corr = np.corrcoef(preds, targets)[0, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- plot 1: parity plot ---
    ax = axes[0]
    ax.scatter(targets, preds, alpha=0.4, s=10, color='steelblue')
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='perfect')
    ax.set_xlabel('True (eV/atom)')
    ax.set_ylabel('Predicted (eV/atom)')
    ax.set_title(f'Parity Plot\nMAE={mae:.3f} | r={corr:.3f}')
    ax.legend()

    # --- plot 2: residuals ---
    ax = axes[1]
    residuals = preds - targets
    ax.scatter(targets, residuals, alpha=0.4, s=10, color='coral')
    ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
    ax.set_xlabel('True (eV/atom)')
    ax.set_ylabel('Residual (pred - true)')
    ax.set_title(f'Residuals\nstd={residuals.std():.3f}')

    # --- plot 3: error distribution ---
    ax = axes[2]
    ax.hist(np.abs(residuals), bins=50, color='mediumseagreen', edgecolor='white')
    ax.axvline(mae, color='red', linestyle='--', label=f'MAE={mae:.3f}')
    ax.set_xlabel('Absolute Error (eV/atom)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig('prediction_quality.png', dpi=150)
    plt.show()

    # summary
    print(f"MAE        : {mae:.4f} eV/atom")
    print(f"RMSE       : {np.sqrt(np.mean(residuals ** 2)):.4f} eV/atom")
    print(f"Pearson r  : {corr:.4f}")
    print(f"R²         : {corr ** 2:.4f}")


# load stats and run
stats = torch.load('mp_y_stats.pt')
plot_predictions(model, data, stats['mean'], stats['std'])

