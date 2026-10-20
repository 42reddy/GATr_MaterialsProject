import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from GATr import GATrModel
from tqdm import tqdm
import random
import warnings

warnings.filterwarnings("ignore")

# --- load MP data ---
mp_data = torch.load('mp_data.pt', weights_only=False)

# --- infer target dimensionality and normalize ---
ys = torch.stack([d.y.reshape(-1) for d in mp_data])  # shape: [N, n_tasks]
n_tasks = ys.shape[1]
y_mean = ys.mean(dim=0)
y_std = ys.std(dim=0)
y_std = torch.clamp(y_std, min=1e-8)  # avoid divide-by-zero for near-constant targets

for i in range(n_tasks):
    print(
        f"Target {i} | mean: {y_mean[i].item():.3f} | std: {y_std[i].item():.3f} | "
        f"min: {ys[:, i].min().item():.3f} | max: {ys[:, i].max().item():.3f}"
    )

for d in mp_data:
    d.y = ((d.y.reshape(-1) - y_mean) / y_std).reshape(1, -1)

# save stats for later
torch.save({'mean': y_mean, 'std': y_std}, 'mp_y_stats.pt')

# --- shuffle and split ---
random.seed(42)
random.shuffle(mp_data)
n          = len(mp_data)
train_data = mp_data[:int(n * 0.9)]
val_data   = mp_data[int(n * 0.9):]
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# --- model output size matches dataset targets ---
model = GATrModel(mv_channels=4, scalar_channels=8, n_heads=1, n_layers=1, n_tasks=n_tasks)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(n_params)

# --- pretraining ---
def run_pretraining(model, train_data, val_data, epochs=100, batch_size=64, lr=1e-3):
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model        = model.to(device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, num_workers=2, pin_memory=True)
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion    = nn.MSELoss()
    best_val     = float('inf')

    for epoch in tqdm(range(epochs), desc='Pretraining'):
        # train
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch      = batch.to(device)
            pred       = model(batch.node_mvs, batch.x, batch.batch, batch.edge_index)

            loss = criterion(pred, batch.y.view(-1, n_tasks))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch     = batch.to(device)
                pred      = model(batch.node_mvs, batch.x, batch.batch, batch.edge_index)
                val_loss += criterion(pred, batch.y.view(-1, n_tasks)).item()

        val_loss_avg = val_loss / len(val_loader)
        scheduler.step(val_loss_avg)

        # save best encoder
        if val_loss_avg < best_val:
            best_val = val_loss_avg
            model.save_encoder('pretrained_encoder.pt')

        if (epoch + 1) % 5 == 0:
            tqdm.write(
                f"Epoch {epoch+1:3d} | "
                f"Train: {train_loss/len(train_loader):.4f} | "
                f"Val: {val_loss_avg:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    print(f"\nBest val loss: {best_val:.4f}")
    print("Encoder saved to pretrained_encoder.pt")
    return model

# --- run ---
model = run_pretraining(model, train_data, val_data, epochs=100, batch_size=64, lr=1e-3)
