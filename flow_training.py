"""
flow_training.py — Riemannian flow matching for crystal generation.

Trains GATrVelocityModel (coord_head + lattice_head) on top of a frozen
pretrained GATrEncoder.  Dataset: piezo_data.pt.

State space:
    frac_coords  : (N, 3)    fractional coordinates on T³
    lattice_gram : (B, 3, 3) Gram matrix G = L @ Lᵀ  ∈ SPD(3)

Run:
    python flow_training.py
    python flow_training.py --epochs 300 --lr 3e-4
    python flow_training.py --finetune_encoder
    python flow_training.py --resume checkpoints/best_flow_model.pt
"""

import argparse
import random
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from GATr import GATrEncoder
from flow_matching import CrystalFlowMatching, GATrVelocityModel

warnings.filterwarnings("ignore")


class EMA:
    """Exponential moving average of trainable parameters (decay → 1 keeps history longer)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.named_parameters()
            if v.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.named_parameters():
            if v.requires_grad:
                self.shadow[k].mul_(d).add_(v.data, alpha=1.0 - d)

    def apply_shadow(self, model: nn.Module) -> None:
        self._backup.clear()
        for k, v in model.named_parameters():
            if v.requires_grad:
                self._backup[k] = v.data.clone()
                v.data.copy_(self.shadow[k])

    def restore(self, model: nn.Module) -> None:
        for k, v in model.named_parameters():
            if v.requires_grad:
                v.data.copy_(self._backup[k])
        self._backup.clear()

    def state_dict(self) -> dict:
        return {
            'decay': self.decay,
            'shadow': {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd.get('decay', self.decay))
        self.shadow = {k: v.clone() for k, v in sd['shadow'].items()}


# ── Data helpers ──────────────────────────────────────────────────────────────

def build_batch_inputs(batch, device):
    """
    Extracts the Riemannian flow-matching state from a PyG batch.

    piezo_data.pt stores:
      node_mvs[:, 12:15]  — Cartesian positions in Å
      lattice_matrix       — (1, 9) row-flattened per graph → (B, 9) after batching

    Returns:
        frac_coords  : (N, 3)    fractional coordinates  (cart @ L⁻¹)
        lattice_gram : (B, 3, 3) Gram matrix G = L @ Lᵀ  ∈ SPD(3)
                       Encodes lattice lengths & angles, always symmetric PD.
        batch_idx    : (N,)      crystal index per atom
    """
    batch_idx   = batch.batch                              # (N,)
    cart        = batch.node_mvs[:, 12:15]                 # (N, 3)
    lattice_mat = batch.lattice_matrix.view(-1, 3, 3)      # (B, 3, 3)  raw lattice

    L_inv       = torch.linalg.inv(lattice_mat)
    frac_coords = torch.bmm(
        cart.unsqueeze(1), L_inv[batch_idx]
    ).squeeze(1)                                           # (N, 3)

    # Gram matrix: G = L @ Lᵀ  — symmetric positive definite
    # Captures all geometric information (lengths, angles) without orientation.
    # Physical lattice matrices have det > 0, so G is always PD.
    lattice_gram = lattice_mat @ lattice_mat.mT            # (B, 3, 3)

    return frac_coords, lattice_gram, batch_idx


# ── Train / val loops ─────────────────────────────────────────────────────────

def train_epoch(fm, loader, optimizer, device, grad_clip, ema=None):
    fm.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        frac_coords, lattice_gram, batch_idx = build_batch_inputs(batch, device)

        loss = fm.flow_matching_loss(frac_coords, lattice_gram, batch_idx, batch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(fm.parameters(), max_norm=grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(fm)
        total += loss.item()

    return total / len(loader)


@torch.no_grad()
def val_epoch(fm, loader, device, ema=None):
    if ema is not None:
        ema.apply_shadow(fm)
    try:
        fm.eval()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            frac_coords, lattice_gram, batch_idx = build_batch_inputs(batch, device)
            total += fm.flow_matching_loss(
                frac_coords, lattice_gram, batch_idx, batch
            ).item()
        return total / len(loader)
    finally:
        if ema is not None:
            ema.restore(fm)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',             default='piezo_data.pt')
    p.add_argument('--encoder_ckpt',     default='pretrained_encoder.pt')
    p.add_argument('--ckpt_dir',         default='checkpoints')
    p.add_argument('--epochs',           type=int,   default=200)
    p.add_argument('--batch_size',       type=int,   default=128)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--grad_clip',        type=float, default=1.0)
    p.add_argument('--val_frac',         type=float, default=0.15)
    p.add_argument('--seed',             type=int,   default=42)
    # Must match the architecture used during pretraining
    p.add_argument('--mv_channels',      type=int,   default=12)
    p.add_argument('--sc_channels',      type=int,   default=16)
    p.add_argument('--n_heads',          type=int,   default=8)
    p.add_argument('--n_layers',         type=int,   default=4)
    p.add_argument('--finetune_encoder', action='store_false')
    p.add_argument('--resume',           default=None)
    p.add_argument('--ema_decay',        type=float, default=0.999)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading {args.data} ...")
    graphs = torch.load(args.data, weights_only=False)
    random.shuffle(graphs)

    n_val      = max(1, int(len(graphs) * args.val_frac))
    val_data   = graphs[:n_val]
    train_data = graphs[n_val:]
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder = GATrEncoder(
        mv_channels    =args.mv_channels,
        scalar_channels=args.sc_channels,
        n_heads        =args.n_heads,
        n_layers       =args.n_layers,
    )

    ckpt_path = Path(args.encoder_ckpt)
    """if ckpt_path.exists():
        encoder.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Pretrained encoder loaded from {ckpt_path}")
    else:
        print(f"WARNING: {ckpt_path} not found — encoder weights are random")"""

    vel_model = GATrVelocityModel(encoder, freeze_encoder=not args.finetune_encoder)
    fm        = CrystalFlowMatching(vel_model).to(device)
    ema       = EMA(fm, decay=args.ema_decay) if args.ema_decay > 0 else None

    n_trainable = sum(p.numel() for p in fm.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in fm.parameters())
    frozen_str  = 'encoder frozen' if not args.finetune_encoder else 'encoder unfrozen'
    print(f"Params: {n_trainable:,} trainable / {n_total:,} total  ({frozen_str})")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in fm.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=15, factor=0.85, min_lr=1e-6,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float('inf')
    if args.resume and Path(args.resume).exists():
        ckpt        = torch.load(args.resume, map_location=device)
        fm.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ema is not None:
            if ckpt.get('ema') is not None:
                ema.load_state_dict(ckpt['ema'])
            else:
                for k, v in fm.named_parameters():
                    if v.requires_grad:
                        ema.shadow[k].copy_(v.data)
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f"Resumed from epoch {start_epoch}  (best val: {best_val:.4f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in tqdm(range(start_epoch, args.epochs), desc='Flow matching'):
        tr_loss = train_epoch(
            fm, train_loader, optimizer, device, args.grad_clip, ema=ema
        )
        va_loss = val_epoch(fm, val_loader, device, ema=ema)
        lr_sched.step(va_loss)

        if (epoch + 1) % 10 == 0:
            tqdm.write(
                f"Ep {epoch+1:4d} | "
                f"train {tr_loss:.4f} | val {va_loss:.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )

        if va_loss < best_val:
            best_val = va_loss
            if ema is not None:
                ema.apply_shadow(fm)
                best_sd = fm.state_dict()
                ema.restore(fm)
            else:
                best_sd = fm.state_dict()
            payload = {
                'epoch'    : epoch,
                'model'    : best_sd,
                'optimizer': optimizer.state_dict(),
                'best_val' : best_val,
                'args'     : vars(args),
            }
            if ema is not None:
                payload['ema'] = ema.state_dict()
            torch.save(payload, Path(args.ckpt_dir) / 'best_flow_model.pt')

        if (epoch + 1) % 50 == 0:
            ep_payload = {
                'epoch'    : epoch,
                'model'    : fm.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val' : best_val,
            }
            if ema is not None:
                ep_payload['ema'] = ema.state_dict()
            torch.save(ep_payload, Path(args.ckpt_dir) / f'flow_epoch{epoch + 1}.pt')

    print(f"\nDone.  Best val loss: {best_val:.4f}")
    print(f"Best model → {args.ckpt_dir}/best_flow_model.pt")


if __name__ == '__main__':
    main()
