import torch
import torch.nn as nn
from EquiLinear import EquivariantLinear
from EquiLayerNorm import EquivariantLayerNorm
from ScalarGELU import GatedGELU
from Attention import EquivariantAttention
from GeoBiLinear import GeoBilinear
from torch_geometric.nn import global_mean_pool


class GATrBlock(nn.Module):
    # unchanged — exactly as you have it
    def __init__(self, mv_channels, scalar_channels, n_heads):
        super().__init__()
        self.norm1 = EquivariantLayerNorm(mv_channels, scalar_channels)
        self.attn = EquivariantAttention(mv_channels, scalar_channels, n_heads)
        self.norm2 = EquivariantLayerNorm(mv_channels, scalar_channels)
        self.mlp_in = EquivariantLinear(mv_channels, mv_channels)
        self.geo_bi = GeoBilinear()
        self.gelu = GatedGELU()
        self.mlp_out = EquivariantLinear(mv_channels * 2, mv_channels)
        self.s_mlp_in = nn.Linear(scalar_channels, scalar_channels)
        self.s_mlp_out = nn.Linear(scalar_channels, scalar_channels)

    def forward(self, x_mv, x_s, edge_index=None):
        r_mv, r_s = x_mv, x_s
        x_mv, x_s = self.norm1(x_mv, x_s)
        x_mv, x_s = self.attn(x_mv, x_s)
        x_mv, x_s = x_mv + r_mv, x_s + r_s

        r_mv, r_s = x_mv, x_s
        x_mv, x_s = self.norm2(x_mv, x_s)
        x_proj = self.mlp_in(x_mv)
        y_proj = self.mlp_in(x_mv)
        z_proj = self.mlp_in(x_mv)
        x_mv = self.geo_bi(x_proj, y_proj, z_proj)
        x_mv, x_s = self.gelu(x_mv, x_s)
        x_mv = self.mlp_out(x_mv)
        x_s = torch.relu(self.s_mlp_in(x_s))
        x_s = self.s_mlp_out(x_s)
        x_mv, x_s = x_mv + r_mv, x_s + r_s
        return x_mv, x_s


class GATrEncoder(nn.Module):
    """
    Shared encoder — pretrain on MP, fine-tune on IC.
    Outputs a graph-level embedding of shape (B, mv_channels + scalar_channels).
    """

    def __init__(self, mv_channels=4, scalar_channels=8, n_heads=2, n_layers=2):
        super().__init__()
        self.mv_channels = mv_channels
        self.scalar_channels = scalar_channels
        self.n_heads = n_heads   # stored so GATrProcessor can match it

        # input projections
        self.mv_input_proj = EquivariantLinear(1, mv_channels)
        self.s_input_proj = nn.Linear(7, scalar_channels)
        self.s_to_mv_grade0 = nn.Linear(7, mv_channels)

        # transformer blocks
        self.blocks = nn.ModuleList([
            GATrBlock(mv_channels, scalar_channels, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, node_mvs, node_scalars, batch, edge_index=None,
                return_node_feats=False, return_node_mvs=False):
        # embed
        x_mv = node_mvs.unsqueeze(1).unsqueeze(0)
        x_mv = self.mv_input_proj(x_mv)
        grade0_mv = torch.zeros_like(x_mv)
        grade0_mv[0, :, :, 0] = self.s_to_mv_grade0(node_scalars)
        x_mv = x_mv + grade0_mv
        x_s = self.s_input_proj(node_scalars).unsqueeze(0)

        # blocks
        for block in self.blocks:
            x_mv, x_s = block(x_mv, x_s, edge_index)

        # pool → graph-level embedding
        x_mv = x_mv.squeeze(0)  # (N, mv_channels, 16)
        x_s = x_s.squeeze(0)    # (N, scalar_channels)
        x_node = torch.cat([x_mv[..., 0], x_s], dim=-1)  # (N, mv_channels + scalar_channels)
        x = global_mean_pool(x_node, batch)               # (B, mv_channels + scalar_channels)

        if return_node_mvs:
            # Return full per-node multivector tensors AND the scalar stream separately.
            # x_mv : (N, mv_channels, 16) — all grades preserved
            # x_s  : (N, scalar_channels) — processed scalar stream (separate from x_mv)
            # These feed directly into GATrProcessor without any re-projection.
            return x, x_node, x_mv, x_s
        if return_node_feats:
            return x, x_node  # graph-level + per-node scalars
        return x


class GATrProcessor(nn.Module):
    """
    Trainable equivariant processor that operates on top of a frozen encoder.

    Role in the velocity model
    ──────────────────────────
    The frozen GATrEncoder is a feature extractor: it maps noisy crystal geometry
    (encoded as grade-3 PGA multivectors) to rich per-node equivariant features.
    Those features know nothing about the flow's time coordinate.

    GATrProcessor adds that missing time awareness while keeping full equivariance:
      1. Time t is projected to scalar_channels and ADDED to the scalar stream.
         This conditions all subsequent attention layers on t without touching the
         multivector stream's geometric structure.
      2. K new GATrBlocks then refine the features — they can attend across atoms
         and lattice nodes together, learning how the velocity field varies with t.

    This separation (frozen encoder = structure, trainable processor = dynamics)
    means the pretrained geometric representation is preserved while the processor
    is free to learn the time-dependent velocity field from scratch.

    Args:
        mv_channels     : must match the frozen encoder's mv_channels
        scalar_channels : must match the frozen encoder's scalar_channels
        n_heads         : attention heads (use encoder.n_heads for consistency)
        n_layers        : number of new GATrBlock layers in the processor
    """

    def __init__(self, mv_channels, scalar_channels, n_heads, n_layers):
        super().__init__()
        # Two-layer time embedding → scalar stream (additive FiLM-free conditioning)
        self.time_proj = nn.Sequential(
            nn.Linear(1, scalar_channels),
            nn.SiLU(),
            nn.Linear(scalar_channels, scalar_channels),
        )
        self.blocks = nn.ModuleList([
            GATrBlock(mv_channels, scalar_channels, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, x_mv, x_s, t_per_node, edge_index=None):
        """
        x_mv       : (N, C, 16)  per-node multivectors from frozen encoder
        x_s        : (N, sc)     per-node scalar features from frozen encoder
        t_per_node : (N,)        time t broadcast to every node in the batch
        Returns    : (N, C, 16), (N, sc)  processed features
        """
        # Inject time into scalar stream once; it then propagates through all blocks
        x_s = x_s + self.time_proj(t_per_node.unsqueeze(-1))   # (N, sc)

        # GATrBlock operates on tensors with a leading batch-of-1 dimension
        x_mv = x_mv.unsqueeze(0)   # (1, N, C, 16)
        x_s  = x_s.unsqueeze(0)    # (1, N, sc)

        for block in self.blocks:
            x_mv, x_s = block(x_mv, x_s, edge_index)

        return x_mv.squeeze(0), x_s.squeeze(0)   # (N, C, 16), (N, sc)


class GATrHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, n_tasks=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_tasks)   # ← one output per task
        )

    def forward(self, x):
        return self.mlp(x)                   # (B, n_tasks)


class GATrModel(nn.Module):
    """
    Full model = encoder + head.

    Usage:
        # pretraining on MP
        model = GATrModel(task='formation_energy')

        # fine-tuning on IC
        model = GATrModel(task='ic')
        model.load_encoder('pretrained_encoder.pt')
    """

    def __init__(self, mv_channels=4, scalar_channels=8, n_heads=2, n_layers=2, n_tasks = 4):
        super().__init__()
        self.n_tasks = n_tasks
        self.encoder = GATrEncoder(mv_channels, scalar_channels, n_heads, n_layers)
        embed_dim = mv_channels + scalar_channels
        self.head = GATrHead(in_dim=embed_dim, hidden_dim=32, n_tasks = self.n_tasks)

    def forward(self, node_mvs, node_scalars, batch, edge_index=None):
        x = self.encoder(node_mvs, node_scalars, batch, edge_index)
        return self.head(x)

    def swap_head(self, hidden_dim=32):
        """Call this when switching from pretraining to fine-tuning."""
        embed_dim = self.encoder.mv_channels + self.encoder.scalar_channels
        self.head = GATrHead(in_dim=embed_dim, hidden_dim=hidden_dim, n_tasks = self.n_tasks)

    def save_encoder(self, path='pretrained_encoder.pt'):
        torch.save(self.encoder.state_dict(), path)
        print(f"Encoder saved to {path}")

    def load_encoder(self, path='pretrained_encoder.pt'):
        self.encoder.load_state_dict(torch.load(path))
        print(f"Encoder loaded from {path}")
