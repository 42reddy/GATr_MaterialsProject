"""
Riemannian conditional flow matching for crystal generation.

State space
───────────
  frac_coords : (N, 3)    fractional coordinates on T³ (periodic [0, 1))
  lattice_gram: (B, 3, 3) Gram matrix G = L @ Lᵀ  ∈ SPD(3)

Manifold operations
───────────────────
  T³    — flat torus; geodesic = wrap(x₀ + t·δ), log = shortest wraparound
  SPD(3)— curved; geodesic and target velocity derived from a single eigendecomp

Input encoding (step 5 in the plan)
────────────────────────────────────
  At time t, the current manifold state is encoded as 16-dim PGA multivectors:
    • Atom nodes  : grade-3 PGA points from Cartesian positions  (idx 11–14)
    • Lattice nodes: grade-3 PGA points from lattice vectors aᵢ  (idx 11–14)
  Time t and atom type are scalar channels fed into the encoder's scalar stream
  (handled by GATrProcessor: t injected additively into x_s before each run).

Three-stage velocity model (GATrVelocityModel)
──────────────────────────────────────────────
  1. FROZEN GATrEncoder  — equivariant feature extraction, per-node, NO pooling
  2. Trainable GATrProcessor — new GATr blocks, time conditioning via scalar stream
  3. Trainable VelocityDecoder — grade-specific extraction:
       atom nodes  grade-2 bivectors [8:11] → Cartesian vel → fractional vel
       lattice nodes grade-3 trivectors [12:15] → δL → V_G = δL Lᵀ + L δLᵀ
"""

import torch
import torch.nn as nn

from GATr import GATrProcessor


# ── SPD(3) manifold operations ─────────────────────────────────────────────────

def _eigh_clamp(A):
    """Eigendecompose symmetric A; clamp eigenvalues to ≥ 1e-8 for stability."""
    L, V = torch.linalg.eigh(A)
    return L.clamp(min=1e-8), V


def spd_sqrt(A):
    """A^(1/2).  A: (*, 3, 3) symmetric PD."""
    L, V = _eigh_clamp(A)
    return V @ torch.diag_embed(L.sqrt()) @ V.mT


def spd_inv_sqrt(A):
    """A^(-1/2).  A: (*, 3, 3) symmetric PD."""
    L, V = _eigh_clamp(A)
    return V @ torch.diag_embed(L.rsqrt()) @ V.mT


def spd_log(A, B):
    """Riemannian log map at A pointing toward B (tangent vector at A)."""
    A_sqrt  = spd_sqrt(A)
    A_isqrt = spd_inv_sqrt(A)
    M       = A_isqrt @ B @ A_isqrt
    Lm, Vm  = _eigh_clamp(M)
    return A_sqrt @ (Vm @ torch.diag_embed(Lm.log()) @ Vm.mT) @ A_sqrt


def spd_exp(A, V):
    """Riemannian exp map at A along tangent V: exp_A(V)."""
    A_sqrt  = spd_sqrt(A)
    A_isqrt = spd_inv_sqrt(A)
    M       = A_isqrt @ V @ A_isqrt
    Lm, Vm  = torch.linalg.eigh(M)          # V may not be PD
    return A_sqrt @ (Vm @ torch.diag_embed(Lm.exp()) @ Vm.mT) @ A_sqrt


def spd_symmetrize(M):
    return (M + M.mT) / 2.0


# ── Torus T³ operations ────────────────────────────────────────────────────────

def torus_wrap(x):
    """Wrap (*, 3) into [0, 1)."""
    return x - torch.floor(x)


def torus_log(x_base, x_target):
    """Shortest-path direction on T³; result in [-0.5, 0.5]³."""
    delta = x_target - x_base
    return delta - torch.round(delta)


# ── SPD noise sampling ─────────────────────────────────────────────────────────

def project_spd(G, eps=1e-4):
    """
    Project (*, 3, 3) onto SPD by symmetrizing and clamping eigenvalues to ≥ eps.
    Applied after each Euler step to prevent floating-point drift off the manifold.
    """
    G = (G + G.mT) / 2.0
    L, V = torch.linalg.eigh(G)
    return V @ torch.diag_embed(L.clamp(min=eps)) @ V.mT


def sample_spd_noise(B, device, scale=5.0):
    """
    Sample B random SPD matrices via Cholesky parameterisation.
    scale=5.0 so eigenvalues are O(5–25), matching real crystal Gram matrices
    (~O(9–100) for lattice params 3–10 Å).  scale=1.0 produced geodesics that
    were ~10× too long, causing large Euler steps and loss of positive-definiteness.
    """
    L = torch.randn(B, 3, 3, device=device) * 0.3
    L = torch.tril(L)
    diag = torch.nn.functional.softplus(
        L.diagonal(dim1=-2, dim2=-1)
    ) * scale + 0.5
    idx  = torch.arange(3, device=device)
    L    = L.clone()
    L[:, idx, idx] = diag
    return L @ L.mT   # (B, 3, 3) SPD


# ── Flow matching core ─────────────────────────────────────────────────────────

class RiemannianCrystalFlowMatching(nn.Module):
    """
    Riemannian conditional flow matching on T³^N × SPD(3).

    Lattice is represented by its Gram matrix G = L @ Lᵀ (always SPD).
    After generation, recover L = cholesky(G).

    Args:
        velocity_model: GATrVelocityModel or any module with signature
            (frac_t, G_t, t, batch) → (v_frac (N,3), v_G (B,3,3))
    """

    def __init__(self, velocity_model: nn.Module):
        super().__init__()
        self.velocity_model = velocity_model

    def flow_matching_loss(
        self,
        frac_coords:  torch.Tensor,   # (N, 3)    target fractional coords
        lattice_gram: torch.Tensor,   # (B, 3, 3) target Gram matrix (SPD)
        batch_idx:    torch.Tensor,   # (N,)
        batch,
    ) -> torch.Tensor:
        device = frac_coords.device
        B = lattice_gram.shape[0]
        N = frac_coords.shape[0]

        t = torch.rand(B, device=device)

        # ── Sample noise ──────────────────────────────────────────────────────
        x0_frac = torch.rand(N, 3, device=device)
        x0_G    = sample_spd_noise(B, device)

        # ── T³ geodesic interpolation ─────────────────────────────────────────
        delta_frac = torus_log(x0_frac, frac_coords)           # (N, 3)
        t_atom     = t[batch_idx].view(-1, 1)
        frac_t     = torus_wrap(x0_frac + t_atom * delta_frac) # (N, 3)

        # ── SPD(3): joint computation of G_t and vt_G_tgt ─────────────────────
        # Derives both from a single eigendecomposition of M = G₀⁻¹/² G₁ G₀⁻¹/².
        # This is more numerically stable than calling spd_geodesic + spd_log
        # separately (each does its own eigendecomp from a numerically noisier G_t).
        #
        # Geodesic:  G_t      = G₀^(1/2) M^t G₀^(1/2)
        # Velocity:  vt_G_tgt = G₀^(1/2) Λ M^t G₀^(1/2)
        #            where Λ = log(M),  M^t = exp(t·Λ)
        #
        # For a linear schedule κ(t)=t, κ̇=1, this equals d/dt[geodesic(t)].
        G0_sqrt  = spd_sqrt(x0_G)
        G0_isqrt = spd_inv_sqrt(x0_G)
        M        = G0_isqrt @ lattice_gram @ G0_isqrt           # (B, 3, 3)

        Lm, Vm   = _eigh_clamp(M)
        log_Lm   = Lm.log()

        # t.unsqueeze(-1) → (B, 1); (B, 1) * (B, 3) → (B, 3) ✓
        # t.view(-1, 1, 1) would give (B, 1, 1) * (B, 3) → (B, B, 3) — wrong
        Mt       = Vm @ torch.diag_embed((t.unsqueeze(-1) * log_Lm).exp()) @ Vm.mT  # M^t
        log_M    = Vm @ torch.diag_embed(log_Lm) @ Vm.mT       # Λ = log(M)

        G_t      = G0_sqrt @ Mt @ G0_sqrt                       # (B, 3, 3)
        vt_G_tgt = G0_sqrt @ log_M @ Mt @ G0_sqrt              # (B, 3, 3)

        # ── Predict velocities ────────────────────────────────────────────────
        pred_v_frac, pred_v_G = self.velocity_model(frac_t, G_t, t, batch)

        # ── Losses ────────────────────────────────────────────────────────────
        loss_frac = ((pred_v_frac - delta_frac) ** 2).mean()

        # Normalise SPD loss: Gram eigenvalues are O(lattice_param²) ≈ O(25–100)
        # while T³ targets are O(1); divide by mean‖G‖² to balance the two terms.
        G_scale  = (lattice_gram ** 2).mean().detach().clamp(min=1.0)
        loss_G   = ((pred_v_G - vt_G_tgt) ** 2).mean() / G_scale

        return loss_frac + loss_G

    @torch.no_grad()
    def generate(
        self,
        n_atoms_per_crystal: list,
        atom_features:       list,
        n_steps:             int = 100,
        device:              str = 'cpu',
    ):
        """
        Riemannian Euler integration from noise (t=0) to data (t=1).

        Returns:
            gen_frac_coords: list of (n_i, 3)
            gen_lattice    : list of (3, 3) lower-triangular (Cholesky of G_final)
        """
        B  = len(n_atoms_per_crystal)
        N  = sum(n_atoms_per_crystal)
        dt = 1.0 / n_steps

        frac_t = torch.rand(N, 3, device=device)
        G_t    = sample_spd_noise(B, device)

        for step in range(n_steps):
            t     = torch.full((B,), step * dt, device=device)
            batch = self._build_gen_batch(atom_features, device)

            v_frac, v_G = self.velocity_model(frac_t, G_t, t, batch)

            frac_t = torus_wrap(frac_t + v_frac * dt)
            G_t    = project_spd(spd_exp(G_t, v_G * dt))   # project back to SPD after each step

        lattice_t = torch.linalg.cholesky(G_t)

        gen_frac_coords, gen_lattice = [], []
        start = 0
        for i, n in enumerate(n_atoms_per_crystal):
            gen_frac_coords.append(frac_t[start:start + n].cpu())
            gen_lattice.append(lattice_t[i].cpu())
            start += n

        return gen_frac_coords, gen_lattice

    def _build_gen_batch(self, atom_features, device):
        from torch_geometric.data import Data, Batch
        return Batch.from_data_list([Data(x=f.to(device)) for f in atom_features])


CrystalFlowMatching = RiemannianCrystalFlowMatching


# ── Velocity decoder ──────────────────────────────────────────────────────────

class VelocityDecoder(nn.Module):
    """
    Grade-specific decoder that reads velocity tangent vectors directly from the
    geometric algebra grade structure of the processor's output multivectors.

    Grade-2 translation bivectors (e₀₁, e₀₂, e₀₃, MV idx 8–10)
        These ARE translations in PGA — extracting them gives Cartesian velocity,
        which is then converted to fractional via v_frac = v_cart @ L⁻¹.

    Grade-3 position trivectors (e₀₁₂, e₀₁₃, e₀₂₃, MV idx 12–14) on lattice nodes
        The processor "displaces" the lattice vector representations.
        Those displacements δaᵢ give the SPD tangent: V_G = δL Lᵀ + L δLᵀ,
        which is symmetric by construction — no symmetrize step needed.

    The channel-mixing projections (Linear over the C encoder channels) are the
    only learned weights here; everything else is determined by the GA grade structure.
    """

    _TRANS_BIV = slice(8, 11)    # e₀₁, e₀₂, e₀₃
    _POS_TRIV  = slice(12, 15)   # e₀₁₂, e₀₁₃, e₀₂₃

    def __init__(self, mv_channels: int, hidden: int = 64):
        super().__init__()
        C = mv_channels
        # Both heads map (*, spatial=3, C) → (*, 3, 1) → (*, 3)
        # Linear acts on last dim so shapes work without any reshape.
        self.atom_head = nn.Sequential(
            nn.Linear(C, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        self.lat_head = nn.Sequential(
            nn.Linear(C, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        atom_mvs_out: torch.Tensor,   # (N,  C, 16)
        lat_mvs_out:  torch.Tensor,   # (3B, C, 16)
        L_t:          torch.Tensor,   # (B,  3, 3)
        batch_idx:    torch.Tensor,   # (N,)
    ):
        B = lat_mvs_out.shape[0] // 3

        # ── Fractional velocity via grade-2 bivectors ─────────────────────────
        # (N, C, 3) → permute → (N, 3, C) → atom_head → (N, 3, 1) → (N, 3)
        trans_biv = atom_mvs_out[:, :, self._TRANS_BIV].permute(0, 2, 1)  # (N, 3, C)
        v_cart    = self.atom_head(trans_biv).squeeze(-1)                  # (N, 3)

        L_inv  = torch.linalg.inv(L_t)                                    # (B, 3, 3)
        v_frac = torch.bmm(v_cart.unsqueeze(1), L_inv[batch_idx]).squeeze(1)  # (N, 3)

        # ── SPD tangent via grade-3 trivectors of lattice nodes ────────────────
        # (3B, C, 3) → permute → (3B, 3, C) → lat_head → (3B, 3, 1) → (3B, 3)
        pos_triv = lat_mvs_out[:, :, self._POS_TRIV].permute(0, 2, 1)    # (3B, 3, C)
        delta_L  = self.lat_head(pos_triv).squeeze(-1).view(B, 3, 3)     # (B, 3, 3)

        # V_G = δL @ Lᵀ + L @ δLᵀ  — symmetric by construction, no extra step
        V_G = delta_L @ L_t.mT + L_t @ delta_L.mT                        # (B, 3, 3)

        return v_frac, V_G


# ── Three-stage velocity model ─────────────────────────────────────────────────

class GATrVelocityModel(nn.Module):
    """
    Three-stage velocity field for RiemannianCrystalFlowMatching.

    Stage 1 — FROZEN GATrEncoder (feature extractor, per-node, no global pooling)
    ──────────────────────────────────────────────────────────────────────────────
    The pretrained encoder was trained to compress *clean* crystal data into a
    single property prediction.  Here it is repurposed as a structural feature
    extractor only: we stop before the global_mean_pool and return per-node
    equivariant features (N+3B, C, 16) and their scalar companions (N+3B, sc).

    Input to the encoder:
      • Atom nodes   (N):  grade-3 PGA points from Cartesian positions at time t
                           cart_t = frac_t @ chol(G_t)
      • Lattice nodes (3B): grade-3 PGA points from lattice vectors aᵢ = rows of chol(G_t)
      • Scalar stream: atom type features (N, 7) for atom nodes, zeros for lattice nodes

    Stage 2 — TRAINABLE GATrProcessor (time-conditioned equivariant refinement)
    ─────────────────────────────────────────────────────────────────────────────
    New GATrBlocks that:
      • Inject time t into the scalar stream additively before the first block
      • Run full equivariant attention across ALL nodes (atoms + lattice together)
      • Learn how the velocity field varies with t from scratch

    Stage 3 — TRAINABLE VelocityDecoder (grade-specific extraction)
    ────────────────────────────────────────────────────────────────
      Atom nodes    grade-2 bivectors [8:11] → Cartesian v_cart → fractional v_frac
      Lattice nodes grade-3 trivectors [12:15] → δL → V_G = δL Lᵀ + L δLᵀ

    Args:
        encoder      : pretrained GATrEncoder (will be frozen)
        freeze_encoder: if True, encoder weights are not updated
        proc_layers  : number of new GATrBlock layers in the processor
        hidden       : hidden dim in the decoder channel-mixing MLPs
    """

    def __init__(
        self,
        encoder,
        freeze_encoder: bool = True,
        proc_layers:    int  = 2,
        hidden:         int  = 64,
    ):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        C  = encoder.mv_channels
        sc = encoder.scalar_channels
        nh = encoder.n_heads

        self.processor = GATrProcessor(C, sc, n_heads=nh, n_layers=proc_layers)
        self.decoder   = VelocityDecoder(C, hidden=hidden)

    def forward(
        self,
        frac_coords_t:  torch.Tensor,   # (N, 3)
        lattice_gram_t: torch.Tensor,   # (B, 3, 3)
        t:              torch.Tensor,   # (B,)
        batch,                          # PyG Batch
    ):
        device    = frac_coords_t.device
        N         = frac_coords_t.shape[0]
        B         = lattice_gram_t.shape[0]
        # Ensure batch_idx is a 1D long tensor for indexing
        batch_idx = batch.batch.long()                                   

        # 1. Recover lattice (Cholesky)
        # L_t shape: (B, 3, 3)
        L_t = torch.linalg.cholesky(lattice_gram_t)              

        # 2. Map lattices to atoms for Cartesian conversion
        # L_t[batch_idx] results in (N, 3, 3)
        # cart_t shape: (N, 3)
        cart_t = torch.bmm(frac_coords_t.unsqueeze(1), L_t[batch_idx]).squeeze(1)

        # 3. Build Atom MVs (N, 16)
        atom_mvs = torch.zeros(N, 16, device=device)
        atom_mvs[:, 11]    = 1.0      
        atom_mvs[:, 12:15] = cart_t   

        # 4. Build Lattice MVs (3B, 16)
        # This was the crash site. We must reshape explicitly.
        lat_mvs = torch.zeros(B * 3, 16, device=device)
        lat_mvs[:, 11]    = 1.0
        # Use .reshape(-1, 3) to be safe, which maps (B, 3, 3) -> (B*3, 3)
        lat_mvs[:, 12:15] = L_t.reshape(-1, 3)

        # 5. Create lattice batch indices (0,0,0, 1,1,1, ..., B-1,B-1,B-1)
        lat_batch = torch.arange(B, device=device).repeat_interleave(3)

        # 6. Combine
        all_mvs     = torch.cat([atom_mvs, lat_mvs], dim=0)          # (N+3B, 16)
        all_batch   = torch.cat([batch_idx, lat_batch], dim=0)       # (N+3B,)
        
        # Ensure all_scalars matches the N+3B count
        # batch.x is (N, 7)
        lat_scalars = torch.zeros(B * 3, 7, device=device)
        all_scalars = torch.cat([batch.x, lat_scalars], dim=0)       # (N+3B, 7)

        # 7. Encoder & Processor
        _, _, x_mv, x_s = self.encoder(
            all_mvs, all_scalars, all_batch, return_node_mvs=True
        )

        t_per_node = t[all_batch]                                  
        z_mv, _    = self.processor(x_mv, x_s, t_per_node)

        # 8. Decode
        # Atom features are the first N, Lattice features are the remaining 3B
        return self.decoder(z_mv[:N], z_mv[N:], L_t, batch_idx)
