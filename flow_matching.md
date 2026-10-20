# Plan: Riemannian Flow Matching with GATr Backbone

## Context

The user has a pretrained GATr encoder that takes crystal data encoded as 16-dim PGA multivectors and outputs equivariant features. They want to upgrade the existing Euclidean flow matching (`flow_matching.py`) to **Riemannian flow matching**, using the correct geometric structure for crystal data:

- Atomic fractional coordinates → Torus **T³** (periodic domain)
- Lattice matrices → **SPD(3)** (symmetric positive definite 3×3 matrices)

The core confusion: how do 16-dim multivectors (what GATr eats) relate to manifold-respecting Riemannian FM? Answer below.

---

## The Key Insight: Two Separate Levels

```
MANIFOLD LEVEL (state space)          COMPUTATIONAL LEVEL (GATr input)
─────────────────────────────         ─────────────────────────────────
frac_t ∈ T³^N                        cart_t = frac_t @ L_t  →  node_mvs (N,16)
L_t ∈ SPD(3)                         GATr processes multivectors as usual
        ↑                                         ↓
 Geodesic interpolation               Outputs per-node features (N, D)
 Log/Exp maps                                     ↓
 Riemannian loss                      Velocity heads decode features
        ↑                             into TANGENT VECTORS on manifolds
 Integration uses Exp map                         ↓
                                      v_frac (N,3) on T T³^N
                                      v_L (B,3,3) symmetric on T_L SPD(3)
```

The multivectors don't need to "be on the manifold." They are just how GATr *processes* the current Cartesian positions. The manifold structure governs:

1. **Interpolation** (geodesic paths, not straight lines)
2. **Target velocity** (log map, not x1 - x0)
3. **Integration at inference** (Exp map, not Euler step)
4. **Loss metric** (Riemannian inner product for SPD)

---

## Full Data Flow for Riemannian FM Training

```
Given training crystal: frac_1 ∈ T³^N, L_1 ∈ SPD(3)

Step 1 — Sample noise
    x0_frac ~ Uniform[0,1)^3  per atom         # base dist on T³
    x0_L    ~ Wishart or LKJ  per crystal       # base dist on SPD(3)

Step 2 — Sample time t ~ Uniform[0,1] per crystal

Step 3 — Geodesic interpolation (compute x_t)
    # On T³ (flat, wrapping):
    delta_frac = wrap(frac_1 - x0_frac)        # shortest-path direction, in [-0.5, 0.5]³
    frac_t = wrap(x0_frac + t * delta_frac)    # linear on the cover, then wrap

    # On SPD(3) (curved, matrix geodesic):
    # Geodesic: L_t = x0_L^(1/2) exp(t * log(x0_L^(-1/2) L_1 x0_L^(-1/2))) x0_L^(1/2)
    L_t = spd_geodesic(x0_L, L_1, t)

Step 4 — Compute TARGET velocities (log maps at x_t)
    # On T³ (constant velocity = delta_frac):
    v_frac_target = delta_frac      # (N, 3)

    # SPD: differentiate the geodesic, don't re-evaluate the log map at L_t
    Λ         = logm(x0_L^(-1/2) L_1 x0_L^(-1/2))
    v_L_target = L_1^(1/2) Λ exp(κ(t)·Λ) L_1^(1/2) * κ̇(t)
           # for linear schedule κ̇ = -1, κ(t) = 1-t             # (B, 3, 3) symmetric

Step 5 — Encode current state for GATr
    cart_t = frac_t @ L_t[batch]               # (N, 3) Cartesian
    node_mvs_t = encode_pga(cart_t)            # (N, 16) multivectors (SAME encoding as always)
    node_scalars = batch.x                     # (N, 7) atom features

Step 6 — GATr encoder (frozen)
    node_feats = encoder(node_mvs_t, node_scalars, batch)  # (N, D)

Step 7 — Velocity prediction heads (trainable)
    t_embed = time_mlp(t[batch_idx])           # (N, T)
    lat_embed = lattice_mlp(L_t[batch_idx].view(-1,9))  # (N, K)
    h = cat([node_feats, t_embed, lat_embed])  # (N, D+T+K)

    v_frac_pred = coord_head(h)                # (N, 3) — tangent on T³
    v_L_pred = lattice_head(global_pool(h))    # (B, 6) → sym (B, 3, 3)

Step 8 — Riemannian loss
    loss_frac = MSE(v_frac_pred, v_frac_target)                    # T³ is flat, L2 is fine
    loss_L = ||v_L_pred - v_L_target||^2_F (or Riemannian metric)  # SPD loss
    loss = loss_frac + loss_L
```

---

## Inference (Riemannian Integration)

```python
frac_t = rand([N, 3])       # noise on T³
L_t = sample_spd(B)         # noise on SPD(3)

for step in range(n_steps):
    t = torch.full([B], step / n_steps)
    v_frac, v_L = velocity_model(frac_t, L_t, t, batch)
    
    # Exp map on T³ (just wrap)
    frac_t = wrap(frac_t + v_frac * dt)
    
    # Exp map on SPD(3)
    L_t = spd_exp(L_t, v_L * dt)
```

---

## Files to Modify / Create


| File               | Action    | Change                                                                                                                               |
| ------------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `flow_matching.py` | Modify    | Replace Euclidean interp with Riemannian; add `spd_log`, `spd_exp`, `spd_geodesic`, `torus_wrap`                                     |
| `flow_training.py` | Modify    | Switch lattice representation from flat (B,9) to SPD matrix (B,3,3); add Cholesky factorization to ensure positivity                 |
| `GATr.py`          | Modify    | Keep `GATrVelocityModel` nearly the same; change lattice head output from 9 → 6 (upper triangle of sym matrix); add symmetrize layer |
| `GA_encoder.py`    | No change | PGA encoding is unchanged — it always takes Cartesian positions                                                                      |


---

## Critical Implementation Details

### Ensuring SPD(3) validity

Lattice matrices are already positive definite (they're physical lattice bases). Use Cholesky: `L = chol(M @ M.T + eps*I)` to parameterize the base distribution, ensuring generated lattices are SPD throughout.

### Torus wrap function

```python
def wrap(x):
    return x - torch.floor(x)  # maps to [0, 1)

def torus_log(x_t, x_1):
    delta = x_1 - x_t
    return delta - torch.round(delta)  # shortest path in [-0.5, 0.5]
```

### SPD geodesic

```python
def spd_geodesic(A, B, t):
    A_sqrt = matrix_sqrt(A)
    A_isqrt = matrix_inv_sqrt(A)
    M = A_isqrt @ B @ A_isqrt
    return A_sqrt @ matrix_pow(M, t) @ A_sqrt

def spd_log(A, B):
    # Tangent vector at A pointing toward B
    A_sqrt = matrix_sqrt(A)
    A_isqrt = matrix_inv_sqrt(A)
    return A_sqrt @ matrix_log(A_isqrt @ B @ A_isqrt) @ A_sqrt
```

Use `torch.linalg.eigh` (symmetric eigendecomposition) for all matrix functions — numerically stable.

### Velocity head output for SPD tangent vectors

SPD(3) tangent space at any point = symmetric 3×3 matrices = 6 independent values. Head outputs (B, 6) → unsqueeze to (B, 3, 3) using upper-triangle indexing → symmetrize.

---

## Verification Plan

1. **Unit test log/exp maps**: `spd_exp(L, spd_log(L, L2)) ≈ L2`
2. **Unit test geodesic**: `spd_geodesic(L0, L1, 0) ≈ L0`, `spd_geodesic(L0, L1, 1) ≈ L1`
3. **Unit test torus wrap**: `wrap(frac + v_frac*dt)` stays in [0,1)
4. **Training sanity**: loss should decrease; generated lattices should have `det(L) > 0`
5. **Generation check**: run 100-step Euler on T³ × SPD(3), verify output is valid crystal

