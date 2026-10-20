"""
visualize.py — generate one crystal and display it with py3Dmol.

Usage:
    python visualize.py                          # uses checkpoints/best_flow_model.pt
    python visualize.py --ckpt path/to/model.pt
    python visualize.py --n_atoms 12 --steps 200

Output: crystal_viz.html  (open in any browser)
In Jupyter: import and call show_view(view) instead.
"""

import argparse
import webbrowser
from pathlib import Path

import torch

from GATr import GATrEncoder
from flow_matching import CrystalFlowMatching, GATrVelocityModel


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path, mv_channels=4, sc_channels=8, n_heads=1, n_layers=1):
    encoder   = GATrEncoder(mv_channels, sc_channels, n_heads, n_layers)
    vel_model = GATrVelocityModel(encoder, freeze_encoder=True)
    fm        = CrystalFlowMatching(vel_model)

    path = Path(ckpt_path)
    if path.exists():
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        fm.load_state_dict(ckpt['model'])
        print(f"Loaded: {path}")
    else:
        print(f"WARNING: {path} not found — using random weights")

    fm.eval()
    return fm


# ── Generation ────────────────────────────────────────────────────────────────

def generate(fm, n_atoms=8, n_steps=100):
    """Generate one crystal. Returns frac_coords (N,3) and lattice (3,3)."""
    # Dummy atom features (N, 7): placeholder, real use would pull from dataset
    atom_feats = torch.zeros(n_atoms, 7)
    atom_feats[:, 0] = 6.0 / 118.0   # carbon-ish atomic number (normalized)

    with torch.no_grad():
        fracs, lats = fm.generate(
            n_atoms_per_crystal=[n_atoms],
            atom_features=[atom_feats],
            n_steps=n_steps,
        )
    return fracs[0], lats[0]   # (N, 3), (3, 3)


# ── py3Dmol visualisation ─────────────────────────────────────────────────────

def _xyz_block(cart, element='C'):
    """Build an XYZ-format string from Cartesian positions."""
    lines = [str(len(cart)), "generated crystal"]
    for x, y, z in cart.tolist():
        lines.append(f"{element}  {x:.4f}  {y:.4f}  {z:.4f}")
    return "\n".join(lines)


def _add_cell_edges(view, L):
    """Draw the 12 edges of the unit-cell parallelepiped."""
    O = torch.zeros(3)
    a, b, c = L[0], L[1], L[2]
    corners = [O, a, b, c, a+b, a+c, b+c, a+b+c]
    edges   = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
    for i, j in edges:
        p, q = corners[i].tolist(), corners[j].tolist()
        view.addCylinder({
            'start': {'x': p[0], 'y': p[1], 'z': p[2]},
            'end':   {'x': q[0], 'y': q[1], 'z': q[2]},
            'radius': 0.06, 'color': '#aaaaaa',
            'fromCap': 1, 'toCap': 1,
        })


def make_view(frac, lattice, element='C'):
    """Return a configured py3Dmol.view object."""
    import py3Dmol

    cart = frac @ lattice          # fractional → Cartesian  (N, 3)
    view = py3Dmol.view(width=650, height=500)
    view.addModel(_xyz_block(cart, element), 'xyz')
    view.setStyle({'sphere': {'radius': 0.45, 'colorscheme': 'Jmol'}})
    _add_cell_edges(view, lattice)
    view.zoomTo()
    view.setBackgroundColor('white')
    return view


def save_html(view, path='crystal_viz.html'):
    """Save the view as a self-contained HTML file."""
    html = view._repr_html_()
    Path(path).write_text(f"<html><body>{html}</body></html>")
    print(f"Saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',    default='checkpoints/best_flow_model.pt')
    p.add_argument('--n_atoms', type=int, default=8)
    p.add_argument('--steps',   type=int, default=100)
    p.add_argument('--element', default='C', help='Element symbol for display')
    p.add_argument('--out',     default='crystal_viz.html')
    p.add_argument('--no_open', action='store_true', help='Skip opening browser')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    fm              = load_model(args.ckpt)
    frac, lattice   = generate(fm, n_atoms=args.n_atoms, n_steps=args.steps)

    # Quick stats
    vol = torch.linalg.det(lattice).abs().item()
    print(f"Atoms      : {args.n_atoms}")
    print(f"Cell volume: {vol:.2f}  (product of Cholesky lattice)")
    print(f"Lattice:\n{lattice.numpy()}")

    view = make_view(frac, lattice, element=args.element)
    out  = save_html(view, args.out)

    if not args.no_open:
        webbrowser.open(f"file://{Path(out).resolve()}")
