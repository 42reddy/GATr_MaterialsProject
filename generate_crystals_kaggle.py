import os
import sys
import torch
import numpy as np

# ── Kaggle paths (mirror finetune_diffcsp_kaggle.py) ──
WORKING        = "/kaggle/working"
DIFFCSP_REPO   = os.path.join(WORKING, "DiffCSP")
PRETRAINED_RUN = "/kaggle/input/mp-gen/mp_gen"       # ← adjust slug
FINETUNED_CKPT = os.path.join(WORKING, "checkpoints", "diffcsp_na_electrolyte_best.pt")
OUT_DIR        = os.path.join(WORKING, "generated_cifs")

os.environ["PROJECT_ROOT"] = DIFFCSP_REPO
os.environ.setdefault("HYDRA_JOBS", DIFFCSP_REPO)
os.environ.setdefault("WABDB_DIR",  DIFFCSP_REPO)

sys.path.append(DIFFCSP_REPO)
sys.path.append(os.path.join(DIFFCSP_REPO, "scripts"))
sys.path.append(WORKING)

# PyTorch 2.6 weights_only guard
import torch.serialization as _ts
_torch_load_real = _ts.load
torch.load = lambda *a, **kw: _torch_load_real(*a, **{**kw, "weights_only": False})

from diffcsp_data import load_pretrained_diffcsp


# ── 1. load the fine-tuned generator ──────────────────────────────────────────
# Architecture + pretrained weights first, then overlay the fine-tuned state dict.
# This is safer than load_from_checkpoint on the fine-tuned file because the
# fine-tuned .pt only stores state_dict, not hparams.

def load_finetuned(pretrained_run, finetuned_ckpt):
    model, cfg = load_pretrained_diffcsp(DIFFCSP_REPO, pretrained_run)
    ckpt = torch.load(finetuned_ckpt)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"Fine-tuned weights loaded (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")
    return model


# ── 2. build a minimal generation batch ───────────────────────────────────────
# DiffCSP's sample() only needs num_atoms per crystal and the batch-assignment
# vector — atom types and coordinates are initialised from noise internally.
# We draw num_atoms from the empirical distribution of Na-electrolyte structures
# (roughly 6–30 atoms per primitive cell).

def make_gen_batch(num_crystals, min_atoms=6, max_atoms=30, seed=0):
    from torch_geometric.data import Data, Batch
    rng = np.random.default_rng(seed)
    sizes = rng.integers(min_atoms, max_atoms + 1, size=num_crystals).tolist()
    data_list = [
        Data(num_atoms=torch.tensor(n), num_nodes=n)
        for n in sizes
    ]
    batch = Batch.from_data_list(data_list)
    # Batch.from_data_list stacks num_atoms as a tensor automatically
    return batch


# ── 3. convert DiffCSP output lattice matrix → pymatgen Structure ─────────────
# sample() returns lattices as (B, 3, 3) Cartesian row-vectors;
# pymatgen's Lattice expects the same convention (row = lattice vector).

def traj_to_structures(traj, chem_symbols):
    from pymatgen.core import Structure, Lattice
    structures = []
    num_atoms_list = traj["num_atoms"].tolist()
    # traj_stack stacks all timesteps: shape (T+1, ...) ordered T→0, so [-1] = final
    atom_types  = traj["atom_types"][-1]        # (N_total,)  argmax'd + 1 in traj_stack
    frac_coords = traj["all_frac_coords"][-1]   # (N_total, 3)
    lattices    = traj["all_lattices"][-1]      # (B, 3, 3)

    offset = 0
    for i, n in enumerate(num_atoms_list):
        latt  = Lattice(lattices[i].cpu().numpy())
        specs = [chem_symbols[z - 1] for z in atom_types[offset:offset + n].tolist()]
        frac  = frac_coords[offset:offset + n].cpu().numpy()
        try:
            s = Structure(latt, specs, frac)
            structures.append(s)
        except Exception as e:
            print(f"  crystal {i} invalid: {e}")
        offset += n
    return structures


# ── 4. save CIFs and visualise in notebook with py3Dmol ───────────────────────

def save_cifs(structures, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, s in enumerate(structures):
        p = os.path.join(out_dir, f"gen_{i:04d}.cif")
        s.to(filename=p)
        paths.append(p)
    print(f"Saved {len(paths)} CIFs → {out_dir}/")
    return paths


def view_in_notebook(structures, max_show=6):
    """Render structures in the Kaggle notebook using py3Dmol."""
    try:
        import py3Dmol
    except ImportError:
        print("pip install py3Dmol  — then re-run this cell")
        return

    for i, s in enumerate(structures[:max_show]):
        cif_str = s.to(fmt="cif")
        view = py3Dmol.view(width=500, height=400)
        view.addModel(cif_str, "cif")
        view.setStyle({"sphere": {"radius": 0.4}, "stick": {"radius": 0.1}})
        view.addUnitCell()
        view.zoomTo()
        print(f"\n── structure {i}  ({s.composition.formula}) ──")
        view.show()


# ── run ───────────────────────────────────────────────────────────────────────
from pymatgen.core.periodic_table import Element

# atomic number → symbol lookup (Z=1 → "H", ..., Z=100 → "Fm")
CHEM_SYMBOLS = [Element.from_Z(z).symbol for z in range(1, 101)]

N_GENERATE = 20          # number of crystals to sample
STEP_LR    = 5e-6        # Langevin step size (from DiffCSP mp_20 recommendation)

model = load_finetuned(PRETRAINED_RUN, FINETUNED_CKPT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch = make_gen_batch(N_GENERATE).to(device)

print(f"Sampling {N_GENERATE} crystals on {device}…")
with torch.no_grad():
    traj0, traj_stack = model.sample(batch, step_lr=STEP_LR)

structures = traj_to_structures(traj_stack, CHEM_SYMBOLS)
print(f"Valid structures: {len(structures)} / {N_GENERATE}")

cif_paths = save_cifs(structures, OUT_DIR)
view_in_notebook(structures, max_show=6)
