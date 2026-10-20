import os
import numpy as np
import pandas as pd
import torch
from mp_api.client import MPRester
from pymatgen.core import Structure
import warnings
warnings.filterwarnings("ignore")

# ── chemistries relevant to Na-ion solid electrolytes ──
# Na must be present; the framework/anion species are restricted to elements
# that actually show up in known Na superionic conductors (NASICON Na3Zr2Si2PO12,
# Na3PS4, beta-alumina Na2O·xAl2O3, halide/closo-borate conductors, ...).
ELECTROLYTE_ELEMENTS = [
    "Na", "O", "S", "P", "Si", "Ge", "Cl", "F", "Br", "I", "B",
    "Al", "Zr", "Sc", "Y", "Ti", "Sn", "Hf",
]

# DiffCSP is trained/fine-tuned on near-equilibrium structures, so we keep only
# structures close to the convex hull (e_above_hull small ⇒ thermodynamically
# plausible ⇒ a sensible target distribution for the generator to learn).
E_HULL_MAX = 0.10   # eV/atom — relaxed from 0.05; keeps metastable phases that
                    # are still thermodynamically accessible as electrolytes


def fetch_electrolyte_structures(api_key, restrict_elements=False):
    """
    Pull Na-containing, near-hull structures from MP.

    restrict_elements=False (default): keep ALL Na structures within E_HULL_MAX —
        gives ~8-15k structures, enough to fine-tune a diffusion model.
        Element filtering is better done at generation time (post-sample) because
        training on a broader Na distribution gives the generator a stronger prior.
    restrict_elements=True: apply the ELECTROLYTE_ELEMENTS whitelist (old behaviour,
        yields ~880 structures — too small for fine-tuning).

    Returns a DataFrame with columns (material_id, formula, num_sites, cif).
    """
    print("Querying Materials Project for Na-containing structures…")
    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(
            elements=["Na"],
            fields=["material_id", "structure", "formula_pretty",
                    "energy_above_hull", "nsites"],
        )
    print(f"Fetched {len(docs)} candidates")

    rows = []
    for doc in docs:
        if doc.energy_above_hull is None or doc.energy_above_hull > E_HULL_MAX:
            continue
        if restrict_elements:
            species = {site.species.elements[0].symbol for site in doc.structure}
            if not species.issubset(set(ELECTROLYTE_ELEMENTS)):
                continue
        rows.append({
            "material_id": doc.material_id,
            "formula": doc.formula_pretty,
            "num_sites": doc.nsites,
            "cif": doc.structure.to(fmt="cif"),
        })

    df = pd.DataFrame(rows)
    print(f"Kept {len(df)} structures (restrict_elements={restrict_elements})")
    return df


def split_and_save(df, train_frac=0.8, val_frac=0.1, seed=0,
                    out_dir="na_electrolyte_cifs"):
    """
    Shuffle, split into train/val/test, and write one .cif file per structure
    under out_dir/{train,val,test}/<material_id>.cif — the raw-structure layout
    DiffCSP's preprocessing can ingest directly (one crystal per CIF file).
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(train_frac * len(df))
    n_val   = int(val_frac * len(df))

    splits = {
        "train": idx[:n_train],
        "val":   idx[n_train:n_train + n_val],
        "test":  idx[n_train + n_val:],
    }
    import shutil
    for name, ids in splits.items():
        split_dir = os.path.join(out_dir, name)
        # wipe any stale files from a previous run first — otherwise leftover
        # CIFs from a different split assignment silently leak across
        # train/val/test and contaminate the held-out sets
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)
        for _, row in df.iloc[ids].iterrows():
            path = os.path.join(split_dir, f"{row['material_id']}.cif")
            with open(path, "w") as f:
                f.write(row["cif"])
        print(f"  {name}: {len(ids):4d} structures → {split_dir}/")


def cif_dir_to_diffcsp_tensors(cif_dir):
    """
    Parse every .cif in cif_dir into the per-crystal tensors DiffCSP trains on:
    fractional coordinates, atomic numbers, lattice (lengths, angles), and the
    atom count — i.e. the (l, a, x, t) representation of a crystal as a point
    in (lattice space) × (fractional-coordinate torus) × (composition).

    Returns a list of dicts, one per structure, each holding plain tensors —
    DiffCSP's CrystDataset assembles exactly these fields per data point.
    """
    data = []
    for fname in sorted(os.listdir(cif_dir)):
        if not fname.endswith(".cif"):
            continue
        structure = Structure.from_file(os.path.join(cif_dir, fname))

        lengths = torch.tensor(structure.lattice.abc,    dtype=torch.float32)
        angles  = torch.tensor(structure.lattice.angles, dtype=torch.float32)
        frac_x  = torch.tensor(structure.frac_coords,    dtype=torch.float32)
        atom_z  = torch.tensor([site.specie.Z for site in structure],
                               dtype=torch.long)

        data.append({
            "material_id": fname.removesuffix(".cif"),
            "frac_coords": frac_x,                 # (N, 3), each entry in [0, 1)
            "atom_types":  atom_z,                 # (N,)    atomic numbers
            "lengths":     lengths,                # (3,)    a, b, c
            "angles":      angles,                 # (3,)    α, β, γ in degrees
            "num_atoms":   torch.tensor(len(structure)),
        })
    print(f"Parsed {len(data)} structures from {cif_dir} into DiffCSP tensors")
    return data


def load_pretrained_diffcsp(diffcsp_repo, run_dir):
    """
    Reconstruct the pretrained CSPDiffusion generator M_θ from a checkpoint
    directory (hparams.yaml + *.ckpt + scalers).

    Mirrors what eval_utils.load_model does, but calls
    CSPDiffusion.load_from_checkpoint directly on the class — newer
    PyTorch Lightning forbids calling it on an instance (even via type(obj)).
    """
    import sys
    import numpy as np
    from pathlib import Path

    repo_root   = os.path.abspath(diffcsp_repo)
    run_dir_abs = os.path.abspath(run_dir)
    os.environ.setdefault("PROJECT_ROOT", repo_root)
    os.environ.setdefault("HYDRA_JOBS", repo_root)
    os.environ.setdefault("WABDB_DIR", repo_root)

    sys.path.append(diffcsp_repo)
    sys.path.append(os.path.join(diffcsp_repo, "scripts"))

    import hydra
    from hydra.experimental import compose
    from hydra import initialize_config_dir
    from diffcsp.pl_modules.diffusion_w_type import CSPDiffusion

    run_path = Path(run_dir_abs)
    with initialize_config_dir(str(run_path)):
        cfg = compose(config_name="hparams")

    # pick checkpoint: prefer 'last', else highest epoch number
    ckpts = list(run_path.glob("*.ckpt"))
    ckpt = None
    for ck in ckpts:
        if "last" in ck.name:
            ckpt = str(ck)
    if ckpt is None:
        epochs = np.array([int(ck.name.split("-")[0].split("=")[1])
                           for ck in ckpts if "last" not in ck.name])
        ckpt = str(ckpts[epochs.argsort()[-1]])

    hparams = str(run_path / "hparams.yaml")
    # call on the class directly — never on an instance
    model = CSPDiffusion.load_from_checkpoint(ckpt, hparams_file=hparams, strict=False)

    try:
        model.lattice_scaler = torch.load(run_path / "lattice_scaler.pt")
        model.scaler         = torch.load(run_path / "prop_scaler.pt")
    except Exception:
        pass

    model.eval()
    print(f"Loaded pretrained DiffCSP generator from {run_dir}")
    return model, cfg


# ── run ──
# guarded so other scripts can `import diffcsp_data` for its functions
# without re-triggering the MP fetch / model load as a side effect
if __name__ == "__main__":
    API_KEY = "F4uWIvwrYTXLDQQWW4V8A41HbxuV13fW"   # get from materialsproject.org

    df = fetch_electrolyte_structures(API_KEY, restrict_elements=False)
    split_and_save(df, out_dir="na_electrolyte_cifs")

    # parse the saved structures into the tensor format DiffCSP trains on
    train_data = cif_dir_to_diffcsp_tensors("na_electrolyte_cifs/train")
    torch.save(train_data, "na_electrolyte_train_tensors.pt")

    # load the original pretrained generator, ready for fine-tuning on train_data
    DIFFCSP_REPO = "DiffCSP"            # local clone of jiaor17/DiffCSP
    PRETRAINED_RUN = "mp_gen"   # downloaded checkpoint folder (IonForge/mp_gen)
    generator, generator_cfg = load_pretrained_diffcsp(DIFFCSP_REPO, PRETRAINED_RUN)
