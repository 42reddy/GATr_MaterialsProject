import os
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from diffcsp_data import load_pretrained_diffcsp

# ── fine-tuning the pretrained generator on the Na-electrolyte subset ──
#
# CSPDiffusion learns the score of a diffusion process over crystals
#   x = (l, f, t)   — lattice l, fractional coordinates f, atom types t
# i.e. it is trained to denoise (l, f, t) back towards the data distribution
# p_data. Fine-tuning is just continuing this same denoising objective,
# but with p_data replaced by our narrower Na-electrolyte distribution —
# so the score gets pulled toward the chemistries we care about.
#
# We reuse CSPDiffusion.training_step / validation_step verbatim (they already
# compute loss = loss_lattice + loss_coord + loss_type internally) and drive
# them with a plain optimizer loop instead of pl.Trainer, to keep things simple
# and transparent.

# resolved to absolute paths up front — load_pretrained_diffcsp chdir's into
# the DiffCSP repo as a side effect (it sets PROJECT_ROOT), which would
# otherwise break every relative path used afterwards
_HERE          = os.path.dirname(os.path.abspath(__file__))
DIFFCSP_REPO   = os.path.join(_HERE, "DiffCSP")
PRETRAINED_RUN = os.path.join(_HERE, "mp_gen")
CIF_DIR        = os.path.join(_HERE, "na_electrolyte_cifs")
CKPT_DIR       = os.path.join(_HERE, "checkpoints")

LR         = 1e-4     # small — we are fine-tuning, not training from scratch
EPOCHS     = 50
BATCH_SIZE = 32

# diffcsp.common.utils reads PROJECT_ROOT (and chdir's into it) the moment any
# diffcsp.* module is imported — set it before such an import happens anywhere,
# mirroring the .env file the authors expect (see DiffCSP/.env.template)
os.environ.setdefault("PROJECT_ROOT", DIFFCSP_REPO)
os.environ.setdefault("HYDRA_JOBS", DIFFCSP_REPO)
os.environ.setdefault("WABDB_DIR", DIFFCSP_REPO)


def cif_csv(cif_dir, split, out_csv):
    """
    CrystDataset reads a CSV with a 'cif' column — rebuild that view from the
    per-structure .cif files we saved, so we can reuse DiffCSP's own
    preprocessing (CIF → graph tensors) rather than duplicating it.
    """
    import pandas as pd
    rows = []
    split_dir = os.path.join(cif_dir, split)
    for fname in sorted(os.listdir(split_dir)):
        if not fname.endswith(".cif"):
            continue
        with open(os.path.join(split_dir, fname)) as f:
            cif_str = f.read()
        rows.append({
            "material_id": fname.removesuffix(".cif"),
            "formula": "",
            "cif": cif_str,
            # CrystDataset needs a `prop` column to build the scaler/target —
            # the diffusion objective itself is unconditional, so the value
            # is irrelevant; zero is the simplest placeholder.
            "formation_energy_per_atom": 0.0,
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def build_dataset(diffcsp_repo, csv_path, save_path):
    """
    Wrap our CSV in DiffCSP's CrystDataset — this performs exactly the same
    CIF → (frac_coords, atom_types, lengths, angles, edges) preprocessing the
    pretrained model was originally trained on, so fine-tuning sees data in
    the same representation.
    """
    import sys
    sys.path.append(diffcsp_repo)
    from diffcsp.pl_data.dataset import CrystDataset
    from diffcsp.common.data_utils import StandardScalerTorch, get_scaler_from_data_list

    dataset = CrystDataset(
        name="na_electrolyte",
        path=csv_path,
        prop="formation_energy_per_atom",
        niggli=True,
        primitive=False,
        graph_method="crystalnn",
        preprocess_workers=4,
        lattice_scale_method="scale_length",
        save_path=save_path,
        tolerance=0.1,
        use_space_group=False,
        use_pos_index=False,
    )
    # the diffusion loss never reads `prop`, but __getitem__ calls
    # self.scaler.transform(...) unconditionally — give it an identity scaler
    dataset.lattice_scaler = get_scaler_from_data_list(dataset.cached_data, "scaled_lattice")
    dataset.scaler = StandardScalerTorch(means=torch.tensor(0.0), stds=torch.tensor(1.0))
    return dataset


def finetune(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, ckpt_dir=CKPT_DIR):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.lattice_scaler = train_loader.dataset.lattice_scaler
    model.scaler         = train_loader.dataset.scaler

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val = float("inf")
    for epoch in tqdm(range(epochs), desc="Fine-tuning DiffCSP"):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.training_step(batch, 0)
            if loss is None:        # NaN guard already inside training_step
                continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                val_loss += out["loss"].item()
        val_loss /= len(val_loader)

        tqdm.write(f"Epoch {epoch+1:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       os.path.join(ckpt_dir, "diffcsp_na_electrolyte_best.pt"))


# ── run ──
if __name__ == "__main__":
    # rebuild the CSVs CrystDataset expects from the CIFs already on disk
    train_csv = cif_csv(CIF_DIR, "train", os.path.join(_HERE, "na_electrolyte_train.csv"))
    val_csv   = cif_csv(CIF_DIR, "val",   os.path.join(_HERE, "na_electrolyte_val.csv"))

    train_set = build_dataset(DIFFCSP_REPO, train_csv, os.path.join(_HERE, "na_electrolyte_train_ori.pt"))
    val_set   = build_dataset(DIFFCSP_REPO, val_csv,   os.path.join(_HERE, "na_electrolyte_val_ori.pt"))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    generator, _ = load_pretrained_diffcsp(DIFFCSP_REPO, PRETRAINED_RUN)
    finetune(generator, train_loader, val_loader)
