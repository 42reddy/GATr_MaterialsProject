import os
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

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
# them with a plain optimizer loop instead of pl.Trainer.

# ── Kaggle paths ──
# __file__ is not defined in notebook kernels; use hardcoded Kaggle paths.
# Layout assumed:
#   /kaggle/working/DiffCSP/            ← git clone https://github.com/jiaor17/DiffCSP
#   /kaggle/input/<your-dataset>/mp_gen/ ← pretrained checkpoint folder
#   /kaggle/working/na_electrolyte_cifs/ ← CIF splits (from diffcsp_data.py)
#   /kaggle/working/diffcsp_data.py      ← uploaded as a utility script
WORKING       = "/kaggle/working"
DIFFCSP_REPO  = os.path.join(WORKING, "DiffCSP")
PRETRAINED_RUN = "/kaggle/input/mp-gen/mp_gen"   # ← adjust dataset slug as needed
CIF_DIR       = os.path.join(WORKING, "na_electrolyte_cifs")
CKPT_DIR      = os.path.join(WORKING, "checkpoints")

LR         = 1e-4
EPOCHS     = 50
BATCH_SIZE = 32

# set before any diffcsp.* import — diffcsp.common.utils chdir's into PROJECT_ROOT
os.environ["PROJECT_ROOT"] = DIFFCSP_REPO
os.environ.setdefault("HYDRA_JOBS", DIFFCSP_REPO)
os.environ.setdefault("WABDB_DIR",  DIFFCSP_REPO)

import sys
sys.path.append(DIFFCSP_REPO)
sys.path.append(os.path.join(DIFFCSP_REPO, "scripts"))
sys.path.append(WORKING)   # so `import diffcsp_data` finds the uploaded script

# PyTorch 2.6 changed torch.load default to weights_only=True, which breaks
# DiffCSP's internal cache loading (it pickles numpy arrays into .pt files).
# Patch globally so every torch.load call in DiffCSP falls back to the old behaviour.
# Use the underlying C-extension function directly to avoid re-patching on cell re-run.
import torch.serialization as _ts
_torch_load_real = _ts.load   # always the unwrapped original, never our lambda
torch.load = lambda *args, **kwargs: _torch_load_real(*args, **{**kwargs, "weights_only": False})

from diffcsp_data import load_pretrained_diffcsp


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
            "formation_energy_per_atom": 0.0,
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def build_dataset(diffcsp_repo, csv_path, save_path):
    """
    Wrap our CSV in DiffCSP's CrystDataset — identical preprocessing to
    the original training run, so fine-tuning sees data in the same form.
    """
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
    dataset.lattice_scaler = get_scaler_from_data_list(dataset.cached_data, "scaled_lattice")
    dataset.scaler = StandardScalerTorch(means=torch.tensor(0.0), stds=torch.tensor(1.0))
    return dataset


def finetune(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, ckpt_dir=CKPT_DIR):
    os.makedirs(ckpt_dir, exist_ok=True)
    # keep the pretrained lattice_scaler — it was fit on all of MP and the
    # decoder weights were trained against it; replacing it with a scaler from
    # our small dataset shifts the input distribution and degrades sampling.
    # Only override the property scaler (unconditional diffusion ignores it).
    model.scaler = train_loader.dataset.scaler

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)

    best_val = float("inf")
    for epoch in tqdm(range(epochs), desc="Fine-tuning DiffCSP"):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.training_step(batch, 0)
            if loss is None:
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
                val_loss += model(batch)["loss"].item()
        val_loss /= len(val_loader)

        tqdm.write(f"Epoch {epoch+1:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       os.path.join(ckpt_dir, "diffcsp_na_electrolyte_best.pt"))


# ── run (paste each block as a notebook cell, or run the whole file) ──
train_csv = cif_csv(CIF_DIR, "train", os.path.join(WORKING, "na_electrolyte_train.csv"))
val_csv   = cif_csv(CIF_DIR, "val",   os.path.join(WORKING, "na_electrolyte_val.csv"))

train_set = build_dataset(DIFFCSP_REPO, train_csv, os.path.join(WORKING, "na_electrolyte_train_ori.pt"))
val_set   = build_dataset(DIFFCSP_REPO, val_csv,   os.path.join(WORKING, "na_electrolyte_val_ori.pt"))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

generator, _ = load_pretrained_diffcsp(DIFFCSP_REPO, PRETRAINED_RUN)
finetune(generator, train_loader, val_loader)
