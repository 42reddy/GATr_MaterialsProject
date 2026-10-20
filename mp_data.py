import numpy as np
import torch
from mp_api.client import MPRester
from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")

# ── expanded electronegativity for MP's diverse chemistry ──
ELECTRONEGATIVITY = {
    'H': 2.20, 'Li': 0.98, 'Na': 0.93, 'K': 0.82, 'Rb': 0.82, 'Cs': 0.79,
    'Be': 1.57, 'Mg': 1.31, 'Ca': 1.00, 'Sr': 0.95, 'Ba': 0.89,
    'Al': 1.61, 'Si': 1.90, 'Ge': 2.01, 'Sn': 1.96, 'Pb': 2.33,
    'O': 3.44,  'F': 3.98, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
    'N': 3.04,  'P': 2.19, 'As': 2.18, 'Sb': 2.05, 'Bi': 2.02,
    'S': 2.58,  'Se': 2.55, 'Te': 2.10,
    'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Mn': 1.55, 'Cr': 1.66, 'V': 1.63,  'Ti': 1.54, 'Sc': 1.36,
    'Zr': 1.33, 'Hf': 1.30, 'Nb': 1.60, 'Mo': 2.16, 'W': 2.36,
    'La': 1.10, 'Ce': 1.12, 'Nd': 1.14, 'Sm': 1.17, 'Gd': 1.20,
    'Y': 1.22,  'Ga': 1.81, 'In': 1.78, 'Tl': 1.62,
    'Ag': 1.93, 'Au': 2.54, 'Pt': 2.28, 'Pd': 2.20,
    'Ta': 1.50, 'Re': 1.90, 'Ir': 2.20, 'Os': 2.20, 'Ru': 2.20,
}

def get_atom_features(symbol: str, occupancy: float) -> np.ndarray:
    pt = Chem.GetPeriodicTable()
    features = [
        pt.GetAtomicNumber(symbol) / 118.0,
        pt.GetAtomicWeight(symbol) / 250.0,
        pt.GetNOuterElecs(symbol) / 8.0,
        pt.GetRvdw(symbol) / 4.0,
        pt.GetRcovalent(symbol) / 3.0,
        ELECTRONEGATIVITY.get(symbol, 0.0) / 4.0,
        occupancy,
    ]
    return np.array(features, dtype=np.float32)

def get_all_atom_features(species, occupancies):
    return np.array([get_atom_features(s, o) for s, o in zip(species, occupancies)])

def process_mp_entry(doc, cutoff=3.0):
    structure = doc.structure

    # ── sites ──
    positions, species, occupancies = [], [], []
    for site in structure:
        element, occ = list(site.species.items())[0]
        positions.append(site.coords)
        species.append(element.symbol)
        occupancies.append(float(occ))

    positions   = np.array(positions)
    occupancies = np.array(occupancies)

    # ── neighbor graph ──
    all_neighbors = structure.get_all_neighbors(cutoff)
    src, dst, displacements = [], [], []
    for i, neighbors in enumerate(all_neighbors):
        for nbr in neighbors:
            src.append(i)
            dst.append(int(nbr.index))
            displacements.append(nbr.coords - positions[i])

    if len(src) == 0:
        return None

    # ── lattice ──
    lattice_matrix = structure.lattice.matrix
    lattice_volume = structure.lattice.volume
    lattice_angles = structure.lattice.angles
    atom_features  = get_all_atom_features(species, occupancies)

    # ── single target: formation energy per atom ──
    target = float(doc.formation_energy_per_atom) if doc.formation_energy_per_atom is not None else None

    if target is None:
        return None

    return {
        'atom_features'  : atom_features,
        'positions'      : positions,
        'lattice_matrix' : lattice_matrix,
        'lattice_volume' : lattice_volume,
        'lattice_angles' : np.array(lattice_angles, dtype=np.float32),
        'src'            : src,
        'dst'            : dst,
        'displacements'  : np.array(displacements),
        'targets'        : target,
        'formula'        : doc.formula_pretty,
        'id'             : doc.material_id,
    }



def fetch_and_process_mp(api_key, cutoff=3.0, max_structures=5000, chunk_size=5000):
    """
    Fetch structures from Materials Project, convert to PyG graphs on-the-fly,
    and save in chunks of chunk_size to avoid holding the full dataset in RAM.

    Returns: (n_chunks, total_saved) so the caller can merge the chunk files.
    Chunk files are named mp_data_chunk_0.pt, mp_data_chunk_1.pt, …
    """
    from GA_encoder import prepare_graph_data, dict_to_pyg_data

    print("Fetching from Materials Project...")
    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(
            fields=[
                "material_id",
                "structure",
                "formation_energy_per_atom",
                "formula_pretty",
                "nelements",
            ],
            formation_energy_per_atom=(-5.0, 1.0),
        )

    print(f"Fetched {len(docs)} structures")

    chunk_buf, chunk_idx, total, skipped = [], 0, 0, []

    for i, doc in enumerate(docs[:max_structures]):
        try:
            if doc.formation_energy_per_atom is None:
                continue
            result = process_mp_entry(doc, cutoff=cutoff)
            if result is not None:
                # convert immediately — never accumulate raw dicts
                chunk_buf.append(dict_to_pyg_data(prepare_graph_data(result)))
                total += 1

                if len(chunk_buf) == chunk_size:
                    torch.save(chunk_buf, f'mp_data_chunk_{chunk_idx}.pt')
                    print(f"  Saved chunk {chunk_idx} ({chunk_size} graphs)")
                    chunk_buf, chunk_idx = [], chunk_idx + 1

        except Exception as e:
            skipped.append((doc.material_id, str(e)))

        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1}/{min(len(docs), max_structures)} | "
                  f"Success: {total} | Skipped: {len(skipped)}")

    if chunk_buf:                                    # flush final partial chunk
        torch.save(chunk_buf, f'mp_data_chunk_{chunk_idx}.pt')
        print(f"  Saved chunk {chunk_idx} ({len(chunk_buf)} graphs)")
        chunk_idx += 1

    print(f"\nDone. {total} graphs in {chunk_idx} chunks, {len(skipped)} skipped")
    return chunk_idx, total


# ── run ──
API_KEY    = "F4uWIvwrYTXLDQQWW4V8A41HbxuV13fW"   # get from materialsproject.org
n_chunks, _ = fetch_and_process_mp(API_KEY, cutoff=3.0, max_structures=150000)

# Merge chunk files into mp_data.pt — loads one chunk at a time to cap RAM usage
mp_graph_data = []
for k in range(n_chunks):
    mp_graph_data.extend(torch.load(f'mp_data_chunk_{k}.pt', weights_only=False))

torch.save(mp_graph_data, 'mp_data.pt')
print(f"MP PyG dataset: {len(mp_graph_data)} graphs")

