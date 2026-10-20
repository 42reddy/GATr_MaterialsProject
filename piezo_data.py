"""
Piezoelectric Data Pipeline for Materials Project
==================================================
Corrected API usage:
  - has_props filter uses HasProps enum on mpr.materials.summary.search()
  - Full tensors (piezo 3×6, dielectric 3×3) come from dedicated endpoints:
      mpr.materials.piezoelectric  →  piezo tensors per material_id
      mpr.materials.dielectric     →  dielectric tensors per material_id
  - Summary endpoint used only to get IDs + scalar metadata (band_gap, symmetry)

Targets:
  'targets'          – max absolute element of Voigt piezo tensor  (C/m²)
  'piezo_tensor'     – (18,) flattened normalised Voigt 3×6
  'dielectric_elec'  – (9,)  electronic dielectric tensor
  'dielectric_ionic' – (9,)  ionic dielectric tensor
  'band_gap'         – scalar eV (raw)
  'graph_attrs'      – (8,) crystal-system one-hot + band_gap

Uses your existing prepare_graph_data and dict_to_pyg_data exactly.
"""

import numpy as np
import torch
from mp_api.client import MPRester
from emmet.core.summary import HasProps          # ← correct import for has_props enum
from rdkit import Chem

from GA_encoder import prepare_graph_data, dict_to_pyg_data


# ── electronegativity table (identical to forward model) ──────────────────────
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

CRYSTAL_SYSTEMS = [
    'triclinic', 'monoclinic', 'orthorhombic',
    'tetragonal', 'trigonal', 'hexagonal', 'cubic'
]


# ── feature helpers ────────────────────────────────────────────────────────────
def get_atom_features(symbol: str, occupancy: float) -> np.ndarray:
    pt = Chem.GetPeriodicTable()
    return np.array([
        pt.GetAtomicNumber(symbol) / 118.0,
        pt.GetAtomicWeight(symbol) / 250.0,
        pt.GetNOuterElecs(symbol) / 8.0,
        pt.GetRvdw(symbol) / 4.0,
        pt.GetRcovalent(symbol) / 3.0,
        ELECTRONEGATIVITY.get(symbol, 0.0) / 4.0,
        occupancy,
    ], dtype=np.float32)

def get_all_atom_features(species, occupancies):
    return np.array([get_atom_features(s, o) for s, o in zip(species, occupancies)])

def crystal_system_onehot(crystal_system) -> np.ndarray:
    vec = np.zeros(len(CRYSTAL_SYSTEMS), dtype=np.float32)
    cs  = str(crystal_system).lower() if crystal_system else ''
    if cs in CRYSTAL_SYSTEMS:
        vec[CRYSTAL_SYSTEMS.index(cs)] = 1.0
    return vec

def flatten_piezo_tensor(voigt) -> np.ndarray:
    """3×6 Voigt → (18,) normalised by 50 C/m²."""
    if voigt is None:
        return np.zeros(18, dtype=np.float32)
    return (np.array(voigt, dtype=np.float32).reshape(-1) / 50.0)

def max_piezo_modulus(voigt):
    if voigt is None:
        return None
    return float(np.max(np.abs(np.array(voigt, dtype=np.float32))))

def flatten_dielectric(tensor_3x3) -> np.ndarray:
    """3×3 → (9,) normalised by 100."""
    if tensor_3x3 is None:
        return np.zeros(9, dtype=np.float32)
    return (np.array(tensor_3x3, dtype=np.float32).reshape(-1) / 100.0)


# ── core fetch ────────────────────────────────────────────────────────────────
def fetch_and_process_piezo(api_key: str, cutoff: float = 3.0):
    """
    3-step fetch strategy
    ─────────────────────
    Step 1  summary.search(has_props=[HasProps.piezoelectric, HasProps.dielectric])
            → get material_ids + band_gap + symmetry + structure (scalar metadata)

    Step 2  mpr.materials.piezoelectric.search(material_ids=[...])
            → full 3×6 Voigt tensors

    Step 3  mpr.materials.dielectric.search(material_ids=[...])
            → full 3×3 electronic + ionic tensors

    Then join everything on material_id and build the graph dicts.
    """
    with MPRester(api_key) as mpr:

        # ── Step 1: IDs with both piezo AND dielectric data, non-metals only ──
        print("Step 1/3  fetching material IDs from summary...")
        summary_docs = mpr.materials.summary.search(
            has_props=[HasProps.piezoelectric, HasProps.dielectric],
            band_gap=(0.01, None),          # exclude metals (can't be piezoelectric)
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "symmetry",
                "structure",
            ],
        )
        print(f"          {len(summary_docs)} candidates found")

        # build lookup: mpid → summary doc
        summary_map = {str(doc.material_id): doc for doc in summary_docs}
        all_ids     = list(summary_map.keys())

        # ── Step 2: piezoelectric tensors ──────────────────────────────────────
        # available fields: 'total', 'ionic', 'electronic', 'e_ij_max', etc.
        # the doc itself IS the piezo data; there is no nested .piezoelectric
        print("Step 2/3  fetching piezoelectric tensors...")
        piezo_docs = mpr.materials.piezoelectric.search(
            material_ids=all_ids,
            fields=["material_id", "total"],
        )
        piezo_map = {str(d.material_id): d for d in piezo_docs}
        print(f"          {len(piezo_map)} piezo tensors retrieved")

        # ── Step 3: dielectric tensors ─────────────────────────────────────────
        # available fields: 'e_electronic', 'e_ionic', 'e_total', etc.
        print("Step 3/3  fetching dielectric tensors...")
        diel_docs = mpr.materials.dielectric.search(
            material_ids=all_ids,
            fields=["material_id", "e_electronic", "e_ionic"],
        )
        diel_map = {str(d.material_id): d for d in diel_docs}
        print(f"          {len(diel_map)} dielectric tensors retrieved")

    # ── build graph dicts (outside the MPRester context) ──────────────────────
    print("\nBuilding graph dicts...")
    processed, skipped = [], 0

    for mpid, sdoc in summary_map.items():
        try:
            piezo_obj = piezo_map.get(mpid)
            diel_obj  = diel_map.get(mpid)

            # doc.total is the 3×6 Voigt tensor directly
            voigt     = piezo_obj.total if piezo_obj is not None else None
            max_piezo = max_piezo_modulus(voigt)

            if max_piezo is None:
                skipped += 1
                continue

            structure = sdoc.structure

            # ── sites ──────────────────────────────────────────────────────────
            positions, species, occupancies = [], [], []
            for site in structure:
                element, occ = list(site.species.items())[0]
                positions.append(site.coords)
                species.append(element.symbol)
                occupancies.append(float(occ))

            positions   = np.array(positions,   dtype=np.float32)
            occupancies = np.array(occupancies, dtype=np.float32)

            # ── neighbour graph ────────────────────────────────────────────────
            all_neighbors = structure.get_all_neighbors(cutoff)
            src, dst, displacements = [], [], []
            for i, neighbors in enumerate(all_neighbors):
                for nbr in neighbors:
                    src.append(i)
                    dst.append(int(nbr.index))
                    displacements.append(nbr.coords - positions[i])

            if len(src) == 0:
                skipped += 1
                continue

            # ── tensors ────────────────────────────────────────────────────────
            piezo_tensor_flat = flatten_piezo_tensor(voigt)

            diel_elec  = flatten_dielectric(
                diel_obj.e_electronic if diel_obj is not None else None
            )
            diel_ionic = flatten_dielectric(
                diel_obj.e_ionic if diel_obj is not None else None
            )

            # ── graph-level conditioning features ──────────────────────────────
            band_gap   = float(sdoc.band_gap) if sdoc.band_gap is not None else 0.0
            cry_system = crystal_system_onehot(
                sdoc.symmetry.crystal_system if sdoc.symmetry is not None else None
            )
            graph_attrs = np.concatenate([
                cry_system,                                         # (7,)
                np.array([band_gap / 10.0], dtype=np.float32),     # (1,)
            ])                                                      # (8,)

            processed.append({
                # identical keys to forward model ──────────────────────────────
                'atom_features'  : get_all_atom_features(species, occupancies),
                'positions'      : positions,
                'lattice_matrix' : structure.lattice.matrix,
                'lattice_volume' : structure.lattice.volume,
                'lattice_angles' : np.array(structure.lattice.angles, dtype=np.float32),
                'src'            : src,
                'dst'            : dst,
                'displacements'  : np.array(displacements, dtype=np.float32),
                'formula'        : sdoc.formula_pretty,
                'id'             : mpid,
                # primary scalar target ────────────────────────────────────────
                'targets'        : max_piezo,           # C/m²
                # extended targets ─────────────────────────────────────────────
                'piezo_tensor'   : piezo_tensor_flat,   # (18,)
                'dielectric_elec': diel_elec,           # (9,)
                'dielectric_ionic': diel_ionic,         # (9,)
                'band_gap'       : band_gap,            # eV raw
                'graph_attrs'    : graph_attrs,         # (8,)
            })

        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  [skip] {mpid}: {e}")

        if len(processed) % 500 == 0 and len(processed) > 0:
            print(f"  processed: {len(processed)} | skipped: {skipped}")

    print(f"\nDone.  Processed: {len(processed)}  |  Skipped: {skipped}")
    return processed


# ── PyG conversion (reuses your functions exactly) ────────────────────────────
def build_pyg_dataset(processed_list):
    pyg_graphs = []
    for entry in processed_list:
        graph_dict = prepare_graph_data(entry)
        # attach extended targets as free attributes on the PyG Data object
        graph_dict['piezo_tensor']     = entry['piezo_tensor']
        graph_dict['dielectric_elec']  = entry['dielectric_elec']
        graph_dict['dielectric_ionic'] = entry['dielectric_ionic']
        graph_dict['band_gap']         = entry['band_gap']
        graph_dict['graph_attrs']      = entry['graph_attrs']
        pyg_graphs.append(dict_to_pyg_data(graph_dict))
    return pyg_graphs


# ── sanity check ──────────────────────────────────────────────────────────────
def print_sample(processed_list, n=3):
    for entry in processed_list[:n]:
        print(f"\n{'─'*60}")
        print(f"  ID            : {entry['id']}")
        print(f"  Formula       : {entry['formula']}")
        print(f"  Atoms / Edges : {entry['atom_features'].shape[0]} / {len(entry['src'])}")
        print(f"  max_piezo     : {entry['targets']:.4f} C/m²")
        print(f"  band_gap      : {entry['band_gap']:.3f} eV")
        print(f"  piezo_tensor  : {entry['piezo_tensor'][:6]}  ...")
        print(f"  crystal_sys   : {entry['graph_attrs'][:7]}  (one-hot)")



API_KEY = "F4uWIvwrYTXLDQQWW4V8A41HbxuV13fW"

piezo_processed = fetch_and_process_piezo(API_KEY, cutoff=3.0)

print_sample(piezo_processed)

torch.save(piezo_processed, 'piezo_processed_data')
print(f"\nSaved {len(piezo_processed)} raw dicts → piezo_processed_data")

piezo_graph_data = build_pyg_dataset(piezo_processed)
torch.save(piezo_graph_data, 'piezo_data.pt')
print(f"Saved {len(piezo_graph_data)} PyG graphs  → piezo_data.pt")

targets = [e['targets'] for e in piezo_processed]
print(f"\nmax_piezo stats →  min: {min(targets):.3f} "
        f" mean: {np.mean(targets):.3f} "
        f" max: {max(targets):.3f}  C/m²")
