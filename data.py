import numpy as np
import pandas as pd
import torch
from obelix import OBELiX
import py3Dmol
from rdkit import Chem
from pymatgen.io.cif import CifWriter
import warnings
warnings.filterwarnings("ignore")


ob = OBELiX()

# Filter for entries that have CIF files
dataset_with_structures = ob.with_cifs()
print(f"Total entries with full structure: {len(dataset_with_structures)}")

# Sample conductivity distribution
conductivities = [entry['Ionic conductivity (S cm-1)'] for entry in dataset_with_structures]
log_sigma = np.log10(conductivities)




def visualize_obelix_entry(entry):
    # Extract the Pymatgen Structure object
    struct = entry['structure']

    # Convert Structure object to a CIF string
    writer = CifWriter(struct)
    cif_str = str(writer)

    # Render using 3Dmol.js
    view = py3Dmol.view(width=500, height=400)
    view.addModel(cif_str, "cif")
    view.setStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.3},
                   'stick': {'colorscheme': 'Jmol', 'radius': 0.15}})
    view.addUnitCell()
    view.zoomTo()
    return view.show()


# Visualize the 'oi6' entry (Li7SbO6)
visualize_obelix_entry(dataset_with_structures[1])


def extract_structure_data(entry, cutoff=3):
    positions = []
    species = []
    occupancies = []

    structure = entry['structure']

    for site in structure:
        comp = site.species

        # get dominant element (first in composition)
        element, occ = list(comp.items())[0]  # ← we discussed this

        positions.append(site.coords)
        species.append(element.symbol)
        occupancies.append(occ)

    positions = np.array(positions)  # shape (N, 3)
    occupancies = np.array(occupancies)  # shape (N,)

    all_neighbors = structure.get_all_neighbors(cutoff)

    src, dst, displacements = [],[],[]

    for i, neighbors in enumerate(all_neighbors):
        for nbr in neighbors:
            src.append(i)
            dst.append(int(nbr.index))  # index of neighbor atom j
            displacements.append(nbr.coords - positions[i])  # vector from i → j

    lattice_matrix = structure.lattice.matrix
    lattice_volume = structure.lattice.volume
    lattice_angles = structure.lattice.angles

    atom_features = get_all_atom_features(species, occupancies)

    y = y = np.log10(entry['Ionic conductivity (S cm-1)'])


    return {
        'atom_features'  : atom_features,
        'positions'      : positions,
        'lattice_matrix' : lattice_matrix,
        'lattice_volume' : lattice_volume,
        'lattice_angles' : lattice_angles,
        'src'            : src,
        'dst'            : dst,
        'displacements'  : np.array(displacements),
        'y'              : y,
        'formula'        : entry['Reduced Composition'],
        'id'             : entry['ID'],
        'space group'    : entry['Space group']
    }




# electronegativity lookup (RDKit doesn't have this built-in)
ELECTRONEGATIVITY = {
    'H': 2.20, 'Li': 0.98, 'Na': 0.93, 'K': 0.82,
    'Mg': 1.31, 'Ca': 1.00, 'Al': 1.61, 'Si': 1.90,
    'O': 3.44, 'F': 3.98, 'Cl': 3.16, 'S': 2.58,
    'N': 3.04, 'P': 2.19, 'Fe': 1.83, 'Bi': 2.02,
    'La': 1.10, 'Ge': 2.01, 'Ga': 1.81, 'Zn': 1.65,
    # add more as you discover new elements in your dataset
}


def get_atom_features(symbol: str, occupancy: float) -> np.ndarray:
    """
    Returns a 1D array of scalar chemical features for one atom.
    These will become grade-0 components of the GA multivector.
    """
    pt = Chem.GetPeriodicTable()

    features = [
        pt.GetAtomicNumber(symbol) / 118.0,  # normalized Z
        pt.GetAtomicWeight(symbol) / 250.0,  # normalized weight
        pt.GetNOuterElecs(symbol) / 8.0,  # normalized valence
        pt.GetRvdw(symbol) / 4.0,  # normalized vdW radius
        pt.GetRcovalent(symbol) / 3.0,  # normalized covalent radius
        ELECTRONEGATIVITY.get(symbol, 0.0) / 4.0,  # normalized electronegativity
        occupancy,  # already in [0,1]
    ]

    return np.array(features, dtype=np.float32)  # shape (7,)


def get_all_atom_features(species: list, occupancies: np.ndarray) -> np.ndarray:
    """
    Applies get_atom_features to every atom in a structure.

    Returns:
        np.ndarray of shape (N, 7)
    """
    all_features = []

    for symbol, occ in zip(species, occupancies):  # ← fill this in
        features = get_atom_features(symbol, occ)  # ← call the function above
        all_features.append(features)

    return np.array(all_features)



processed_data = []

for entry in dataset_with_structures:

    processed_data.append(extract_structure_data(entry, 3))

torch.save(processed_data, 'processed_data')
