from mp_api.client import MPRester

# Replace with your actual API Key
API_KEY = "F4uWIvwrYTXLDQQWW4V8A41HbxuV13fW"

with MPRester(API_KEY) as mpr:
    # We search for:
    # 1. Elements: Must contain Lithium
    # 2. Band Gap: Between 0.5 and 10 eV (to ensure it's an insulator)
    # 3. Energy Above Hull: < 0.1 eV/atom (to ensure the crystal is stable)

    docs = mpr.summary.search(
        elements=["Li"],
        band_gap=(0.5, 10.0),
        energy_above_hull=(0, 0.1),
        fields=["material_id", "structure", "band_gap", "formation_energy_per_atom", "density"]
    )

    print(f"Found {len(docs)} materials matching your criteria.")
