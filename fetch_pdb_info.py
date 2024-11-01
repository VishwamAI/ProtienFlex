import requests
import json
from typing import Dict, Any, Optional, List

def fetch_structure_info(pdb_id: str) -> Optional[Dict[str, Any]]:
    """Fetch basic structure information using RCSB REST API."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching structure info: {e}")
        return None

def fetch_assembly_info(pdb_id: str) -> Optional[Dict[str, Any]]:
    """Fetch assembly information using RCSB REST API."""
    url = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id}/1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching assembly info: {e}")
        return None

def fetch_ligand_info(pdb_id: str) -> List[Dict[str, Any]]:
    """Fetch information about ligands using RCSB REST API."""
    url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return [data[entity_id] for entity_id in data] if isinstance(data, dict) else []
    except Exception as e:
        print(f"Error fetching ligand info: {e}")
        return []

def create_structure_description(pdb_id: str) -> str:
    """Create a detailed description of the structure focusing on protein-glycan-ion interactions."""
    print(f"\nFetching information for PDB {pdb_id}...")

    # Fetch all required information
    structure_info = fetch_structure_info(pdb_id)
    assembly_info = fetch_assembly_info(pdb_id)
    ligand_info = fetch_ligand_info(pdb_id)

    if not structure_info:
        return "Error: Failed to fetch structure information"

    # Print raw data for debugging
    print("\nRaw Structure Data:")
    print(json.dumps(structure_info, indent=2))
    print("\nRaw Assembly Data:")
    print(json.dumps(assembly_info, indent=2))
    print("\nRaw Ligand Data:")
    print(json.dumps(ligand_info, indent=2))

    # Extract relevant information
    title = structure_info.get('struct', {}).get('title', 'Unknown')
    description = structure_info.get('struct', {}).get('pdbx_descriptor', 'No description available')

    # Get experimental details
    exp_method = structure_info.get('exptl', [{}])[0].get('method', 'Unknown method')
    resolution = structure_info.get('refine', [{}])[0].get('ls_d_res_high', 'Unknown')

    # Get entity information
    entity_info = structure_info.get('entity', [])
    protein_components = []
    for entity in entity_info:
        if entity.get('type') == 'polymer':
            desc = entity.get('pdbx_description', 'Unknown protein')
            protein_components.append(desc)

    # Get keywords and assembly details
    pdbx_keywords = structure_info.get('struct_keywords', {}).get('pdbx_keywords', 'Unknown')
    keyword_text = structure_info.get('struct_keywords', {}).get('text', '')
    assembly_details = assembly_info.get('rcsb_assembly_info', {}).get('polymer_composition', 'Unknown')

    # Create comprehensive description
    description = f"""Design a protein structure based on PDB entry {pdb_id} with the following characteristics:

Title: {title}
Primary Function: Pectate lyase B - catalyzes the cleavage of pectate/pectin
Resolution: {resolution} Å

Structure Overview:
- Class: {pdbx_keywords}
- Features: {keyword_text}
- Assembly: {assembly_details}

Key Structural Elements:
1. Core Architecture:
   - Predominantly beta-sheets forming parallel beta-helix
   - Specialized binding cleft for pectate/pectin substrate
   - Ca2+ binding sites essential for catalysis

2. Interaction Interfaces:
   - Glycan Recognition: Specific binding sites for pectate/pectin polymers
   - Ion Coordination: Ca2+ binding sites crucial for catalysis
   - Surface Features: Aromatic residues for carbohydrate stacking

Design Requirements:
1. Essential Elements:
   - Preserve parallel beta-helix core structure
   - Maintain calcium binding sites
   - Optimize substrate binding groove

2. Optimization Priorities:
   - Proper glycan recognition
   - Stable ion coordination
   - Catalytic efficiency
"""
    return description

if __name__ == "__main__":
    # Create description for PDB 7BBV
    description = create_structure_description("7BBV")
    print("\nFinal Structure Description:")
    print("="*50)
    print(description)
