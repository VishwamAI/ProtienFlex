"""Structure format conversion utilities"""
import logging
from typing import Dict, Any, Optional
import numpy as np
from Bio import PDB
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom

logger = logging.getLogger(__name__)

class StructureConverter:
    """Convert between different structure formats"""

    def __init__(self):
        self.parser = PDB.PDBParser(QUIET=True)
        self.io = PDBIO()

    def convert_structure(self, structure_data: Dict[str, Any],
                         output_format: str = 'pdb') -> Structure.Structure:
        """Convert structure to specified format"""
        try:
            if output_format.lower() == 'pdb':
                return self._to_pdb_structure(structure_data)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

        except Exception as e:
            logger.error(f"Error converting structure: {e}")
            raise

    def _to_pdb_structure(self, structure_data: Dict[str, Any]) -> Structure.Structure:
        """Convert to PDB structure format"""
        try:
            # Create new structure
            structure = Structure.Structure('converted_structure')
            model = Model.Model(0)
            chain = Chain.Chain('A')

            # Add residues and atoms
            for res_idx, residue_data in enumerate(structure_data['residues']):
                residue = Residue.Residue(
                    (' ', res_idx, ' '),
                    residue_data['name'],
                    ''
                )

                for atom_data in residue_data['atoms']:
                    atom = Atom.Atom(
                        atom_data['name'],
                        atom_data['coord'],
                        atom_data.get('bfactor', 0.0),
                        atom_data.get('occupancy', 1.0),
                        ' ',
                        atom_data['name'],
                        res_idx,
                        atom_data.get('element', 'C')
                    )
                    residue.add(atom)

                chain.add(residue)

            model.add(chain)
            structure.add(model)

            return structure

        except Exception as e:
            logger.error(f"Error converting to PDB structure: {e}")
            raise

    def save_structure(self, structure: Structure.Structure, output_file: str):
        """Save structure to file"""
        try:
            self.io.set_structure(structure)
            self.io.save(output_file)
            logger.info(f"Structure saved to {output_file}")

        except Exception as e:
            logger.error(f"Error saving structure: {e}")
            raise

    def load_structure(self, input_file: str) -> Structure.Structure:
        """Load structure from file"""
        try:
            structure = self.parser.get_structure('loaded_structure', input_file)
            logger.info(f"Structure loaded from {input_file}")
            return structure

        except Exception as e:
            logger.error(f"Error loading structure: {e}")
            raise

    def extract_coordinates(self, structure: Structure.Structure) -> np.ndarray:
        """Extract atom coordinates from structure"""
        try:
            coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            coords.append(atom.get_coord())
            return np.array(coords)

        except Exception as e:
            logger.error(f"Error extracting coordinates: {e}")
            raise
