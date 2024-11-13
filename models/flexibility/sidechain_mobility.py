"""Sidechain mobility analysis module"""
import numpy as np
import mdtraj as md
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SidechainMobility:
    """Analyze protein sidechain mobility"""

    def __init__(self):
        self.trajectory = None
        self.topology = None

    def load_trajectory(self, traj_file: str, top_file: str):
        """Load trajectory for analysis"""
        try:
            self.trajectory = md.load(traj_file, top=top_file)
            self.topology = self.trajectory.topology
            logger.info(f"Loaded trajectory with {len(self.trajectory)} frames")
        except Exception as e:
            logger.error(f"Error loading trajectory: {e}")
            raise

    def calculate_chi_angles(self) -> Dict[str, np.ndarray]:
        """Calculate chi angles for all residues"""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")

        chi_angles = {}

        for residue in self.topology.residues:
            if residue.name not in ['GLY', 'ALA', 'PRO']:
                indices = self._get_chi_indices(residue)
                if indices:
                    angles = md.compute_dihedrals(self.trajectory, [indices])
                    chi_angles[f"{residue.name}{residue.resSeq}"] = angles

        return chi_angles

    def analyze_rotamer_populations(self, chi_angles: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
        """Analyze rotamer populations from chi angles"""
        rotamer_pops = {}

        for residue, angles in chi_angles.items():
            # Define rotamer bins (-180 to 180 degrees, 60-degree bins)
            bins = np.linspace(-np.pi, np.pi, 7)
            hist, _ = np.histogram(angles, bins=bins, density=True)
            rotamer_pops[residue] = hist.tolist()

        return rotamer_pops


    def _get_chi_indices(self, residue) -> Optional[List[int]]:
        """Get atom indices for chi angle calculation"""
        chi_atoms = {
            'ARG': ['N', 'CA', 'CB', 'CG'],
            'ASN': ['N', 'CA', 'CB', 'CG'],
            'ASP': ['N', 'CA', 'CB', 'CG'],
            'CYS': ['N', 'CA', 'CB', 'SG'],
            'GLN': ['N', 'CA', 'CB', 'CG'],
            'GLU': ['N', 'CA', 'CB', 'CG'],
            'HIS': ['N', 'CA', 'CB', 'CG'],
            'ILE': ['N', 'CA', 'CB', 'CG1'],
            'LEU': ['N', 'CA', 'CB', 'CG'],
            'LYS': ['N', 'CA', 'CB', 'CG'],
            'MET': ['N', 'CA', 'CB', 'CG'],
            'PHE': ['N', 'CA', 'CB', 'CG'],
            'SER': ['N', 'CA', 'CB', 'OG'],
            'THR': ['N', 'CA', 'CB', 'OG1'],
            'TRP': ['N', 'CA', 'CB', 'CG'],
            'TYR': ['N', 'CA', 'CB', 'CG'],
            'VAL': ['N', 'CA', 'CB', 'CG1']
        }

        if residue.name in chi_atoms:
            try:
                return [atom.index for atom in residue.atoms if atom.name in chi_atoms[residue.name]]
            except KeyError:
                return None
        return None
