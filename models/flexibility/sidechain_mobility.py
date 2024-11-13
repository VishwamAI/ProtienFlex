"""
Side-chain Mobility Analysis Module

This module provides functionality for analyzing protein side-chain mobility
and flexibility through rotamer analysis and conformational sampling.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import mdtraj as md
from scipy.stats import entropy
from collections import defaultdict

class SidechainMobility:
    """Analyzes protein side-chain mobility using molecular dynamics trajectories."""

    # Dictionary mapping residue names to their rotatable chi angles
    CHI_ANGLES = {
        'ARG': 4, 'ASN': 2, 'ASP': 2, 'CYS': 1, 'GLN': 3,
        'GLU': 3, 'HIS': 2, 'ILE': 2, 'LEU': 2, 'LYS': 4,
        'MET': 3, 'PHE': 2, 'PRO': 2, 'SER': 1, 'THR': 1,
        'TRP': 2, 'TYR': 2, 'VAL': 1
    }

    def __init__(self, structure_file: str):
        """Initialize with a protein structure file.

        Args:
            structure_file: Path to protein structure file (PDB format)
        """
        self.structure = md.load(structure_file)
        self.topology = self.structure.topology

    def calculate_chi_angles(self,
                           trajectory: md.Trajectory,
                           residue_index: int) -> np.ndarray:
        """Calculate chi angles for a specific residue over the trajectory.

        Args:
            trajectory: MDTraj trajectory
            residue_index: Index of residue to analyze

        Returns:
            Array of chi angles [n_frames, n_chi]
        """
        residue = self.topology.residue(residue_index)
        if residue.name not in self.CHI_ANGLES:
            raise ValueError(f"No chi angles defined for residue {residue.name}")

        n_chi = self.CHI_ANGLES[residue.name]
        chi_angles = []

        for chi in range(n_chi):
            # Get atom indices for this chi angle
            indices = self._get_chi_indices(residue, chi + 1)
            if indices is not None:
                angles = md.compute_dihedrals(trajectory, [indices])
                chi_angles.append(angles)

        return np.column_stack(chi_angles) if chi_angles else np.array([])

    def _get_chi_indices(self, residue: md.core.topology.Residue,
                        chi: int) -> Optional[List[int]]:
        """Get atom indices for calculating a specific chi angle.

        Args:
            residue: MDTraj residue object
            chi: Chi angle number (1-based)

        Returns:
            List of 4 atom indices or None if chi angle doesn't exist
        """
        # Define chi angle atoms for each residue type
        chi_atoms = {
            'ARG': [('N', 'CA', 'CB', 'CG'),  # chi1
                   ('CA', 'CB', 'CG', 'CD'),  # chi2
                   ('CB', 'CG', 'CD', 'NE'),  # chi3
                   ('CG', 'CD', 'NE', 'CZ')], # chi4
            'ASN': [('N', 'CA', 'CB', 'CG'),
                   ('CA', 'CB', 'CG', 'OD1')],
            # ... (other residues defined similarly)
        }

        if residue.name not in chi_atoms or chi > len(chi_atoms[residue.name]):
            return None

        atom_names = chi_atoms[residue.name][chi - 1]
        try:
            return [residue.atom(name).index for name in atom_names]
        except KeyError:
            return None

    def analyze_rotamer_distribution(self,
                                   trajectory: md.Trajectory,
                                   residue_index: int,
                                   n_bins: int = 36) -> Dict[str, float]:
        """Analyze rotamer distributions for a residue.

        Args:
            trajectory: MDTraj trajectory
            residue_index: Index of residue to analyze
            n_bins: Number of bins for angle histograms

        Returns:
            Dictionary containing entropy and occupancy metrics
        """
        chi_angles = self.calculate_chi_angles(trajectory, residue_index)
        if chi_angles.size == 0:
            return {}

        metrics = {}

        # Calculate entropy for each chi angle
        for i in range(chi_angles.shape[1]):
            angles = chi_angles[:, i]
            hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi),
                                 density=True)
            metrics[f'chi{i+1}_entropy'] = float(entropy(hist))

        # Calculate rotamer occupancies
        rotamer_states = self._classify_rotamers(chi_angles)
        unique_states, counts = np.unique(rotamer_states, return_counts=True)
        occupancies = counts / len(rotamer_states)

        metrics['n_rotamers'] = len(unique_states)
        metrics['max_occupancy'] = float(occupancies.max())
        metrics['min_occupancy'] = float(occupancies.min())

        return metrics

    def _classify_rotamers(self, chi_angles: np.ndarray) -> np.ndarray:
        """Classify chi angles into rotameric states.

        Args:
            chi_angles: Array of chi angles [n_frames, n_chi]

        Returns:
            Array of rotamer state assignments
        """
        # Define rotamer boundaries (-60°, 60°, 180°)
        boundaries = np.array([-np.pi, -np.pi/3, np.pi/3, np.pi])

        # Classify each angle into states
        states = np.zeros(len(chi_angles), dtype=int)
        multiplier = 1

        for i in range(chi_angles.shape[1]):
            angle_states = np.digitize(chi_angles[:, i], boundaries) - 1
            states += angle_states * multiplier
            multiplier *= 3

        return states

    def calculate_sidechain_flexibility(self,
                                      trajectory: md.Trajectory) -> Dict[int, float]:
        """Calculate overall side-chain flexibility scores for all residues.

        Args:
            trajectory: MDTraj trajectory

        Returns:
            Dictionary mapping residue indices to flexibility scores
        """
        flexibility_scores = {}

        for residue in self.topology.residues:
            if residue.name in self.CHI_ANGLES:
                try:
                    metrics = self.analyze_rotamer_distribution(trajectory,
                                                             residue.index)
                    if metrics:
                        # Combine entropy values into overall flexibility score
                        entropy_values = [v for k, v in metrics.items()
                                        if k.endswith('_entropy')]
                        flexibility_scores[residue.index] = float(np.mean(entropy_values))
                except Exception as e:
                    print(f"Warning: Could not analyze residue {residue.index}: {e}")

        return flexibility_scores
