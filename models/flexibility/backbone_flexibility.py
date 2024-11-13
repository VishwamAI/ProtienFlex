"""
Backbone Flexibility Analysis Module

This module provides functionality for analyzing protein backbone flexibility
using various metrics including RMSF and B-factors prediction.
"""

import numpy as np
from typing import List, Optional, Tuple
import mdtraj as md
from scipy.stats import gaussian_kde

class BackboneFlexibility:
    """Analyzes protein backbone flexibility using molecular dynamics trajectories."""

    def __init__(self, structure_file: str):
        """Initialize with a protein structure file.

        Args:
            structure_file: Path to protein structure file (PDB format)
        """
        self.structure = md.load(structure_file)
        self._validate_structure()

    def _validate_structure(self):
        """Validate the loaded structure."""
        if self.structure is None:
            raise ValueError("Failed to load structure file")
        if self.structure.n_atoms == 0:
            raise ValueError("Structure contains no atoms")

    def calculate_rmsf(self,
                      trajectory: md.Trajectory,
                      selection: str = 'backbone',
                      align: bool = True) -> np.ndarray:
        """Calculate Root Mean Square Fluctuation for selected atoms.

        Args:
            trajectory: MDTraj trajectory object
            selection: Atom selection string (default: 'backbone')
            align: Whether to align trajectory before calculation

        Returns:
            numpy array of RMSF values per selected atom
        """
        # Select atoms for analysis
        atom_indices = trajectory.topology.select(selection)
        if len(atom_indices) == 0:
            raise ValueError(f"No atoms selected with '{selection}'")

        traj_subset = trajectory.atom_slice(atom_indices)

        # Align trajectory if requested
        if align:
            traj_subset.superpose(traj_subset, 0)

        # Calculate RMSF
        xyz = traj_subset.xyz
        average_xyz = xyz.mean(axis=0)
        rmsf = np.sqrt(np.mean(np.sum((xyz - average_xyz)**2, axis=2), axis=0))

        return rmsf

    def predict_bfactors(self,
                        trajectory: md.Trajectory,
                        selection: str = 'all',
                        smoothing: bool = True) -> np.ndarray:
        """Predict B-factors from molecular dynamics trajectory.

        Args:
            trajectory: MDTraj trajectory
            selection: Atom selection string
            smoothing: Apply Gaussian smoothing to predictions

        Returns:
            numpy array of predicted B-factors per selected atom
        """
        # Select atoms and calculate fluctuations
        atom_indices = trajectory.topology.select(selection)
        traj_subset = trajectory.atom_slice(atom_indices)
        traj_subset.superpose(traj_subset, 0)

        xyz = traj_subset.xyz
        mean_xyz = xyz.mean(axis=0)

        # Calculate B-factors (B = 8π²/3 * <r²>)
        fluctuations = np.mean((xyz - mean_xyz)**2, axis=0)
        bfactors = (8 * np.pi**2 / 3) * np.sum(fluctuations, axis=1)

        # Apply smoothing if requested
        if smoothing:
            bfactors = self._smooth_bfactors(bfactors)

        return bfactors

    def _smooth_bfactors(self, bfactors: np.ndarray,
                        window_size: int = 3) -> np.ndarray:
        """Apply Gaussian smoothing to B-factors.

        Args:
            bfactors: Raw B-factor values
            window_size: Size of smoothing window

        Returns:
            Smoothed B-factor values
        """
        kernel = gaussian_kde(np.arange(-window_size, window_size + 1))
        weights = kernel(np.arange(-window_size, window_size + 1))
        weights /= weights.sum()

        smoothed = np.convolve(bfactors, weights, mode='same')
        return smoothed

    def analyze_secondary_structure_flexibility(self,
                                             trajectory: md.Trajectory) -> dict:
        """Analyze flexibility patterns in different secondary structure elements.

        Args:
            trajectory: MDTraj trajectory

        Returns:
            Dictionary containing flexibility metrics per secondary structure type
        """
        # Calculate DSSP for secondary structure assignment
        dssp = md.compute_dssp(trajectory)

        # Get backbone RMSF
        rmsf = self.calculate_rmsf(trajectory)

        # Analyze flexibility per secondary structure type
        ss_types = {'H': 'helix', 'E': 'sheet', 'C': 'coil'}
        results = {}

        for ss_type, ss_name in ss_types.items():
            # Find residues with this secondary structure
            ss_mask = (dssp == ss_type).any(axis=0)
            if not ss_mask.any():
                continue

            # Calculate average RMSF for this secondary structure
            ss_rmsf = rmsf[ss_mask]
            results[ss_name] = {
                'mean_rmsf': float(ss_rmsf.mean()),
                'std_rmsf': float(ss_rmsf.std()),
                'count': int(ss_mask.sum())
            }

        return results
