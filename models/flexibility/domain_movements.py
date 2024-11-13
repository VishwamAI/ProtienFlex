"""
Domain Movement Analysis Module

This module provides functionality for analyzing protein domain movements
and large-scale conformational changes using molecular dynamics trajectories.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import mdtraj as md
from scipy.spatial import distance_matrix
from scipy.cluster import hierarchy

class DomainMovements:
    """Analyzes protein domain movements and large-scale conformational changes."""

    def __init__(self, structure_file: str):
        """Initialize with a protein structure file.

        Args:
            structure_file: Path to protein structure file (PDB format)
        """
        self.structure = md.load(structure_file)
        self.topology = self.structure.topology

    def identify_domains(self,
                        trajectory: md.Trajectory,
                        min_domain_size: int = 20,
                        contact_cutoff: float = 0.8) -> List[List[int]]:
        """Identify protein domains based on contact map analysis.

        Args:
            trajectory: MDTraj trajectory
            min_domain_size: Minimum number of residues for a domain
            contact_cutoff: Distance cutoff for contact definition (nm)

        Returns:
            List of residue indices for each identified domain
        """
        # Calculate average contact map
        contact_map = self._calculate_contact_map(trajectory, contact_cutoff)

        # Perform hierarchical clustering
        distances = 1 - contact_map
        linkage = hierarchy.linkage(distances[np.triu_indices_from(distances, k=1)],
                                  method='ward')

        # Cut tree to get domains
        clusters = hierarchy.fcluster(linkage,
                                    t=min_domain_size,
                                    criterion='maxclust')

        # Group residues by cluster
        domains = []
        for i in range(1, clusters.max() + 1):
            domain_residues = np.where(clusters == i)[0]
            if len(domain_residues) >= min_domain_size:
                domains.append(domain_residues.tolist())

        return domains

    def _calculate_contact_map(self,
                             trajectory: md.Trajectory,
                             cutoff: float) -> np.ndarray:
        """Calculate average contact map over trajectory.

        Args:
            trajectory: MDTraj trajectory
            cutoff: Distance cutoff for contacts (nm)

        Returns:
            2D numpy array of contact frequencies
        """
        # Get CA atoms for contact calculation
        ca_indices = trajectory.topology.select('name CA')
        n_residues = len(ca_indices)
        contact_freq = np.zeros((n_residues, n_residues))

        # Calculate contacts for each frame
        for frame in trajectory:
            xyz = frame.atom_slice(ca_indices).xyz[0]
            dist_matrix = distance_matrix(xyz, xyz)
            contacts = dist_matrix < cutoff
            contact_freq += contacts

        return contact_freq / len(trajectory)

    def analyze_domain_motion(self,
                            trajectory: md.Trajectory,
                            domain1_residues: List[int],
                            domain2_residues: List[int]) -> Dict[str, float]:
        """Analyze relative motion between two protein domains.

        Args:
            trajectory: MDTraj trajectory
            domain1_residues: List of residue indices for first domain
            domain2_residues: List of residue indices for second domain

        Returns:
            Dictionary containing motion metrics
        """
        # Get atom indices for domains (CA atoms)
        top = trajectory.topology
        d1_atoms = top.select(f'name CA and resid {" ".join(map(str, domain1_residues))}')
        d2_atoms = top.select(f'name CA and resid {" ".join(map(str, domain2_residues))}')

        # Calculate domain centers and orientations over time
        d1_coords = trajectory.atom_slice(d1_atoms).xyz
        d2_coords = trajectory.atom_slice(d2_atoms).xyz

        d1_centers = np.mean(d1_coords, axis=1)  # [n_frames, 3]
        d2_centers = np.mean(d2_coords, axis=1)  # [n_frames, 3]

        # Calculate relative translation
        translations = np.linalg.norm(d2_centers - d1_centers, axis=1)

        # Calculate relative rotation using SVD
        rotations = []
        for i in range(len(trajectory)):
            R = self._calculate_rotation_matrix(d1_coords[i] - d1_centers[i],
                                             d2_coords[i] - d2_centers[i])
            angle = np.arccos((np.trace(R) - 1) / 2)
            rotations.append(angle)

        rotations = np.array(rotations)

        return {
            'mean_translation': float(np.mean(translations)),
            'std_translation': float(np.std(translations)),
            'max_translation': float(np.max(translations)),
            'mean_rotation': float(np.degrees(np.mean(rotations))),
            'std_rotation': float(np.degrees(np.std(rotations))),
            'max_rotation': float(np.degrees(np.max(rotations)))
        }

    def _calculate_rotation_matrix(self,
                                 coords1: np.ndarray,
                                 coords2: np.ndarray) -> np.ndarray:
        """Calculate rotation matrix between two sets of coordinates.

        Args:
            coords1: First set of coordinates [n_atoms, 3]
            coords2: Second set of coordinates [n_atoms, 3]

        Returns:
            3x3 rotation matrix
        """
        # Center coordinates
        coords1 = coords1 - np.mean(coords1, axis=0)
        coords2 = coords2 - np.mean(coords2, axis=0)

        # Calculate correlation matrix
        H = coords1.T @ coords2

        # SVD decomposition
        U, _, Vt = np.linalg.svd(H)

        # Calculate rotation matrix
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T

        return R

    def calculate_hinge_points(self,
                             trajectory: md.Trajectory,
                             domain1_residues: List[int],
                             domain2_residues: List[int]) -> List[int]:
        """Identify hinge points between two domains.

        Args:
            trajectory: MDTraj trajectory
            domain1_residues: List of residue indices for first domain
            domain2_residues: List of residue indices for second domain

        Returns:
            List of residue indices identified as hinge points
        """
        # Get all residues between domains
        all_residues = set(range(trajectory.topology.n_residues))
        domain_residues = set(domain1_residues + domain2_residues)
        linker_residues = sorted(all_residues - domain_residues)

        if not linker_residues:
            return []

        # Calculate RMSF for linker region
        ca_indices = []
        for res_idx in linker_residues:
            atom_indices = trajectory.topology.select(f'name CA and resid {res_idx}')
            if len(atom_indices) > 0:
                ca_indices.append(atom_indices[0])

        if not ca_indices:
            return []

        # Calculate RMSF
        traj_ca = trajectory.atom_slice(ca_indices)
        traj_ca.superpose(traj_ca, 0)
        xyz = traj_ca.xyz
        mean_xyz = xyz.mean(axis=0)
        rmsf = np.sqrt(np.mean(np.sum((xyz - mean_xyz)**2, axis=2), axis=0))

        # Identify hinge points as residues with high RMSF
        threshold = np.mean(rmsf) + np.std(rmsf)
        hinge_indices = [linker_residues[i] for i, r in enumerate(rmsf) if r > threshold]

        return hinge_indices
