"""
Flexibility Analysis Module

This module provides specialized tools for analyzing protein flexibility
from molecular dynamics trajectories, including backbone fluctuations,
side-chain mobility, and domain movements.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import mdtraj as md
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import logging
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor

class FlexibilityAnalysis:
    """Analysis tools for protein flexibility from MD trajectories."""

    def __init__(self, trajectory: md.Trajectory):
        """Initialize analysis with trajectory.

        Args:
            trajectory: MDTraj trajectory object
        """
        self.trajectory = trajectory
        self.topology = trajectory.topology
        self._cache = {}

    def calculate_rmsf(self,
                      atom_indices: Optional[List[int]] = None,
                      align: bool = True) -> np.ndarray:
        """Calculate Root Mean Square Fluctuation.

        Args:
            atom_indices: Specific atoms to analyze (default: all atoms)
            align: Whether to align trajectory first

        Returns:
            Array of RMSF values per atom
        """
        if atom_indices is None:
            atom_indices = self.topology.select('protein')

        # Align trajectory if requested
        if align:
            reference = self.trajectory[0]
            aligned = self.trajectory.superpose(reference, atom_indices=atom_indices)
        else:
            aligned = self.trajectory

        # Calculate RMSF
        xyz = aligned.xyz[:, atom_indices]
        average_structure = xyz.mean(axis=0)
        diff = xyz - average_structure
        rmsf = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=0))

        return rmsf

    def analyze_secondary_structure_flexibility(self) -> Dict[str, float]:
        """Analyze flexibility by secondary structure type.

        Returns:
            Dictionary of average RMSF per secondary structure type
        """
        # Calculate secondary structure
        ss = md.compute_dssp(self.trajectory, simplified=True)

        # Calculate RMSF
        rmsf = self.calculate_rmsf()
        ca_indices = self.topology.select('name CA')

        # Group by secondary structure
        ss_flexibility = {
            'H': [],  # Alpha helix
            'E': [],  # Beta sheet
            'C': []   # Coil
        }

        for i, idx in enumerate(ca_indices):
            ss_type = ss[0, i]  # Use first frame's assignment
            if ss_type in ss_flexibility:
                ss_flexibility[ss_type].append(rmsf[i])

        # Calculate averages
        return {
            ss_type: np.mean(values)
            for ss_type, values in ss_flexibility.items()
            if values
        }

    def calculate_residue_correlations(self,
                                     method: str = 'linear') -> np.ndarray:
        """Calculate residue motion correlations.

        Args:
            method: Correlation method ('linear' or 'mutual_information')

        Returns:
            Correlation matrix
        """
        ca_indices = self.topology.select('name CA')
        n_residues = len(ca_indices)

        if method == 'linear':
            # Calculate linear correlation
            xyz = self.trajectory.xyz[:, ca_indices]
            flat_traj = xyz.reshape(xyz.shape[0], -1)
            corr_matrix = np.corrcoef(flat_traj.T)

        elif method == 'mutual_information':
            # Calculate mutual information
            corr_matrix = np.zeros((n_residues, n_residues))
            xyz = self.trajectory.xyz[:, ca_indices]

            for i in range(n_residues):
                for j in range(i, n_residues):
                    mi = self._calculate_mutual_information(
                        xyz[:, i],
                        xyz[:, j]
                    )
                    corr_matrix[i, j] = mi
                    corr_matrix[j, i] = mi

        return corr_matrix

    def _calculate_mutual_information(self,
                                   x: np.ndarray,
                                   y: np.ndarray,
                                   bins: int = 20) -> float:
        """Calculate mutual information between two coordinate trajectories.

        Args:
            x: First coordinate trajectory
            y: Second coordinate trajectory
            bins: Number of bins for histogram

        Returns:
            Mutual information value
        """
        hist_xy, _, _ = np.histogram2d(x.flatten(), y.flatten(), bins=bins)
        hist_x, _ = np.histogram(x.flatten(), bins=bins)
        hist_y, _ = np.histogram(y.flatten(), bins=bins)

        # Normalize
        hist_xy = hist_xy / np.sum(hist_xy)
        hist_x = hist_x / np.sum(hist_x)
        hist_y = hist_y / np.sum(hist_y)

        # Calculate mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if hist_xy[i, j] > 0:
                    mi += hist_xy[i, j] * np.log(
                        hist_xy[i, j] / (hist_x[i] * hist_y[j])
                    )

        return mi

    def identify_flexible_regions(self,
                                percentile: float = 90.0) -> List[Tuple[int, int]]:
        """Identify contiguous flexible regions.

        Args:
            percentile: Percentile threshold for flexibility

        Returns:
            List of (start, end) residue indices for flexible regions
        """
        # Calculate RMSF for CA atoms
        ca_indices = self.topology.select('name CA')
        rmsf = self.calculate_rmsf(atom_indices=ca_indices)

        # Find highly flexible residues
        threshold = np.percentile(rmsf, percentile)
        flexible_mask = rmsf > threshold

        # Find contiguous regions
        regions = []
        start = None

        for i, is_flexible in enumerate(flexible_mask):
            if is_flexible and start is None:
                start = i
            elif not is_flexible and start is not None:
                regions.append((start, i-1))
                start = None

        if start is not None:
            regions.append((start, len(flexible_mask)-1))

        return regions

    def analyze_domain_movements(self,
                               contact_cutoff: float = 0.8) -> Dict[str, np.ndarray]:
        """Analyze relative domain movements.

        Args:
            contact_cutoff: Distance cutoff for contact map (nm)

        Returns:
            Dictionary with domain analysis results
        """
        # Calculate contact map
        ca_indices = self.topology.select('name CA')
        contact_map = self._calculate_contact_map(ca_indices, contact_cutoff)

        # Cluster contact map to identify domains
        clustering = DBSCAN(eps=0.3, min_samples=5)
        domains = clustering.fit_predict(contact_map)

        # Calculate domain centers and movements
        domain_centers = {}
        domain_movements = {}

        for domain_id in np.unique(domains):
            if domain_id == -1:  # Skip noise
                continue

            domain_indices = ca_indices[domains == domain_id]

            # Calculate domain center trajectory
            xyz = self.trajectory.xyz[:, domain_indices]
            centers = xyz.mean(axis=1)

            domain_centers[f'domain_{domain_id}'] = centers

            # Calculate domain movement relative to initial position
            movements = np.linalg.norm(centers - centers[0], axis=1)
            domain_movements[f'domain_{domain_id}'] = movements

        return {
            'domain_centers': domain_centers,
            'domain_movements': domain_movements,
            'domain_assignments': domains
        }

    def _calculate_contact_map(self,
                             atom_indices: np.ndarray,
                             cutoff: float) -> np.ndarray:
        """Calculate contact map for given atoms.

        Args:
            atom_indices: Atom indices to analyze
            cutoff: Distance cutoff (nm)

        Returns:
            Contact map matrix
        """
        # Calculate average distances
        distances = np.zeros((len(atom_indices), len(atom_indices)))

        for frame in self.trajectory:
            dist_matrix = squareform(pdist(frame.xyz[0, atom_indices]))
            distances += dist_matrix

        distances /= len(self.trajectory)

        # Convert to contact map
        contact_map = distances < cutoff
        return contact_map.astype(float)

    def calculate_flexibility_profile(self) -> Dict[str, np.ndarray]:
        """Calculate comprehensive flexibility profile.

        Returns:
            Dictionary with various flexibility metrics
        """
        # Calculate basic metrics
        ca_indices = self.topology.select('name CA')
        rmsf = self.calculate_rmsf(atom_indices=ca_indices)

        # Calculate secondary structure flexibility
        ss_flex = self.analyze_secondary_structure_flexibility()

        # Calculate correlations
        correlations = self.calculate_residue_correlations()

        # Identify flexible regions
        flexible_regions = self.identify_flexible_regions()

        # Analyze domain movements
        domain_analysis = self.analyze_domain_movements()

        return {
            'rmsf': rmsf,
            'ss_flexibility': ss_flex,
            'correlations': correlations,
            'flexible_regions': flexible_regions,
            'domain_analysis': domain_analysis
        }

    def analyze_conformational_substates(self,
                                       n_clusters: int = 5) -> Dict[str, np.ndarray]:
        """Analyze conformational substates using clustering.

        Args:
            n_clusters: Number of conformational substates to identify

        Returns:
            Dictionary with clustering results
        """
        from sklearn.cluster import KMeans

        # Get CA coordinates
        ca_indices = self.topology.select('name CA')
        xyz = self.trajectory.xyz[:, ca_indices]
        n_frames = xyz.shape[0]

        # Reshape for clustering
        reshaped_xyz = xyz.reshape(n_frames, -1)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(reshaped_xyz)

        # Calculate cluster centers and convert back to 3D
        centers = kmeans.cluster_centers_.reshape(-1, len(ca_indices), 3)

        # Calculate transition matrix
        transitions = np.zeros((n_clusters, n_clusters))
        for i in range(len(labels)-1):
            transitions[labels[i], labels[i+1]] += 1

        # Normalize transitions
        row_sums = transitions.sum(axis=1)
        transitions = transitions / row_sums[:, np.newaxis]

        return {
            'labels': labels,
            'centers': centers,
            'transitions': transitions,
            'populations': np.bincount(labels) / len(labels)
        }

    def calculate_entropy_profile(self,
                                window_size: int = 10) -> np.ndarray:
        """Calculate position-wise conformational entropy.

        Args:
            window_size: Window size for local entropy calculation

        Returns:
            Array of entropy values per residue
        """
        ca_indices = self.topology.select('name CA')
        xyz = self.trajectory.xyz[:, ca_indices]
        n_residues = len(ca_indices)

        entropy_profile = np.zeros(n_residues)

        for i in range(n_residues):
            # Get local window
            start = max(0, i - window_size//2)
            end = min(n_residues, i + window_size//2)

            # Calculate local conformational entropy
            local_xyz = xyz[:, start:end].reshape(len(xyz), -1)

            # Use kernel density estimation for entropy
            from sklearn.neighbors import KernelDensity
            kde = KernelDensity(bandwidth=0.2)
            kde.fit(local_xyz)

            # Sample points and calculate entropy
            sample_points = kde.sample(1000)
            log_dens = kde.score_samples(sample_points)
            entropy_profile[i] = -np.mean(log_dens)

        return entropy_profile
