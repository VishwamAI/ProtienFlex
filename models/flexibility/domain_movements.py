"""Domain movement analysis module"""
import numpy as np
import mdtraj as md
from typing import Dict, List, Tuple, Optional
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

class DomainMovements:
    """Analyze protein domain movements"""

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

    def identify_domains(self, n_domains: int = 2) -> Dict[int, List[int]]:
        """Identify protein domains using hierarchical clustering"""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")

        # Calculate distance matrix
        distances = np.zeros((self.trajectory.n_residues, self.trajectory.n_residues))
        ca_indices = [atom.index for atom in self.topology.atoms if atom.name == 'CA']

        for i in range(len(ca_indices)):
            for j in range(i+1, len(ca_indices)):
                dist = md.compute_distances(self.trajectory, [[ca_indices[i], ca_indices[j]]])
                distances[i,j] = distances[j,i] = np.mean(dist)

        # Perform hierarchical clustering
        Z = linkage(pdist(distances), method='ward')
        labels = fcluster(Z, n_domains, criterion='maxclust')

        # Group residues by domain
        domains = {}
        for i in range(n_domains):
            domains[i] = [j for j, label in enumerate(labels) if label == i+1]

        return domains

    def calculate_domain_movements(self, domains: Dict[int, List[int]]) -> Dict[str, np.ndarray]:
        """Calculate relative movements between domains"""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")

        movements = {}

        # Calculate center of mass for each domain
        domain_cms = {}
        for domain_id, residues in domains.items():
            atoms = []
            for res in residues:
                atoms.extend([atom.index for atom in self.topology.residue(res).atoms])
            domain_cms[domain_id] = md.compute_center_of_mass(self.trajectory.atom_slice(atoms))

        # Calculate relative movements
        for i in range(len(domains)):
            for j in range(i+1, len(domains)):
                key = f"domain_{i}_to_{j}"
                movements[key] = np.linalg.norm(domain_cms[i] - domain_cms[j], axis=1)

        return movements

    def analyze_hinge_regions(self, domains: Dict[int, List[int]]) -> List[int]:
        """Identify potential hinge regions between domains"""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")

        hinge_residues = []

        # Calculate per-residue RMSF
        ref = self.trajectory[0]
        aligned = self.trajectory.superpose(ref)
        xyz = aligned.xyz
        mean_xyz = xyz.mean(axis=0)
        rmsf = np.sqrt(np.mean((xyz - mean_xyz)**2, axis=0)).mean(axis=1)

        # Identify residues at domain boundaries with high mobility
        for i in range(len(domains)-1):
            domain1_end = max(domains[i])
            domain2_start = min(domains[i+1])

            # Check residues at the boundary
            boundary_range = range(domain1_end-2, domain2_start+3)
            for res in boundary_range:
                if 0 <= res < len(rmsf) and rmsf[res] > np.mean(rmsf) + np.std(rmsf):
                    hinge_residues.append(res)

        return sorted(hinge_residues)
