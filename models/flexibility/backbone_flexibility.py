"""Backbone flexibility analysis module"""
import numpy as np
import mdtraj as md
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BackboneFlexibility:
    """Analyze protein backbone flexibility"""

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

    def calculate_rmsd(self, reference_frame: Optional[int] = 0) -> np.ndarray:
        """Calculate RMSD relative to reference frame"""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")

        ref = self.trajectory[reference_frame]
        rmsd = np.zeros(len(self.trajectory))

        for i in range(len(self.trajectory)):
            frame = self.trajectory[i]
            rmsd[i] = md.rmsd(frame, ref, 0)[0]

        return rmsd

    def calculate_rmsf(self) -> np.ndarray:
        """Calculate RMSF for backbone atoms"""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")

        # Align trajectory to first frame
        ref = self.trajectory[0]
        aligned = self.trajectory.superpose(ref)

        # Calculate RMSF
        xyz = aligned.xyz
        mean_xyz = xyz.mean(axis=0)
        rmsf = np.sqrt(np.mean((xyz - mean_xyz)**2, axis=0))

        return rmsf.mean(axis=1)

    def analyze_secondary_structure(self) -> Dict[str, List[float]]:
        """Analyze secondary structure stability"""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")

        dssp = md.compute_dssp(self.trajectory)

        # Calculate secondary structure propensity
        n_frames = len(dssp)
        ss_types = {'H': [], 'E': [], 'C': []}

        for i in range(dssp.shape[1]):
            residue_ss = dssp[:, i]
            ss_types['H'].append(np.sum(residue_ss == 'H') / n_frames)
            ss_types['E'].append(np.sum(residue_ss == 'E') / n_frames)
            ss_types['C'].append(np.sum(residue_ss == 'C') / n_frames)

        return ss_types
