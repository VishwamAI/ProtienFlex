"""
Molecular Dynamics Validation Module

This module provides tools for validating molecular dynamics simulations
and comparing results with experimental data.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import mdtraj as md
from scipy import stats
import logging
from Bio.PDB import PDBList, PDBParser
import requests

class SimulationValidator:
    """Validation tools for molecular dynamics simulations."""

    def __init__(self, trajectory: md.Trajectory):
        """Initialize validator with trajectory.

        Args:
            trajectory: MDTraj trajectory object
        """
        self.trajectory = trajectory
        self.topology = trajectory.topology
        self._cache = {}

    def validate_simulation_stability(self) -> Dict[str, float]:
        """Validate simulation stability metrics.

        Returns:
            Dictionary of stability metrics
        """
        metrics = {}

        # Calculate RMSD relative to first frame
        rmsd = md.rmsd(self.trajectory, self.trajectory, 0)
        metrics['rmsd_mean'] = np.mean(rmsd)
        metrics['rmsd_std'] = np.std(rmsd)
        metrics['rmsd_drift'] = rmsd[-1] - rmsd[0]

        # Calculate radius of gyration
        rg = md.compute_rg(self.trajectory)
        metrics['rg_mean'] = np.mean(rg)
        metrics['rg_std'] = np.std(rg)
        metrics['rg_drift'] = rg[-1] - rg[0]

        # Calculate total energy if available
        if hasattr(self.trajectory, 'energies'):
            energy = self.trajectory.energies
            metrics['energy_mean'] = np.mean(energy)
            metrics['energy_std'] = np.std(energy)
            metrics['energy_drift'] = energy[-1] - energy[0]

        return metrics

    def validate_sampling_quality(self,
                                n_clusters: int = 5) -> Dict[str, float]:
        """Validate sampling quality metrics.

        Args:
            n_clusters: Number of clusters for conformational analysis

        Returns:
            Dictionary of sampling quality metrics
        """
        from sklearn.cluster import KMeans

        # Get CA coordinates
        ca_indices = self.topology.select('name CA')
        xyz = self.trajectory.xyz[:, ca_indices]
        n_frames = xyz.shape[0]

        # Perform clustering
        reshaped_xyz = xyz.reshape(n_frames, -1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(reshaped_xyz)

        # Calculate metrics
        metrics = {}

        # Population entropy
        populations = np.bincount(labels) / len(labels)
        metrics['population_entropy'] = stats.entropy(populations)

        # Transition density
        transitions = np.zeros((n_clusters, n_clusters))
        for i in range(len(labels)-1):
            transitions[labels[i], labels[i+1]] += 1
        metrics['transition_density'] = np.count_nonzero(transitions) / transitions.size

        # RMSD coverage
        rmsd_matrix = np.zeros((n_frames, n_frames))
        for i in range(n_frames):
            rmsd_matrix[i] = md.rmsd(self.trajectory, self.trajectory, i, atom_indices=ca_indices)
        metrics['rmsd_coverage'] = np.mean(rmsd_matrix)

        return metrics

    def compare_with_experimental_bfactors(self,
                                         pdb_id: str) -> Dict[str, float]:
        """Compare simulation fluctuations with experimental B-factors.

        Args:
            pdb_id: PDB ID of experimental structure

        Returns:
            Dictionary of comparison metrics
        """
        # Download experimental structure
        pdbl = PDBList()
        parser = PDBParser()
        pdb_file = pdbl.retrieve_pdb_file(pdb_id, file_format='pdb')
        structure = parser.get_structure(pdb_id, pdb_file)

        # Extract experimental B-factors
        exp_bfactors = []
        for atom in structure.get_atoms():
            if atom.name == 'CA':
                exp_bfactors.append(atom.bfactor)
        exp_bfactors = np.array(exp_bfactors)

        # Calculate simulation B-factors
        ca_indices = self.topology.select('name CA')
        rmsf = md.rmsf(self.trajectory, self.trajectory, atom_indices=ca_indices)
        sim_bfactors = (8 * np.pi**2 / 3) * rmsf**2

        # Calculate comparison metrics
        metrics = {}
        metrics['correlation'] = stats.pearsonr(exp_bfactors, sim_bfactors)[0]
        metrics['rmse'] = np.sqrt(np.mean((exp_bfactors - sim_bfactors)**2))
        metrics['relative_error'] = np.mean(np.abs(exp_bfactors - sim_bfactors) / exp_bfactors)

        return metrics

    def validate_replica_exchange(self,
                                temperatures: List[float],
                                exchanges: List[int]) -> Dict[str, float]:
        """Validate replica exchange simulation.

        Args:
            temperatures: List of replica temperatures
            exchanges: List of accepted exchanges

        Returns:
            Dictionary of replica exchange metrics
        """
        metrics = {}

        # Calculate exchange acceptance rate
        metrics['exchange_rate'] = len(exchanges) / (len(temperatures) - 1)

        # Calculate temperature diffusion
        temp_transitions = np.zeros((len(temperatures), len(temperatures)))
        for ex in exchanges:
            temp_transitions[ex, ex+1] += 1
            temp_transitions[ex+1, ex] += 1

        # Normalize transitions
        temp_transitions /= np.sum(temp_transitions, axis=1)[:, np.newaxis]

        # Calculate diffusion metrics
        metrics['temp_diffusion'] = np.mean(temp_transitions)
        metrics['temp_mixing'] = np.std(temp_transitions)

        return metrics

    def validate_metadynamics(self,
                            cv_values: List[np.ndarray],
                            bias_potential: List[float]) -> Dict[str, float]:
        """Validate metadynamics simulation.

        Args:
            cv_values: List of collective variable values
            bias_potential: List of bias potential values

        Returns:
            Dictionary of metadynamics metrics
        """
        metrics = {}

        # Convert to arrays
        cv_values = np.array(cv_values)
        bias_potential = np.array(bias_potential)

        # Calculate CV coverage
        for i in range(cv_values.shape[1]):
            cv = cv_values[:, i]
            metrics[f'cv{i}_coverage'] = (np.max(cv) - np.min(cv)) / np.std(cv)

        # Calculate bias growth rate
        metrics['bias_growth_rate'] = np.polyfit(
            np.arange(len(bias_potential)),
            bias_potential,
            1
        )[0]

        # Calculate CV distribution entropy
        for i in range(cv_values.shape[1]):
            hist, _ = np.histogram(cv_values[:, i], bins=20, density=True)
            metrics[f'cv{i}_entropy'] = stats.entropy(hist)

        return metrics

    def validate_against_experimental_data(self,
                                         exp_data: Dict[str, float]) -> Dict[str, float]:
        """Validate simulation against experimental measurements.

        Args:
            exp_data: Dictionary of experimental measurements

        Returns:
            Dictionary of validation metrics
        """
        metrics = {}

        # Calculate simulation observables
        sim_observables = self._calculate_observables()

        # Compare with experimental data
        for observable, exp_value in exp_data.items():
            if observable in sim_observables:
                sim_value = sim_observables[observable]
                metrics[f'{observable}_error'] = abs(exp_value - sim_value) / exp_value
                metrics[f'{observable}_zscore'] = (sim_value - exp_value) / exp_value

        return metrics

    def _calculate_observables(self) -> Dict[str, float]:
        """Calculate common experimental observables from simulation.

        Returns:
            Dictionary of calculated observables
        """
        observables = {}

        # Calculate radius of gyration
        observables['rg'] = np.mean(md.compute_rg(self.trajectory))

        # Calculate end-to-end distance
        n_term = self.topology.select('name N and resid 0')
        c_term = self.topology.select('name C and resid -1')
        if len(n_term) > 0 and len(c_term) > 0:
            end_to_end = md.compute_distances(
                self.trajectory,
                [[n_term[0], c_term[0]]]
            )
            observables['end_to_end'] = np.mean(end_to_end)

        # Calculate solvent accessible surface area
        observables['sasa'] = np.mean(md.shrake_rupley(self.trajectory))

        return observables

    def generate_validation_report(self) -> Dict[str, Dict[str, float]]:
        """Generate comprehensive validation report.

        Returns:
            Dictionary with all validation metrics
        """
        report = {}

        # Stability validation
        report['stability'] = self.validate_simulation_stability()

        # Sampling validation
        report['sampling'] = self.validate_sampling_quality()

        # Calculate basic observables
        report['observables'] = self._calculate_observables()

        # Add timestamp and trajectory info
        report['metadata'] = {
            'n_frames': self.trajectory.n_frames,
            'n_atoms': self.trajectory.n_atoms,
            'time_step': self.trajectory.timestep if hasattr(self.trajectory, 'timestep') else None,
            'total_time': self.trajectory.time[-1] if hasattr(self.trajectory, 'time') else None
        }

        return report
