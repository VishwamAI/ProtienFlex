"""
Flexibility Analysis Pipeline

This module provides a unified pipeline that combines AlphaFold3 structure prediction,
enhanced molecular dynamics, and comprehensive flexibility analysis.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import mdtraj as md
from datetime import datetime

from ..prediction import AlphaFold3Interface, StructureConverter
from ..dynamics import EnhancedSampling, FlexibilityAnalysis, SimulationValidator
from Bio import SeqIO
from Bio.Seq import Seq

class FlexibilityPipeline:
    """Unified pipeline for protein flexibility analysis."""

    def __init__(self,
                 alphafold_model_dir: str,
                 output_dir: str,
                 n_workers: int = 4):
        """Initialize pipeline.

        Args:
            alphafold_model_dir: Directory containing AlphaFold3 model
            output_dir: Directory for output files
            n_workers: Number of parallel workers
        """
        self.alphafold_model_dir = alphafold_model_dir
        self.output_dir = output_dir
        self.n_workers = n_workers

        # Initialize components
        self.predictor = AlphaFold3Interface(alphafold_model_dir)
        self.converter = StructureConverter()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.output_dir, 'pipeline.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FlexibilityPipeline')

    def analyze_sequence(self,
                        sequence: str,
                        name: str = None,
                        experimental_data: Dict = None) -> Dict:
        """Run complete flexibility analysis pipeline.

        Args:
            sequence: Protein sequence
            name: Optional name for the analysis
            experimental_data: Optional experimental data for validation

        Returns:
            Dictionary with analysis results
        """
        try:
            # Generate unique name if not provided
            if name is None:
                name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.logger.info(f"Starting analysis for {name}")

            # Create analysis directory
            analysis_dir = os.path.join(self.output_dir, name)
            os.makedirs(analysis_dir, exist_ok=True)

            # Step 1: Structure Prediction
            self.logger.info("Running structure prediction")
            positions, confidence = self.predictor.predict_structure(sequence)
            structure = self.converter.alphafold_to_openmm(
                positions, sequence, confidence['plddt']
            )

            # Step 2: Enhanced Sampling
            self.logger.info("Running enhanced sampling")
            sampling_results = self._run_enhanced_sampling(
                structure,
                analysis_dir
            )

            # Step 3: Flexibility Analysis
            self.logger.info("Analyzing flexibility")
            flexibility_results = self._analyze_flexibility(
                sampling_results['trajectories'],
                parallel=True
            )

            # Step 4: Validation
            self.logger.info("Validating results")
            validation_results = self._validate_results(
                sampling_results['trajectories'],
                experimental_data
            )

            # Combine results
            results = {
                'structure_prediction': {
                    'confidence': confidence,
                    'structure': structure
                },
                'sampling': sampling_results,
                'flexibility': flexibility_results,
                'validation': validation_results
            }

            # Save results
            self._save_results(results, analysis_dir)

            self.logger.info(f"Analysis completed for {name}")
            return results

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _run_enhanced_sampling(self,
                             structure: object,
                             output_dir: str) -> Dict:
        """Run enhanced sampling simulations.

        Args:
            structure: OpenMM structure
            output_dir: Output directory

        Returns:
            Dictionary with sampling results
        """
        # Initialize enhanced sampling
        simulator = EnhancedSampling(structure)

        # Setup replica exchange
        replicas = simulator.setup_replica_exchange(
            n_replicas=4,
            temp_min=300.0,
            temp_max=400.0
        )

        # Run parallel simulations
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for i, replica in enumerate(replicas):
                replica_dir = os.path.join(output_dir, f'replica_{i}')
                future = executor.submit(
                    simulator.run_replica_exchange,
                    n_steps=1000000,
                    output_dir=replica_dir
                )
                futures.append(future)

            # Collect results
            results = [f.result() for f in futures]

        # Load trajectories
        trajectories = []
        for i in range(len(replicas)):
            traj_file = os.path.join(output_dir, f'replica_{i}/traj.h5')
            traj = md.load(traj_file)
            trajectories.append(traj)

        return {
            'trajectories': trajectories,
            'exchange_stats': results
        }

    def _analyze_flexibility(self,
                           trajectories: List[md.Trajectory],
                           parallel: bool = True) -> Dict:
        """Analyze flexibility from trajectories.

        Args:
            trajectories: List of MD trajectories
            parallel: Whether to use parallel processing

        Returns:
            Dictionary with flexibility analysis results
        """
        results = {}

        if parallel:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                # Analyze each trajectory in parallel
                futures = []
                for traj in trajectories:
                    analyzer = FlexibilityAnalysis(traj)
                    future = executor.submit(analyzer.calculate_flexibility_profile)
                    futures.append(future)

                # Collect results
                traj_results = [f.result() for f in futures]

            # Aggregate results
            results = self._aggregate_flexibility_results(traj_results)
        else:
            # Sequential analysis
            traj_results = []
            for traj in trajectories:
                analyzer = FlexibilityAnalysis(traj)
                result = analyzer.calculate_flexibility_profile()
                traj_results.append(result)

            results = self._aggregate_flexibility_results(traj_results)

        return results

    def _aggregate_flexibility_results(self,
                                    traj_results: List[Dict]) -> Dict:
        """Aggregate flexibility results from multiple trajectories.

        Args:
            traj_results: List of trajectory analysis results

        Returns:
            Aggregated results dictionary
        """
        aggregated = {}

        # Aggregate RMSF
        rmsf_values = [r['rmsf'] for r in traj_results]
        aggregated['rmsf'] = {
            'mean': np.mean(rmsf_values, axis=0),
            'std': np.std(rmsf_values, axis=0)
        }

        # Aggregate secondary structure flexibility
        ss_flex = {}
        for ss_type in ['H', 'E', 'C']:
            values = [r['ss_flexibility'][ss_type] for r in traj_results]
            ss_flex[ss_type] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        aggregated['ss_flexibility'] = ss_flex

        # Aggregate correlations
        corr_matrices = [r['correlations'] for r in traj_results]
        aggregated['correlations'] = {
            'mean': np.mean(corr_matrices, axis=0),
            'std': np.std(corr_matrices, axis=0)
        }

        # Aggregate domain analysis
        domain_results = []
        for r in traj_results:
            domain_results.extend(r['domain_analysis']['domain_movements'].values())
        aggregated['domain_movements'] = {
            'mean': np.mean(domain_results, axis=0),
            'std': np.std(domain_results, axis=0)
        }

        return aggregated

    def _validate_results(self,
                         trajectories: List[md.Trajectory],
                         experimental_data: Optional[Dict] = None) -> Dict:
        """Validate simulation results.

        Args:
            trajectories: List of MD trajectories
            experimental_data: Optional experimental data

        Returns:
            Dictionary with validation results
        """
        validation_results = {}

        # Validate each trajectory
        for i, traj in enumerate(trajectories):
            validator = SimulationValidator(traj)

            # Basic validation
            stability = validator.validate_simulation_stability()
            sampling = validator.validate_sampling_quality()

            validation_results[f'replica_{i}'] = {
                'stability': stability,
                'sampling': sampling
            }

            # Compare with experimental data if available
            if experimental_data:
                exp_comparison = validator.validate_against_experimental_data(
                    experimental_data
                )
                validation_results[f'replica_{i}']['experimental'] = exp_comparison

        # Aggregate validation results
        validation_results['aggregate'] = self._aggregate_validation_results(
            [v for k, v in validation_results.items() if k != 'aggregate']
        )

        return validation_results

    def _aggregate_validation_results(self, replica_results: List[Dict]) -> Dict:
        """Aggregate validation results from multiple replicas.

        Args:
            replica_results: List of replica validation results

        Returns:
            Aggregated validation metrics
        """
        aggregated = {'stability': {}, 'sampling': {}}

        # Aggregate stability metrics
        for metric in replica_results[0]['stability']:
            values = [r['stability'][metric] for r in replica_results]
            aggregated['stability'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

        # Aggregate sampling metrics
        for metric in replica_results[0]['sampling']:
            values = [r['sampling'][metric] for r in replica_results]
            aggregated['sampling'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

        # Aggregate experimental comparison if available
        if 'experimental' in replica_results[0]:
            aggregated['experimental'] = {}
            for metric in replica_results[0]['experimental']:
                values = [r['experimental'][metric] for r in replica_results]
                aggregated['experimental'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }

        return aggregated

    def _save_results(self, results: Dict, output_dir: str) -> None:
        """Save analysis results to files.

        Args:
            results: Analysis results dictionary
            output_dir: Output directory
        """
        import json
        import pickle

        # Save JSON-serializable results
        json_results = {
            'confidence': results['structure_prediction']['confidence'],
            'sampling': {
                k: v for k, v in results['sampling'].items()
                if k != 'trajectories'
            },
            'flexibility': results['flexibility'],
            'validation': results['validation']
        }

        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)

        # Save full results including trajectories
        with open(os.path.join(output_dir, 'full_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        self.logger.info(f"Results saved to {output_dir}")

    def load_results(self, analysis_dir: str) -> Dict:
        """Load saved analysis results.

        Args:
            analysis_dir: Analysis directory

        Returns:
            Dictionary with analysis results
        """
        import pickle

        results_file = os.path.join(analysis_dir, 'full_results.pkl')
        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        return results
