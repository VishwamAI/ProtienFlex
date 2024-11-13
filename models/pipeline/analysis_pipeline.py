"""
Analysis Pipeline Module

This module provides parallel processing capabilities for analyzing multiple
proteins and aggregating results across different analyses.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

from .flexibility_pipeline import FlexibilityPipeline

class AnalysisPipeline:
    """Pipeline for parallel protein analysis."""

    def __init__(self,
                 alphafold_model_dir: str,
                 output_dir: str,
                 n_workers: int = 4,
                 batch_size: int = 10):
        """Initialize analysis pipeline.

        Args:
            alphafold_model_dir: Directory containing AlphaFold3 model
            output_dir: Directory for output files
            n_workers: Number of parallel workers
            batch_size: Size of protein batches for parallel processing
        """
        self.alphafold_model_dir = alphafold_model_dir
        self.output_dir = output_dir
        self.n_workers = n_workers
        self.batch_size = batch_size

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.output_dir, 'analysis_pipeline.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AnalysisPipeline')

    def analyze_proteins(self,
                        proteins: List[Dict[str, str]],
                        experimental_data: Optional[Dict[str, Dict]] = None) -> Dict:
        """Analyze multiple proteins in parallel.

        Args:
            proteins: List of protein dictionaries with 'sequence' and 'name' keys
            experimental_data: Optional dictionary of experimental data by protein name

        Returns:
            Dictionary with analysis results for all proteins
        """
        try:
            self.logger.info(f"Starting analysis of {len(proteins)} proteins")

            # Create batches
            batches = [
                proteins[i:i + self.batch_size]
                for i in range(0, len(proteins), self.batch_size)
            ]

            # Process batches in parallel
            results = {}
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for batch in batches:
                    future = executor.submit(
                        self._process_batch,
                        batch,
                        experimental_data
                    )
                    futures.append(future)

                # Collect results
                for future in futures:
                    batch_results = future.result()
                    results.update(batch_results)

            # Aggregate results across all proteins
            aggregated = self._aggregate_results(results)

            # Save results
            self._save_results(results, aggregated)

            self.logger.info("Analysis completed successfully")
            return {
                'individual': results,
                'aggregated': aggregated
            }

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _process_batch(self,
                      batch: List[Dict[str, str]],
                      experimental_data: Optional[Dict[str, Dict]] = None) -> Dict:
        """Process a batch of proteins.

        Args:
            batch: List of protein dictionaries
            experimental_data: Optional experimental data

        Returns:
            Dictionary with results for batch
        """
        results = {}
        pipeline = FlexibilityPipeline(
            self.alphafold_model_dir,
            os.path.join(self.output_dir, 'individual')
        )

        for protein in batch:
            name = protein['name']
            sequence = protein['sequence']
            exp_data = experimental_data.get(name) if experimental_data else None

            try:
                result = pipeline.analyze_sequence(
                    sequence,
                    name=name,
                    experimental_data=exp_data
                )
                results[name] = result
            except Exception as e:
                self.logger.error(f"Analysis failed for {name}: {str(e)}")
                results[name] = {'error': str(e)}

        return results

    def _aggregate_results(self, results: Dict) -> Dict:
        """Aggregate results across all proteins.

        Args:
            results: Dictionary of results by protein

        Returns:
            Dictionary with aggregated statistics
        """
        aggregated = {
            'flexibility_stats': self._aggregate_flexibility_stats(results),
            'validation_stats': self._aggregate_validation_stats(results),
            'performance_stats': self._calculate_performance_stats(results)
        }

        if any('experimental' in r.get('validation', {}) for r in results.values()):
            aggregated['experimental_comparison'] = self._aggregate_experimental_comparison(results)

        return aggregated

    def _aggregate_flexibility_stats(self, results: Dict) -> Dict:
        """Aggregate flexibility statistics across proteins.

        Args:
            results: Dictionary of results by protein

        Returns:
            Dictionary with aggregated flexibility statistics
        """
        stats = {
            'rmsf': [],
            'ss_flexibility': {
                'H': [], 'E': [], 'C': []
            },
            'domain_movements': []
        }

        for protein_results in results.values():
            if 'flexibility' not in protein_results:
                continue

            flex = protein_results['flexibility']

            # Aggregate RMSF
            if 'rmsf' in flex:
                stats['rmsf'].append(flex['rmsf']['mean'])

            # Aggregate secondary structure flexibility
            if 'ss_flexibility' in flex:
                for ss_type in ['H', 'E', 'C']:
                    if ss_type in flex['ss_flexibility']:
                        stats['ss_flexibility'][ss_type].append(
                            flex['ss_flexibility'][ss_type]['mean']
                        )

            # Aggregate domain movements
            if 'domain_movements' in flex:
                stats['domain_movements'].append(
                    flex['domain_movements']['mean']
                )

        # Calculate summary statistics
        summary = {}
        for metric, values in stats.items():
            if metric == 'ss_flexibility':
                summary[metric] = {
                    ss_type: {
                        'mean': np.mean(vals) if vals else None,
                        'std': np.std(vals) if vals else None
                    }
                    for ss_type, vals in values.items()
                }
            else:
                if values:
                    summary[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }

        return summary

    def _aggregate_validation_stats(self, results: Dict) -> Dict:
        """Aggregate validation statistics across proteins.

        Args:
            results: Dictionary of results by protein

        Returns:
            Dictionary with aggregated validation statistics
        """
        stats = {
            'stability': {},
            'sampling': {}
        }

        for protein_results in results.values():
            if 'validation' not in protein_results:
                continue

            val = protein_results['validation'].get('aggregate', {})

            # Aggregate stability metrics
            for metric, values in val.get('stability', {}).items():
                if metric not in stats['stability']:
                    stats['stability'][metric] = []
                stats['stability'][metric].append(values['mean'])

            # Aggregate sampling metrics
            for metric, values in val.get('sampling', {}).items():
                if metric not in stats['sampling']:
                    stats['sampling'][metric] = []
                stats['sampling'][metric].append(values['mean'])

        # Calculate summary statistics
        summary = {}
        for category, metrics in stats.items():
            summary[category] = {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                for metric, values in metrics.items()
                if values
            }

        return summary

    def _calculate_performance_stats(self, results: Dict) -> Dict:
        """Calculate performance statistics.

        Args:
            results: Dictionary of results by protein

        Returns:
            Dictionary with performance statistics
        """
        stats = {
            'success_rate': len([r for r in results.values() if 'error' not in r]) / len(results),
            'error_types': {},
            'processing_times': []
        }

        # Collect error types
        for result in results.values():
            if 'error' in result:
                error_type = type(result['error']).__name__
                stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1

        return stats

    def _aggregate_experimental_comparison(self, results: Dict) -> Dict:
        """Aggregate experimental comparison statistics.

        Args:
            results: Dictionary of results by protein

        Returns:
            Dictionary with aggregated experimental comparison
        """
        comparisons = {
            'correlation': [],
            'rmse': [],
            'relative_error': []
        }

        for protein_results in results.values():
            if 'validation' not in protein_results:
                continue

            exp = protein_results['validation'].get('aggregate', {}).get('experimental', {})
            for metric in comparisons:
                if metric in exp:
                    comparisons[metric].append(exp[metric]['mean'])

        # Calculate summary statistics
        summary = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in comparisons.items()
            if values
        }

        return summary

    def _save_results(self,
                     individual_results: Dict,
                     aggregated_results: Dict) -> None:
        """Save analysis results.

        Args:
            individual_results: Results for individual proteins
            aggregated_results: Aggregated statistics
        """
        # Save individual results
        for name, results in individual_results.items():
            protein_dir = os.path.join(self.output_dir, 'individual', name)
            os.makedirs(protein_dir, exist_ok=True)

            # Save JSON-serializable results
            with open(os.path.join(protein_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)

        # Save aggregated results
        with open(os.path.join(self.output_dir, 'aggregated_results.json'), 'w') as f:
            json.dump(aggregated_results, f, indent=2)

        # Generate summary report
        self._generate_summary_report(individual_results, aggregated_results)

    def _generate_summary_report(self,
                               individual_results: Dict,
                               aggregated_results: Dict) -> None:
        """Generate summary report of analysis results.

        Args:
            individual_results: Results for individual proteins
            aggregated_results: Aggregated statistics
        """
        report_file = os.path.join(self.output_dir, 'analysis_report.md')

        with open(report_file, 'w') as f:
            # Write header
            f.write('# Protein Flexibility Analysis Report\n\n')
            f.write(f'Analysis completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

            # Write summary statistics
            f.write('## Summary Statistics\n\n')
            f.write(f'Total proteins analyzed: {len(individual_results)}\n')
            f.write(f'Success rate: {aggregated_results["performance_stats"]["success_rate"]:.2%}\n\n')

            # Write flexibility statistics
            f.write('## Flexibility Analysis\n\n')
            flex_stats = aggregated_results['flexibility_stats']
            for metric, stats in flex_stats.items():
                f.write(f'### {metric.replace("_", " ").title()}\n')
                if isinstance(stats, dict) and 'mean' in stats:
                    f.write(f'Mean: {stats["mean"]:.3f}\n')
                    f.write(f'Std: {stats["std"]:.3f}\n\n')
                else:
                    for submetric, substats in stats.items():
                        f.write(f'- {submetric}: {substats["mean"]:.3f} ± {substats["std"]:.3f}\n')
                    f.write('\n')

            # Write validation statistics
            f.write('## Validation Results\n\n')
            val_stats = aggregated_results['validation_stats']
            for category, metrics in val_stats.items():
                f.write(f'### {category.replace("_", " ").title()}\n')
                for metric, stats in metrics.items():
                    f.write(f'- {metric}: {stats["mean"]:.3f} ± {stats["std"]:.3f}\n')
                f.write('\n')

            # Write experimental comparison if available
            if 'experimental_comparison' in aggregated_results:
                f.write('## Experimental Comparison\n\n')
                exp_stats = aggregated_results['experimental_comparison']
                for metric, stats in exp_stats.items():
                    f.write(f'- {metric}: {stats["mean"]:.3f} ± {stats["std"]:.3f}\n')

            # Write error summary if any
            if aggregated_results['performance_stats'].get('error_types'):
                f.write('\n## Error Summary\n\n')
                for error_type, count in aggregated_results['performance_stats']['error_types'].items():
                    f.write(f'- {error_type}: {count} occurrences\n')
