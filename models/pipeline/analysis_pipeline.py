"""Analysis pipeline implementation"""
import logging
from typing import Dict, Any, Optional
from ..flexibility import BackboneFlexibility, SidechainMobility, DomainMovements
from ..prediction import AlphaFoldInterface, StructureConverter
from ..optimization import GPUManager, DataHandler, ProgressTracker

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """Pipeline for comprehensive protein structure analysis"""

    def __init__(self, gpu_required: bool = True):
        self.gpu_manager = GPUManager() if gpu_required else None
        self.data_handler = DataHandler()
        self.progress_tracker = None

        # Analysis components
        self.structure_predictor = AlphaFoldInterface()
        self.structure_converter = StructureConverter()
        self.flexibility_analyzer = BackboneFlexibility()
        self.domain_analyzer = DomainMovements()

    def analyze_sequence(self, sequence: str, output_dir: str) -> Dict[str, Any]:
        """Run complete analysis pipeline from sequence"""
        try:
            # Initialize progress tracking
            self.progress_tracker = ProgressTracker(total_steps=4,
                                                  description="Sequence Analysis")
            self.progress_tracker.start()

            # Predict structure
            predicted_structure = self.structure_predictor.predict_structure(sequence)
            self.progress_tracker.update()

            # Convert structure format
            converted_structure = self.structure_converter.convert_structure(
                predicted_structure,
                output_format='pdb'
            )
            self.progress_tracker.update()

            # Save structure
            structure_file = f"{output_dir}/predicted_structure.pdb"
            self.structure_converter.save_structure(
                converted_structure,
                structure_file
            )
            self.progress_tracker.update()

            # Analyze flexibility
            self.flexibility_analyzer.load_trajectory(structure_file, structure_file)
            flexibility_results = {
                'rmsd': self.flexibility_analyzer.calculate_rmsd().tolist(),
                'rmsf': self.flexibility_analyzer.calculate_rmsf().tolist(),
                'secondary_structure': self.flexibility_analyzer.analyze_secondary_structure()
            }

            # Analyze domains
            self.domain_analyzer.load_trajectory(structure_file, structure_file)
            domains = self.domain_analyzer.identify_domains()
            domain_results = {
                'domains': domains,
                'hinge_regions': self.domain_analyzer.analyze_hinge_regions(domains)
            }
            self.progress_tracker.update()

            # Compile results
            results = {
                'structure_file': structure_file,
                'flexibility_analysis': flexibility_results,
                'domain_analysis': domain_results,
                'progress': self.progress_tracker.get_progress(),
                'checkpoints': self.progress_tracker.get_checkpoints()
            }

            return results

        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            raise

    def save_results(self, results: Dict[str, Any], filename: str):
        """Save analysis results"""
        try:
            self.data_handler.save_trajectory(
                traj_data=results['flexibility_analysis']['rmsd'],
                metadata={
                    'analysis_type': 'sequence',
                    'components': list(results.keys())
                },
                filename=filename
            )
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
