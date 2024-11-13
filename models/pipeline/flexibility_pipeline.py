"""Flexibility analysis pipeline implementation"""
import logging
from typing import Dict, Any, Optional
from ..flexibility import BackboneFlexibility, SidechainMobility, DomainMovements
from ..optimization import GPUManager, DataHandler, ProgressTracker, CheckpointManager

logger = logging.getLogger(__name__)

class FlexibilityPipeline:
    """Pipeline for comprehensive protein flexibility analysis"""

    def __init__(self, gpu_required: bool = True):
        self.gpu_manager = GPUManager() if gpu_required else None
        self.data_handler = DataHandler()
        self.progress_tracker = None
        self.checkpoint_manager = CheckpointManager()

        # Analysis components
        self.backbone_analyzer = BackboneFlexibility()
        self.sidechain_analyzer = SidechainMobility()
        self.domain_analyzer = DomainMovements()

    def analyze_protein(self, pdb_file: str, trajectory_file: str) -> Dict[str, Any]:
        """Run complete flexibility analysis pipeline"""
        try:
            # Initialize progress tracking
            self.progress_tracker = ProgressTracker(total_steps=4,
                                                  description="Flexibility Analysis")
            self.progress_tracker.start()

            # Load and prepare data
            self.backbone_analyzer.load_trajectory(trajectory_file, pdb_file)
            self.sidechain_analyzer.load_trajectory(trajectory_file, pdb_file)
            self.domain_analyzer.load_trajectory(trajectory_file, pdb_file)
            self.progress_tracker.update()

            # Analyze backbone flexibility
            backbone_results = {
                'rmsd': self.backbone_analyzer.calculate_rmsd().tolist(),
                'rmsf': self.backbone_analyzer.calculate_rmsf().tolist(),
                'secondary_structure': self.backbone_analyzer.analyze_secondary_structure()
            }
            self.progress_tracker.update()

            # Analyze sidechain mobility
            chi_angles = self.sidechain_analyzer.calculate_chi_angles()
            sidechain_results = {
                'chi_angles': {k: v.tolist() for k, v in chi_angles.items()},
                'rotamer_populations': self.sidechain_analyzer.analyze_rotamer_populations(chi_angles)
            }
            self.progress_tracker.update()

            # Analyze domain movements
            domains = self.domain_analyzer.identify_domains()
            domain_results = {
                'domains': domains,
                'movements': self.domain_analyzer.calculate_domain_movements(domains),
                'hinge_regions': self.domain_analyzer.analyze_hinge_regions(domains)
            }
            self.progress_tracker.update()

            # Compile results
            results = {
                'backbone_flexibility': backbone_results,
                'sidechain_mobility': sidechain_results,
                'domain_analysis': domain_results,
                'progress': self.progress_tracker.get_progress(),
                'checkpoints': self.progress_tracker.get_checkpoints()
            }

            return results

        except Exception as e:
            logger.error(f"Error in flexibility analysis pipeline: {e}")
            raise

    def save_results(self, results: Dict[str, Any], filename: str):
        """Save analysis results"""
        try:
            self.data_handler.save_trajectory(
                traj_data=results['backbone_flexibility']['rmsd'],
                metadata={
                    'analysis_type': 'flexibility',
                    'components': list(results.keys())
                },
                filename=filename
            )
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
