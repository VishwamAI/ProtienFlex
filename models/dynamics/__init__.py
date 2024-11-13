"""
ProtienFlex Molecular Dynamics Module

This module provides enhanced molecular dynamics capabilities with
specialized tools for protein flexibility analysis.

Example usage:
    from models.dynamics import EnhancedSampling, FlexibilityAnalysis, SimulationValidator

    # Setup enhanced sampling simulation
    simulator = EnhancedSampling(structure)
    replicas = simulator.setup_replica_exchange(n_replicas=4)
    stats = simulator.run_replica_exchange(n_steps=1000000)

    # Analyze flexibility
    analyzer = FlexibilityAnalysis(trajectory)
    profile = analyzer.calculate_flexibility_profile()

    # Validate results
    validator = SimulationValidator(trajectory)
    report = validator.generate_validation_report()
"""

from .simulation import EnhancedSampling
from .analysis import FlexibilityAnalysis
from .validation import SimulationValidator

__all__ = [
    'EnhancedSampling',
    'FlexibilityAnalysis',
    'SimulationValidator'
]

__version__ = '0.1.0'
