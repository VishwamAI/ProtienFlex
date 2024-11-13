"""
ProtienFlex Flexibility Analysis Module

This package provides comprehensive tools for analyzing protein flexibility
at multiple scales, from atomic fluctuations to domain movements.

Example usage:
    from models.flexibility import BackboneFlexibility, SidechainMobility, DomainMovements

    # Initialize analyzers
    backbone = BackboneFlexibility('protein.pdb')
    sidechain = SidechainMobility('protein.pdb')
    domains = DomainMovements('protein.pdb')

    # Analyze trajectory
    rmsf = backbone.calculate_rmsf(trajectory)
    bfactors = backbone.predict_bfactors(trajectory)

    # Analyze side-chain mobility
    rotamers = sidechain.analyze_rotamer_distribution(trajectory, residue_index=10)

    # Analyze domain movements
    domain_list = domains.identify_domains(trajectory)
    motion = domains.analyze_domain_motion(trajectory, domain_list[0], domain_list[1])
"""

from .backbone_flexibility import BackboneFlexibility
from .sidechain_mobility import SidechainMobility
from .domain_movements import DomainMovements

__all__ = [
    'BackboneFlexibility',
    'SidechainMobility',
    'DomainMovements'
]

__version__ = '0.1.0'
