"""
ProtienFlex Analysis Pipeline

This package provides a comprehensive pipeline for protein flexibility analysis,
combining structure prediction, molecular dynamics, and flexibility analysis.

Example usage:
    from models.pipeline import FlexibilityPipeline, AnalysisPipeline

    # Single protein analysis
    pipeline = FlexibilityPipeline('/path/to/model', '/path/to/output')
    results = pipeline.analyze_sequence('MKLLVLGLRSGSGKS', name='protein1')

    # Multiple protein analysis
    analysis = AnalysisPipeline('/path/to/model', '/path/to/output')
    proteins = [
        {'name': 'protein1', 'sequence': 'MKLLVLGLRSGSGKS'},
        {'name': 'protein2', 'sequence': 'MALWMRLLPLLALLALWGPD'}
    ]
    results = analysis.analyze_proteins(proteins)
"""

from .flexibility_pipeline import FlexibilityPipeline
from .analysis_pipeline import AnalysisPipeline

__all__ = [
    'FlexibilityPipeline',
    'AnalysisPipeline'
]

__version__ = '0.1.0'
