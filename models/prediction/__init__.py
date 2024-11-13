"""
ProtienFlex Structure Prediction Module

This package provides integration with AlphaFold3's structure prediction pipeline
and utilities for converting between different structure formats.

Example usage:
    from models.prediction import AlphaFold3Interface, StructureConverter

    # Initialize predictor
    predictor = AlphaFold3Interface('/path/to/model_dir')

    # Predict structure
    sequence = 'MKLLVLGLRSGSGKS'
    structure, confidence = predictor.predict_and_convert(sequence)

    # Convert between formats
    converter = StructureConverter()
    mdtraj_struct = converter.openmm_to_mdtraj(structure)
"""

from .alphafold_interface import AlphaFold3Interface
from .structure_converter import StructureConverter

__all__ = [
    'AlphaFold3Interface',
    'StructureConverter'
]

__version__ = '0.1.0'
