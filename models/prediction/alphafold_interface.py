"""
AlphaFold3 Interface Module

This module provides an interface to AlphaFold3's structure prediction pipeline,
handling model loading, prediction, and confidence score integration.
"""

import os
import logging
from typing import Dict, Tuple, Optional, Union
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from Bio import SeqIO
from Bio.Seq import Seq

from .structure_converter import StructureConverter

class AlphaFold3Interface:
    """Interface to AlphaFold3's structure prediction pipeline."""

    def __init__(self,
                 model_dir: str,
                 max_gpu_memory: float = 16.0):
        """Initialize AlphaFold3 interface.

        Args:
            model_dir: Directory containing AlphaFold3 model parameters
            max_gpu_memory: Maximum GPU memory to use in GB
        """
        self.model_dir = model_dir
        self.max_gpu_memory = max_gpu_memory
        self.converter = StructureConverter()

        # Configure JAX for GPU
        jax.config.update('jax_platform_name', 'gpu')
        jax.config.update('jax_enable_x64', True)

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize AlphaFold3 model and load weights."""
        try:
            # Import AlphaFold3 modules (assuming they're in PYTHONPATH)
            from alphafold3.model import config
            from alphafold3.model import model
            from alphafold3.model import modules

            # Load model configuration
            self.model_config = config.model_config()
            self.model_config.max_gpu_memory = self.max_gpu_memory

            # Initialize model
            self.model = model.AlphaFold3Model(self.model_config)

            # Load model parameters
            self._load_parameters()

        except ImportError as e:
            logging.error(f"Failed to import AlphaFold3: {e}")
            raise ImportError("AlphaFold3 must be installed and in PYTHONPATH")

    def _load_parameters(self):
        """Load model parameters from checkpoint."""
        params_path = os.path.join(self.model_dir, 'params.npz')
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Model parameters not found at {params_path}")

        try:
            self.params = np.load(params_path)
        except Exception as e:
            logging.error(f"Failed to load model parameters: {e}")
            raise

    def predict_structure(self,
                         sequence: str,
                         temperature: float = 0.1,
                         num_samples: int = 1) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Predict protein structure using AlphaFold3.

        Args:
            sequence: Amino acid sequence
            temperature: Sampling temperature
            num_samples: Number of structure samples to generate

        Returns:
            Tuple of (atom positions, confidence scores)
        """
        # Validate sequence
        if not self._validate_sequence(sequence):
            raise ValueError("Invalid amino acid sequence")

        try:
            # Prepare input features
            features = self._prepare_features(sequence)

            # Run prediction
            @jax.jit
            def predict_fn(params, features, key):
                return self.model.predict(params, features, key)

            key = jax.random.PRNGKey(0)
            predictions = []
            confidence_scores = []

            for i in range(num_samples):
                key, subkey = jax.random.split(key)
                pred = predict_fn(self.params, features, subkey)
                predictions.append(pred['positions'])
                confidence_scores.append({
                    'plddt': pred['plddt'],
                    'pae': pred.get('pae', None)
                })

            # Average predictions and confidence scores
            avg_positions = jnp.mean(jnp.stack(predictions), axis=0)
            avg_confidence = {
                'plddt': jnp.mean(jnp.stack([s['plddt'] for s in confidence_scores]), axis=0),
                'pae': jnp.mean(jnp.stack([s['pae'] for s in confidence_scores
                                         if s['pae'] is not None]), axis=0)
                if confidence_scores[0]['pae'] is not None else None
            }

            return avg_positions, avg_confidence

        except Exception as e:
            logging.error(f"Structure prediction failed: {e}")
            raise

    def _prepare_features(self, sequence: str) -> Dict[str, jnp.ndarray]:
        """Prepare input features for AlphaFold3.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary of input features
        """
        # Convert sequence to features
        features = {
            'aatype': self._sequence_to_onehot(sequence),
            'residue_index': jnp.arange(len(sequence)),
            'seq_length': jnp.array(len(sequence)),
        }

        return features

    def _sequence_to_onehot(self, sequence: str) -> jnp.ndarray:
        """Convert amino acid sequence to one-hot encoding.

        Args:
            sequence: Amino acid sequence

        Returns:
            One-hot encoded sequence [L, 20]
        """
        # Amino acid to index mapping
        aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

        # Convert to indices
        indices = jnp.array([aa_to_idx.get(aa, -1) for aa in sequence])

        # Convert to one-hot
        onehot = jax.nn.one_hot(indices, num_classes=20)

        return onehot

    def _validate_sequence(self, sequence: str) -> bool:
        """Validate amino acid sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            True if sequence is valid
        """
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        return all(aa in valid_aas for aa in sequence)

    def get_confidence_metrics(self,
                             confidence_scores: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate confidence metrics from prediction.

        Args:
            confidence_scores: Dictionary with plddt and pae scores

        Returns:
            Dictionary of confidence metrics
        """
        metrics = {}

        # pLDDT metrics
        plddt = confidence_scores['plddt']
        metrics['mean_plddt'] = float(np.mean(plddt))
        metrics['min_plddt'] = float(np.min(plddt))
        metrics['max_plddt'] = float(np.max(plddt))

        # PAE metrics if available
        pae = confidence_scores.get('pae')
        if pae is not None:
            metrics['mean_pae'] = float(np.mean(pae))
            metrics['max_pae'] = float(np.max(pae))

        return metrics

    def predict_and_convert(self,
                          sequence: str,
                          temperature: float = 0.1) -> Tuple[object, Dict[str, float]]:
        """Predict structure and convert to OpenMM format.

        Args:
            sequence: Amino acid sequence
            temperature: Sampling temperature

        Returns:
            Tuple of (OpenMM structure, confidence metrics)
        """
        # Predict structure
        positions, confidence = self.predict_structure(sequence, temperature)

        # Convert to OpenMM
        structure = self.converter.alphafold_to_openmm(
            positions,
            sequence,
            confidence['plddt']
        )

        # Calculate confidence metrics
        metrics = self.get_confidence_metrics(confidence)

        return structure, metrics
