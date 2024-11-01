"""
Unified Model for ProteinFlex.
Combines local generative model with external API capabilities.
"""

from typing import Dict, Any, List, Optional, Union
import torch
import numpy as np

from .api_integration import APIManager
from .protein_generator import ProteinGenerativeModel
from .structure_predictor import StructurePredictor
from .virtual_screening import VirtualScreeningModel

class UnifiedProteinModel:
    """
    Unified model that combines local generative capabilities with external APIs.
    Provides seamless integration between local and cloud-based predictions.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the unified model with both local and API-based capabilities.

        Args:
            use_gpu: Whether to use GPU for local model computations
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Initialize local models
        self.local_generator = ProteinGenerativeModel().to(self.device)
        self.structure_predictor = StructurePredictor().to(self.device)
        self.screening_model = VirtualScreeningModel().to(self.device)

        # Initialize API manager
        self.api_manager = APIManager()

    def generate_sequence(self,
                         prompt: str,
                         use_apis: bool = True,
                         temperature: float = 0.7,
                         max_length: int = 512) -> Dict[str, Any]:
        """
        Generate protein sequence using both local model and APIs if available.

        Args:
            prompt: Input prompt describing desired protein properties
            use_apis: Whether to also use external APIs
            temperature: Sampling temperature
            max_length: Maximum sequence length

        Returns:
            Dictionary containing generated sequences and confidence scores
        """
        results = {
            'local': self.local_generator.generate(
                prompt,
                temperature=temperature,
                max_length=max_length
            )
        }

        if use_apis:
            for api_name in self.api_manager.apis:
                try:
                    api = self.api_manager.get_api(api_name)
                    results[api_name] = api.generate(prompt)
                except Exception as e:
                    results[api_name] = {'error': str(e)}

        return results

    def predict_structure(self,
                         sequence: str,
                         use_apis: bool = True) -> Dict[str, Any]:
        """
        Predict protein structure using both local model and APIs if available.

        Args:
            sequence: Input protein sequence
            use_apis: Whether to also use external APIs

        Returns:
            Dictionary containing predicted structures and confidence scores
        """
        results = {
            'local': self.structure_predictor.predict(sequence)
        }

        if use_apis:
            api_results = self.api_manager.analyze_with_all(sequence)
            results.update(api_results)

        return results

    def analyze_stability(self,
                         sequence: str,
                         structure: Optional[torch.Tensor] = None,
                         use_apis: bool = True) -> Dict[str, Any]:
        """
        Analyze protein stability using both local model and APIs.

        Args:
            sequence: Input protein sequence
            structure: Optional 3D structure tensor
            use_apis: Whether to also use external APIs

        Returns:
            Dictionary containing stability analyses and confidence scores
        """
        results = {
            'local': {
                'stability_score': self.local_generator.analyze_stability(sequence),
                'structure_quality': self.structure_predictor.evaluate_quality(structure)
                if structure is not None else None
            }
        }

        if use_apis:
            for api_name in self.api_manager.apis:
                try:
                    api = self.api_manager.get_api(api_name)
                    results[api_name] = api.analyze_protein(sequence)
                except Exception as e:
                    results[api_name] = {'error': str(e)}

        return results

    def screen_compounds(self,
                        protein_structure: torch.Tensor,
                        compounds: List[str],
                        use_apis: bool = True) -> Dict[str, Any]:
        """
        Screen compounds against protein structure using both local model and APIs.

        Args:
            protein_structure: 3D protein structure tensor
            compounds: List of compound SMILES strings
            use_apis: Whether to also use external APIs

        Returns:
            Dictionary containing screening results and binding scores
        """
        results = {
            'local': self.screening_model.screen(protein_structure, compounds)
        }

        if use_apis:
            for api_name in self.api_manager.apis:
                try:
                    api = self.api_manager.get_api(api_name)
                    results[api_name] = api.analyze_protein(
                        f"Analyze binding potential for protein with compounds: {', '.join(compounds)}"
                    )
                except Exception as e:
                    results[api_name] = {'error': str(e)}

        return results

    @torch.no_grad()
    def ensemble_predict(self,
                        sequence: str,
                        weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Make ensemble predictions combining local and API results.

        Args:
            sequence: Input protein sequence
            weights: Optional dictionary of weights for different models

        Returns:
            Dictionary containing ensemble predictions and confidence scores
        """
        # Get predictions from all sources
        structure_predictions = self.predict_structure(sequence, use_apis=True)
        stability_analysis = self.analyze_stability(sequence, use_apis=True)

        # Use equal weights if none provided
        if weights is None:
            available_models = len(structure_predictions)
            weights = {model: 1.0/available_models for model in structure_predictions}

        # Combine predictions using weights
        ensemble_results = {
            'structure': structure_predictions,
            'stability': stability_analysis,
            'weights': weights,
            'confidence_score': sum(
                weight * (1 - float(isinstance(pred.get('error', None), str)))
                for model, (weight, pred) in zip(
                    weights.items(),
                    structure_predictions.items()
                )
            )
        }

        return ensemble_results
