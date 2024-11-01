"""
Text to Protein Generation Module.
Integrates Gemini API for enhanced protein sequence generation.
"""

import os
from typing import Dict, Any, List, Optional, Union
import torch
import google.generativeai as genai
from .unified_model import UnifiedProteinModel

class TextToProteinGenerator:
    """
    Text to Protein Generator using Gemini API and local models.
    Provides enhanced protein sequence generation from text descriptions.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize the text to protein generator.

        Args:
            use_gpu: Whether to use GPU for local computations
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.unified_model = UnifiedProteinModel(use_gpu=use_gpu)

        # Initialize Gemini API
        gemini_api_key = os.getenv('Gemini_api')
        if not gemini_api_key:
            raise ValueError("Gemini API key not found in environment variables")
        genai.configure(api_key=gemini_api_key)

        # Initialize Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-pro')

    def _format_protein_prompt(self, description: str) -> str:
        """Format the input description into a protein-specific prompt."""
        return f"""
        Task: Generate a protein sequence based on the following description.
        Description: {description}

        Requirements:
        1. The sequence should use standard amino acid letters (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
        2. Consider protein stability and folding properties
        3. Include key functional domains if specified
        4. Optimize for the described functionality

        Please provide:
        1. The protein sequence
        2. Key features and properties
        3. Predicted stability assessment
        """

    async def generate_protein(self,
                             description: str,
                             temperature: float = 0.7,
                             max_length: int = 512) -> Dict[str, Any]:
        """
        Generate protein sequence from text description using Gemini API and local models.

        Args:
            description: Text description of desired protein
            temperature: Sampling temperature
            max_length: Maximum sequence length

        Returns:
            Dictionary containing generated sequence and analysis
        """
        # Format prompt for Gemini
        prompt = self._format_protein_prompt(description)

        try:
            # Get Gemini's response
            gemini_response = await self.gemini_model.generate_content(prompt)
            gemini_text = gemini_response.text

            # Extract protein sequence from Gemini's response
            # Assuming the sequence is the first continuous string of amino acids
            import re
            sequence_match = re.search(r'[ACDEFGHIKLMNPQRSTVWY]+', gemini_text)
            if not sequence_match:
                raise ValueError("No valid protein sequence found in Gemini's response")

            sequence = sequence_match.group(0)

            # Analyze the sequence using unified model
            analysis = self.unified_model.analyze_stability(sequence)
            structure = self.unified_model.predict_structure(sequence)

            return {
                'sequence': sequence,
                'gemini_response': gemini_text,
                'stability_analysis': analysis,
                'predicted_structure': structure,
                'source': 'gemini+local',
                'description': description
            }

        except Exception as e:
            # Fallback to local generation if Gemini fails
            local_result = self.unified_model.generate_sequence(
                description,
                temperature=temperature,
                max_length=max_length
            )

            return {
                'sequence': local_result['local']['sequence'],
                'error': str(e),
                'stability_analysis': self.unified_model.analyze_stability(
                    local_result['local']['sequence']
                ),
                'predicted_structure': self.unified_model.predict_structure(
                    local_result['local']['sequence']
                ),
                'source': 'local_fallback',
                'description': description
            }

    def validate_sequence(self, sequence: str) -> Dict[str, Any]:
        """
        Validate generated protein sequence.

        Args:
            sequence: Generated protein sequence

        Returns:
            Dictionary containing validation results
        """
        # Check sequence validity
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        is_valid = all(aa in valid_aas for aa in sequence)

        # Get detailed analysis if sequence is valid
        if is_valid:
            stability = self.unified_model.analyze_stability(sequence)
            structure = self.unified_model.predict_structure(sequence)

            return {
                'is_valid': True,
                'length': len(sequence),
                'stability': stability,
                'structure': structure
            }

        return {
            'is_valid': False,
            'error': 'Invalid amino acids in sequence',
            'invalid_residues': [aa for aa in sequence if aa not in valid_aas]
        }


    def batch_generate(self,
                      descriptions: List[str],
                      temperature: float = 0.7,
                      max_length: int = 512) -> List[Dict[str, Any]]:
        """
        Generate multiple protein sequences from descriptions.

        Args:
            descriptions: List of protein descriptions
            temperature: Sampling temperature
            max_length: Maximum sequence length

        Returns:
            List of dictionaries containing generated sequences and analyses
        """
        import asyncio

        async def _batch_generate():
            tasks = [
                self.generate_protein(desc, temperature, max_length)
                for desc in descriptions
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        # Run batch generation
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(_batch_generate())

        # Process results and handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    'error': str(result),
                    'source': 'error'
                })
            else:
                processed_results.append(result)

        return processed_results
