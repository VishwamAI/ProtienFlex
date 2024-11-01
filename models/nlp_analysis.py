# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from Bio.PDB import *
import biotite.structure as struc
import biotite.structure.io as strucio

logger = logging.getLogger(__name__)

class ProteinNLPAnalyzer:
    def __init__(self):
        try:
            # Initialize ESM-based protein language model for sequence analysis
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.model = AutoModelForSequenceClassification.from_pretrained("facebook/esm2_t33_650M_UR50D")
            self.model.eval()

            # Initialize text generation pipeline for natural language descriptions
            self.nlp_pipeline = pipeline("text2text-generation", model="facebook/bart-large")

            logger.info("NLP analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error loading NLP model: {e}")
            raise

    def analyze_sequence(self, sequence):
        """Analyze protein sequence and generate natural language description"""
        try:
            # Basic protein analysis
            protein_analysis = ProteinAnalysis(sequence)
            molecular_weight = protein_analysis.molecular_weight()
            isoelectric_point = protein_analysis.isoelectric_point()
            aromaticity = protein_analysis.aromaticity()
            instability_index = protein_analysis.instability_index()
            secondary_structure = protein_analysis.secondary_structure_fraction()

            # Generate sequence embeddings for advanced analysis
            inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.logits

            # Calculate sequence properties
            hydrophobicity = self._calculate_hydrophobicity(sequence)
            complexity = self._calculate_sequence_complexity(sequence)
            domains = self._predict_domains(sequence)
            stability = self._analyze_stability(sequence)
            function_prediction = self._predict_function(embeddings)

            # Generate comprehensive natural language description
            description = self._generate_description(
                sequence_length=len(sequence),
                molecular_weight=molecular_weight,
                isoelectric_point=isoelectric_point,
                aromaticity=aromaticity,
                instability_index=instability_index,
                secondary_structure=secondary_structure,
                hydrophobicity=hydrophobicity,
                complexity=complexity,
                domains=domains,
                stability=stability,
                function=function_prediction
            )

            return description

        except Exception as e:
            logger.error(f"Error in sequence analysis: {e}")
            return "Error analyzing sequence"

    def _calculate_hydrophobicity(self, sequence):
        """Calculate average hydrophobicity using Kyte-Doolittle scale"""
        hydropathy_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        return np.mean([hydropathy_scale.get(aa, 0) for aa in sequence])

    def _calculate_sequence_complexity(self, sequence):
        """Calculate sequence complexity based on amino acid diversity"""
        unique_aa = len(set(sequence))
        return (unique_aa / 20) * 100  # Normalize to percentage

    def _predict_domains(self, sequence):
        """Predict protein domains using sequence patterns"""
        domains = []
        # Simple domain prediction based on sequence patterns
        if len(sequence) > 50:
            for i in range(0, len(sequence) - 50, 25):
                segment = sequence[i:i+50]
                hydrophobicity = self._calculate_hydrophobicity(segment)
                if hydrophobicity > 2.0:
                    domains.append(f"Hydrophobic domain ({i+1}-{i+50})")
                elif hydrophobicity < -2.0:
                    domains.append(f"Hydrophilic domain ({i+1}-{i+50})")
        return domains

    def _analyze_stability(self, sequence):
        """Analyze protein stability based on sequence properties"""
        try:
            analysis = ProteinAnalysis(sequence)
            instability_index = analysis.instability_index()
            flexibility = analysis.flexibility()
            stability = {
                'index': instability_index,
                'is_stable': instability_index < 40,
                'flexibility': np.mean(flexibility)
            }
            return stability
        except Exception as e:
            logger.error(f"Error in stability analysis: {e}")
            return None

    def _predict_function(self, embeddings):
        """Predict protein function based on sequence embeddings"""
        # Simple function prediction based on embedding patterns
        embedding_mean = torch.mean(embeddings).item()
        if embedding_mean > 0.5:
            return "Likely enzymatic activity"
        elif embedding_mean > 0:
            return "Possible structural role"
        else:
            return "Potential regulatory function"

    def _generate_description(self, **properties):
        """Generate natural language description of protein properties"""
        # Basic information
        description = f"""This protein sequence consists of {properties['sequence_length']} amino acids with a molecular weight of {properties['molecular_weight']:.2f} Da. """

        # Structural information
        helix, sheet, coil = properties['secondary_structure']
        description += f"""The predicted secondary structure composition includes {helix:.1%} alpha-helix, {sheet:.1%} beta-sheet, and {coil:.1%} random coil. """

        # Stability information
        if properties['stability']:
            stability_status = "stable" if properties['stability']['is_stable'] else "potentially unstable"
            description += f"""The protein is predicted to be {stability_status} with an instability index of {properties['stability']['index']:.1f}. """
            description += f"""It shows {'high' if properties['stability']['flexibility'] > 0.5 else 'moderate' if properties['stability']['flexibility'] > 0.3 else 'low'} flexibility. """

        # Chemical properties
        description += f"""It has an isoelectric point of {properties['isoelectric_point']:.2f} and an aromaticity of {properties['aromaticity']:.2f}. """

        # Domain information
        if properties['domains']:
            description += f"""The sequence contains {len(properties['domains'])} predicted domains: {', '.join(properties['domains'])}. """

        # Sequence properties
        hydro_desc = "hydrophobic" if properties['hydrophobicity'] > 0 else "hydrophilic"
        description += f"""The sequence is generally {hydro_desc} with an average hydrophobicity of {properties['hydrophobicity']:.2f}. """
        description += f"""The sequence complexity is {properties['complexity']:.1f}%, indicating {'high' if properties['complexity'] > 75 else 'moderate' if properties['complexity'] > 50 else 'low'} amino acid diversity. """

        # Functional prediction
        description += f"""Based on the sequence analysis, this protein may have {properties['function'].lower()}."""

        return description
