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

import torch
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification
from Bio import SeqIO
from Bio.PDB import *
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

class ProteinLanguageModel:
    def __init__(self):
        # Initialize ESM-2 model for protein language modeling
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize stability prediction model
        self.stability_model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/esm-1b-stability-prediction"
        ).to(self.device)

        # Initialize text generation pipeline for protein descriptions
        self.text_pipeline = pipeline("text2text-generation",
                                    model="facebook/bart-large-cnn",
                                    device=0 if torch.cuda.is_available() else -1)

        # Initialize real-time adjustment parameters
        self.attention_threshold = 0.5
        self.confidence_threshold = 0.7
        self.update_rate = 0.1

    def analyze_sequence(self, sequence, adjust_params=True):
        """Analyze protein sequence using ESM-2 model with real-time adjustments."""
        try:
            # Basic protein analysis
            protein_analysis = ProteinAnalysis(sequence)
            basic_properties = {
                'start': 0,
                'end': len(sequence),
                'score': float(protein_analysis.instability_index() / 100),
                'type': 'basic_properties',
                'properties': {
                    'molecular_weight': protein_analysis.molecular_weight(),
                    'aromaticity': protein_analysis.aromaticity(),
                    'instability_index': protein_analysis.instability_index(),
                    'isoelectric_point': protein_analysis.isoelectric_point()
                }
            }

            # Tokenize sequence
            inputs = self.tokenizer(sequence, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings with real-time adjustment
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state
                attention_weights = torch.mean(outputs.attentions[-1], dim=1)
                residue_importance = torch.mean(attention_weights, dim=1).squeeze()

                if adjust_params:
                    self._adjust_parameters(residue_importance)

                # Find region with highest attention for analysis focus
                window_size = 15
                max_attention_start = 0
                max_attention_score = 0
                for i in range(len(sequence) - window_size + 1):
                    window_score = float(residue_importance[i:i+window_size].mean())
                    if window_score > max_attention_score:
                        max_attention_score = window_score
                        max_attention_start = i

            return {
                'start': max_attention_start,
                'end': max_attention_start + window_size,
                'score': float(max_attention_score),
                'type': 'sequence_analysis',
                'embeddings': {
                    'start': 0,
                    'end': len(sequence),
                    'score': float(max_attention_score),
                    'type': 'embedding_data',
                    'data': embeddings.cpu().numpy()
                },
                'residue_importance': {
                    'start': 0,
                    'end': len(sequence),
                    'score': float(max_attention_score),
                    'type': 'residue_data',
                    'data': residue_importance.cpu().numpy()
                },
                'basic_properties': basic_properties,
                'analysis_region': {
                    'start': max_attention_start,
                    'end': max_attention_start + window_size,
                    'score': float(max_attention_score),
                    'type': 'attention_region'
                }
            }
        except Exception as e:
            return {
                'start': 0,
                'end': min(15, len(sequence)),
                'score': 0.0,
                'type': 'sequence_analysis_error',
                'error': str(e)
            }

    def predict_mutations(self, sequence, positions):
        """Predict effects of mutations at specified positions."""
        try:
            results = []
            best_mutation = None
            best_score = -float('inf')
            best_position = positions[0] if positions else 0

            for pos in positions:
                mutations = []
                original_aa = sequence[pos]
                for new_aa in "ACDEFGHIKLMNPQRSTVWY":
                    if new_aa != original_aa:
                        # Create mutated sequence
                        mutated_seq = sequence[:pos] + new_aa + sequence[pos+1:]

                        # Predict stability change
                        stability_change = self._predict_stability_change(sequence, mutated_seq)

                        # Calculate mutation impact
                        impact = self._calculate_mutation_impact(sequence, pos, new_aa)

                        # Calculate confidence
                        confidence = self._calculate_confidence(stability_change, impact)

                        mutation_info = {
                            'mutation': f"{original_aa}{pos+1}{new_aa}",
                            'stability_change': stability_change,
                            'impact': impact,
                            'confidence': confidence
                        }
                        mutations.append(mutation_info)

                        # Track best mutation
                        if confidence > best_score:
                            best_score = confidence
                            best_mutation = mutation_info
                            best_position = pos

                results.append({
                    'position': pos,
                    'mutations': sorted(mutations, key=lambda x: x['confidence'], reverse=True)
                })

            # Return dictionary with required fields using best mutation
            return {
                'start': best_position,
                'end': best_position + 1,
                'score': float(best_score / 100),  # Normalize to 0-1
                'type': 'mutation_analysis',
                'best_mutation': best_mutation,
                'all_results': results
            }
        except Exception as e:
            return {
                'start': positions[0] if positions else 0,
                'end': (positions[0] if positions else 0) + 1,
                'score': 0.0,
                'type': 'mutation_analysis_error',
                'error': str(e)
            }

    def analyze_drug_binding(self, sequence, ligand_features=None):
        """Analyze potential drug binding sites and interactions."""
        try:
            sequence_features = self.analyze_sequence(sequence)

            # Identify potential binding pockets
            binding_sites = self.get_interaction_sites(sequence_features)

            # Enhanced binding site analysis
            detailed_sites = []
            best_site = None
            best_score = -float('inf')
            best_position = 0

            for site_idx, score in zip(binding_sites['sites'], binding_sites['scores']):
                site_analysis = {
                    'position': site_idx,
                    'score': float(score),
                    'properties': self._analyze_binding_site_properties(sequence, site_idx),
                    'druggability': self._calculate_druggability_score(sequence, site_idx)
                }
                if ligand_features:
                    site_analysis['ligand_compatibility'] = self._predict_ligand_compatibility(
                        sequence, site_idx, ligand_features
                    )
                detailed_sites.append(site_analysis)

                # Track best binding site
                if score > best_score:
                    best_score = score
                    best_site = site_analysis
                    best_position = site_idx

            # If no binding sites found, use default values
            if not best_site:
                return {
                    'start': 0,
                    'end': min(15, len(sequence)),
                    'score': 0.0,
                    'type': 'binding_site_analysis',
                    'message': 'No significant binding sites found',
                    'all_sites': []
                }

            # Return dictionary with required fields using best binding site
            window_size = 15  # Size of binding site window
            return {
                'start': max(0, best_position - window_size//2),
                'end': min(len(sequence), best_position + window_size//2),
                'score': float(best_score),
                'type': 'binding_site_analysis',
                'best_site': best_site,
                'all_sites': detailed_sites
            }

        except Exception as e:
            return {
                'start': 0,
                'end': min(15, len(sequence)),
                'score': 0.0,
                'type': 'binding_site_analysis_error',
                'error': str(e)
            }

    def _predict_stability_change(self, original_seq, mutated_seq):
        """Predict stability change for mutation."""
        inputs = self.tokenizer([original_seq, mutated_seq], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.stability_model(**inputs)
            scores = torch.sigmoid(outputs.logits)
            return float(scores[1] - scores[0])

    def _calculate_mutation_impact(self, sequence, position, new_aa):
        """Calculate overall impact of mutation."""
        # Analyze sequence context
        context_features = self.analyze_sequence(sequence)
        position_importance = context_features['residue_importance'][position]

        # Calculate conservation score
        conservation = self._calculate_conservation_score(sequence, position)

        return (position_importance + conservation) / 2

    def _calculate_confidence(self, stability_change, impact):
        """Calculate confidence score for predictions."""
        base_confidence = (abs(stability_change) + impact) / 2
        return min(100, max(0, base_confidence * 100))

    def _adjust_parameters(self, importance_scores):
        """Adjust model parameters based on prediction confidence."""
        mean_importance = torch.mean(importance_scores)
        if mean_importance > self.attention_threshold:
            self.attention_threshold += self.update_rate
        else:
            self.attention_threshold -= self.update_rate
        self.attention_threshold = max(0.1, min(0.9, self.attention_threshold))

    def _analyze_binding_site_properties(self, sequence, position):
        """Analyze properties of potential binding site."""
        window_size = 5
        start = max(0, position - window_size)
        end = min(len(sequence), position + window_size + 1)
        site_sequence = sequence[start:end]

        analysis = ProteinAnalysis(site_sequence)
        return {
            'hydrophobicity': analysis.gravy(),
            'flexibility': analysis.flexibility(),
            'surface_accessibility': self._calculate_surface_accessibility(sequence, position)
        }

    def _calculate_surface_accessibility(self, sequence, position):
        """Calculate surface accessibility score."""
        window_size = 7
        start = max(0, position - window_size)
        end = min(len(sequence), position + window_size + 1)
        site_sequence = sequence[start:end]

        # Calculate relative surface accessibility using amino acid properties
        aa_exposure = {
            'A': 0.48, 'R': 0.95, 'N': 0.85, 'D': 0.85, 'C': 0.32,
            'Q': 0.87, 'E': 0.93, 'G': 0.51, 'H': 0.84, 'I': 0.28,
            'L': 0.28, 'K': 0.97, 'M': 0.40, 'F': 0.35, 'P': 0.58,
            'S': 0.70, 'T': 0.71, 'W': 0.39, 'Y': 0.61, 'V': 0.27
        }

        # Calculate weighted accessibility score
        center_weight = 2.0
        total_weight = center_weight + 2 * window_size
        weighted_score = 0

        for i, aa in enumerate(site_sequence):
            weight = center_weight if i == window_size else 1.0
            weighted_score += weight * aa_exposure.get(aa, 0.5)

        return weighted_score / total_weight

    def _calculate_druggability_score(self, sequence, position):
        """Calculate druggability score for binding site."""
        properties = self._analyze_binding_site_properties(sequence, position)
        return (properties['hydrophobicity'] + properties['surface_accessibility']) / 2

    def _predict_ligand_compatibility(self, sequence, position, ligand_features):
        """Predict compatibility between binding site and ligand."""
        site_properties = self._analyze_binding_site_properties(sequence, position)

        # Calculate physicochemical compatibility scores
        hydrophobic_match = 1 - abs(site_properties['hydrophobicity'] - ligand_features.get('logP', 0)) / 5
        size_match = 1 - abs(ligand_features.get('molecular_weight', 500) - 500) / 1000

        # Calculate hydrogen bonding potential
        h_bond_potential = min(
            ligand_features.get('hbd', 0) + ligand_features.get('hba', 0),
            site_properties['surface_accessibility'] * 10
        ) / 10

        # Calculate overall compatibility score
        compatibility = (
            0.4 * hydrophobic_match +
            0.3 * size_match +
            0.3 * h_bond_potential
        )

        return float(np.clip(compatibility, 0, 1))

    def _calculate_conservation_score(self, sequence, position):
        """Calculate evolutionary conservation score."""
        window_size = 5
        start = max(0, position - window_size)
        end = min(len(sequence), position + window_size + 1)
        site_sequence = sequence[start:end]

        # Calculate position-specific scoring matrix
        inputs = self.tokenizer(site_sequence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            attention = torch.mean(outputs.attentions[-1], dim=1).squeeze()

            # Calculate conservation based on attention patterns
            position_scores = attention[window_size]  # Center position
            conservation_score = float(torch.sigmoid(torch.mean(position_scores)))

            # Adjust score based on amino acid properties
            aa_properties = {
                'A': 0.8, 'G': 0.8, 'P': 0.8,  # Small
                'R': 0.7, 'K': 0.7, 'H': 0.7,  # Basic
                'D': 0.7, 'E': 0.7,            # Acidic
                'Q': 0.6, 'N': 0.6,            # Amide
                'S': 0.6, 'T': 0.6,            # Hydroxyl
                'C': 0.9, 'M': 0.7,            # Sulfur
                'V': 0.5, 'I': 0.5, 'L': 0.5,  # Hydrophobic
                'F': 0.4, 'Y': 0.4, 'W': 0.4   # Aromatic
            }

            aa_score = aa_properties.get(sequence[position], 0.5)
            return (conservation_score + aa_score) / 2

    def generate_protein_sequence(self, prompt, length=100):
        """Generate protein sequence based on input prompt."""
        try:
            # Generate sequence using text pipeline
            generated = self.text_pipeline(f"Generate protein sequence: {prompt}",
                                        max_length=length*4,  # Account for tokens
                                        num_return_sequences=1)[0]['generated_text']

            # Extract amino acid sequence from generated text
            sequence = ''.join(c for c in generated if c in "ACDEFGHIKLMNPQRSTVWY")
            sequence = sequence[:length]  # Trim to desired length

            # Analyze sequence quality
            analysis = self.analyze_sequence(sequence)
            quality_score = analysis['score']

            return {
                'start': 0,
                'end': len(sequence),
                'score': float(quality_score),
                'type': 'sequence_generation',
                'sequence': {
                    'start': 0,
                    'end': len(sequence),
                    'score': float(quality_score),
                    'type': 'generated_sequence',
                    'text': sequence
                },
                'prompt': {
                    'start': 0,
                    'end': len(prompt),
                    'score': 1.0,
                    'type': 'input_prompt',
                    'text': prompt
                },
                'analysis': analysis
            }
        except Exception as e:
            return {
                'start': 0,
                'end': 0,
                'score': 0.0,
                'type': 'sequence_generation_error',
                'error': str(e)
            }

    def optimize_sequence(self, sequence, optimization_target='stability'):
        """Optimize protein sequence for specific properties."""
        try:
            original_score = self.analyze_sequence(sequence)['score']
            best_sequence = sequence
            best_score = original_score
            best_position = 0

            # Analyze each position for potential improvements
            for i in range(len(sequence)):
                mutations = []
                for aa in "ACDEFGHIKLMNPQRSTVWY":
                    if aa != sequence[i]:
                        mutated = sequence[:i] + aa + sequence[i+1:]
                        score = self.analyze_sequence(mutated)['score']
                        if score > best_score:
                            best_sequence = mutated
                            best_score = score
                            best_position = i

            return {
                'start': best_position,
                'end': best_position + 1,
                'score': float(best_score),
                'type': 'sequence_optimization',
                'original_sequence': sequence,
                'optimized_sequence': best_sequence,
                'improvement': float(best_score - original_score)
            }
        except Exception as e:
            return {
                'start': 0,
                'end': len(sequence),
                'score': 0.0,
                'type': 'sequence_optimization_error',
                'error': str(e)
            }
