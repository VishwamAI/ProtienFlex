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
        # Basic protein analysis
        protein_analysis = ProteinAnalysis(sequence)
        basic_properties = {
            'molecular_weight': protein_analysis.molecular_weight(),
            'aromaticity': protein_analysis.aromaticity(),
            'instability_index': protein_analysis.instability_index(),
            'isoelectric_point': protein_analysis.isoelectric_point()
        }

        # Tokenize sequence
        inputs = self.tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings with real-time adjustment
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state

            # Calculate attention scores with dynamic threshold
            attention_weights = torch.mean(outputs.attentions[-1], dim=1)
            residue_importance = torch.mean(attention_weights, dim=1).squeeze()

            if adjust_params:
                self._adjust_parameters(residue_importance)

        return {
            'embeddings': embeddings.cpu().numpy(),
            'residue_importance': residue_importance.cpu().numpy(),
            'basic_properties': basic_properties
        }

    def predict_mutations(self, sequence, positions):
        """Predict effects of mutations at specified positions."""
        results = []
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

                    mutations.append({
                        'mutation': f"{original_aa}{pos+1}{new_aa}",
                        'stability_change': stability_change,
                        'impact': impact,
                        'confidence': self._calculate_confidence(stability_change, impact)
                    })

            results.append({
                'position': pos,
                'mutations': sorted(mutations, key=lambda x: x['stability_change'], reverse=True)
            })

        return results

    def analyze_drug_binding(self, sequence, ligand_features=None):
        """Analyze potential drug binding sites and interactions."""
        sequence_features = self.analyze_sequence(sequence)

        # Identify potential binding pockets
        binding_sites = self.get_interaction_sites(sequence_features)

        # Enhanced binding site analysis
        detailed_sites = []
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

        return detailed_sites

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
        # Placeholder for surface accessibility calculation
        return 0.5

    def _calculate_druggability_score(self, sequence, position):
        """Calculate druggability score for binding site."""
        properties = self._analyze_binding_site_properties(sequence, position)
        return (properties['hydrophobicity'] + properties['surface_accessibility']) / 2

    def _predict_ligand_compatibility(self, sequence, position, ligand_features):
        """Predict compatibility between binding site and ligand."""
        site_properties = self._analyze_binding_site_properties(sequence, position)
        # Placeholder for ligand compatibility calculation
        return 0.5

    def _calculate_conservation_score(self, sequence, position):
        """Calculate evolutionary conservation score."""
        # Placeholder for conservation score calculation
        return 0.5
