from proteinflex.models.utils.lazy_imports import numpy, torch, openmm
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

"""Helper module for mutation analysis and prediction"""
from typing import List, Dict, Tuple
from Bio import PDB
from Bio.PDB import *
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
import logging
import esm
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)

class MutationAnalyzer:
    def __init__(self, esm_model, device):
        self.esm_model = esm_model
        self.device = device
        self.batch_converter = esm_model.alphabet.get_batch_converter()
        self.parser = PDB.PDBParser(QUIET=True)
        self.dssp = PDB.DSSP
        # Initialize memory management
        self.cache = {}
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self._setup_memory_hooks()
        # Detect model architecture and configure layers
        self.num_layers = len(self.esm_model.layers)
        self.repr_layer_num = self.num_layers - 1  # Use last layer for representations

    def _setup_memory_hooks(self):
        """Setup hooks for memory management"""
        def hook(module, input, output):
            if hasattr(output, 'detach'):
                return output.detach()
            return output
        for module in self.esm_model.modules():
            module.register_forward_hook(hook)

    def predict_mutation_effect(self, sequence: str, position: int, new_aa: str) -> Dict:
        """Predict the effect of a mutation"""
        try:
            # Validate inputs
            if not sequence or not isinstance(sequence, str):
                raise ValueError("Invalid sequence")
            if position < 0 or position >= len(sequence):
                raise ValueError(f"Position {position} is out of range for sequence length {len(sequence)}")
            if not new_aa or len(new_aa) != 1:
                raise ValueError("Invalid amino acid")
            # Calculate stability impact
            stability_score = self._calculate_stability_impact(sequence, position, new_aa)

            # Calculate structural impact
            structural_score = self._calculate_structural_impact(sequence, position)

            # Calculate evolutionary conservation
            conservation_score = self._calculate_conservation(sequence, position)

            # Validate scores
            if np.isnan(stability_score) or np.isnan(structural_score) or np.isnan(conservation_score):
                raise ValueError("Invalid score calculation")

            # Combined effect prediction
            impact_score = (stability_score + structural_score + conservation_score) / 3

            return {
                'stability_impact': stability_score,
                'structural_impact': structural_score,
                'conservation_score': conservation_score,
                'overall_impact': impact_score,
                'confidence': self._calculate_confidence(stability_score, structural_score, conservation_score)
            }
        except Exception as e:
            logger.error(f"Error in mutation effect prediction: {e}")
            return {'error': str(e)}

    def _calculate_stability_impact(self, sequence: str, position: int, new_aa: str) -> float:
        """Calculate stability impact of mutation using ESM model"""
        try:
            # Create mutated sequence
            mutated_sequence = sequence[:position] + new_aa + sequence[position + 1:]

            # Prepare sequences for ESM model
            data = [
                ("wild_type", sequence),
                ("mutant", mutated_sequence)
            ]

            # Move to appropriate device
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            # Get embeddings for both sequences
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[self.repr_layer_num], return_contacts=True)
                token_representations = results["representations"][self.repr_layer_num]

                # Extract embeddings for the mutation site and surrounding residues
                window_size = 5
                start_pos = max(0, position - window_size)
                end_pos = min(len(sequence), position + window_size + 1)

                wt_embeddings = token_representations[0, start_pos:end_pos]
                mut_embeddings = token_representations[1, start_pos:end_pos]

                # Calculate stability impact score based on embedding differences
                embedding_diff = torch.norm(wt_embeddings - mut_embeddings, dim=1)
                stability_impact = torch.mean(embedding_diff).item()

                # Analyze protein properties
                wt_analysis = ProteinAnalysis(sequence[start_pos:end_pos])
                mut_analysis = ProteinAnalysis(mutated_sequence[start_pos:end_pos])

                # Compare stability parameters
                wt_instability = wt_analysis.instability_index()
                mut_instability = mut_analysis.instability_index()
                instability_change = abs(mut_instability - wt_instability) / (abs(wt_instability) + 1e-10)

                # Combine embedding and property-based scores
                combined_score = (stability_impact + abs(instability_change)) / 2
                normalized_score = np.clip(combined_score / 2, 0, 1)

                return float(normalized_score)

        except Exception as e:
            logger.error(f"Error calculating stability impact: {e}")
            raise

    def _calculate_structural_impact(self, sequence: str, position: int) -> float:
        """Calculate structural impact of mutation using attention patterns"""
        try:
            # Prepare sequence for ESM model
            data = [("protein", sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            # Get attention maps
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[self.repr_layer_num], return_contacts=True)
                attention_maps = results["attentions"]

                # Analyze attention patterns around mutation site
                attention = attention_maps[0]  # First batch item
                position_attention = attention[:, :, position]  # All heads, all positions to mutation site

                # Calculate structural impact based on attention patterns with NaN prevention
                local_attention = position_attention[:, max(0, position-5):min(len(sequence), position+6)]
                attention_score = torch.mean(local_attention).item() if local_attention.numel() > 0 else 0.0

                # Normalize score to [0, 1] with NaN prevention
                normalized_score = np.clip(attention_score if not np.isnan(attention_score) else 0.0, 0, 1)

                return float(normalized_score)

        except Exception as e:
            logger.error(f"Error calculating structural impact: {e}")
            raise

    def _calculate_conservation(self, sequence: str, position: int) -> float:
        """Calculate evolutionary conservation score using ESM embeddings"""
        try:
            # Prepare sequence for ESM model
            data = [("protein", sequence)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            # Get embeddings and attention with memory optimization
            with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
                results = self.esm_model(batch_tokens, repr_layers=[self.repr_layer_num], return_contacts=True)

                # Extract needed tensors and immediately move to CPU if needed
                token_representations = results["representations"][self.repr_layer_num].detach()
                attention = results["attentions"][0].detach()

                # Calculate conservation score based on attention and embedding patterns
                position_embedding = token_representations[0, position].cpu()
                position_attention = attention[:, :, position].cpu()

                # Free up GPU memory
                del results, token_representations, attention
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Calculate embedding-based conservation score
                embedding_norm = torch.norm(position_embedding).item()
                attention_score = torch.mean(position_attention).item()

                # Combine scores and normalize
                conservation_score = (embedding_norm + attention_score) / 2
                normalized_score = np.clip(conservation_score, 0, 1)

                return float(normalized_score)

        except Exception as e:
            logger.error(f"Error calculating conservation score: {e}")
            raise

    def _calculate_confidence(self, stability: float, structural: float, conservation: float) -> float:
        """Calculate confidence score for prediction"""
        scores = [stability, structural, conservation]
        return min(100, (np.mean(scores) + np.std(scores)) * 100)
