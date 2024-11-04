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
import esm
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ESMModelHandler:
    def __init__(self):
        try:
            # Load ESM-2 model
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.model = self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Using CUDA for ESM model")
            else:
                logger.info("Using CPU for ESM model")
        except Exception as e:
            logger.error(f"Error initializing ESM model: {e}")
            raise

    def predict_structure(self, sequence: str) -> Dict:
        try:
            # Prepare data
            batch_converter = self.alphabet.get_batch_converter()
            data = [("protein", sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            # Move to GPU if available
            if torch.cuda.is_available():
                batch_tokens = batch_tokens.cuda()

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33])

            # Get per-residue representations
            token_representations = results["representations"][33]

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence(token_representations)

            # Find region with highest confidence
            window_size = 15
            max_conf_start = 0
            max_conf_score = 0
            for i in range(len(sequence) - window_size + 1):
                window_score = sum(confidence_scores[i:i+window_size]) / window_size
                if window_score > max_conf_score:
                    max_conf_score = window_score
                    max_conf_start = i

            return {
                'start': max_conf_start,
                'end': max_conf_start + window_size,
                'score': float(max_conf_score),
                'type': 'structure_prediction',
                'representations': token_representations.cpu().numpy(),
                'confidence_scores': confidence_scores
            }

        except Exception as e:
            logger.error(f"Error in structure prediction: {e}")
            return {
                'start': 0,
                'end': min(15, len(sequence)),
                'score': 0.0,
                'type': 'structure_prediction_error',
                'error': str(e)
            }

    def _calculate_confidence(self, representations: torch.Tensor) -> List[float]:
        """Calculate per-residue confidence scores based on representation entropy"""
        try:
            # Calculate entropy of representations as a confidence metric
            entropy = torch.mean(torch.var(representations, dim=-1), dim=0)
            confidence = 1.0 / (1.0 + entropy.cpu().numpy())
            return confidence.tolist()
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {e}")
            raise

    def get_sequence_embeddings(self, sequence: str) -> Dict:
        """Get embeddings for a protein sequence."""
        try:
            batch_converter = self.alphabet.get_batch_converter()
            data = [("protein", sequence)]
            _, _, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33])
            embeddings = results["representations"][33]

            # Calculate region with highest information content
            window_size = 15
            max_info_start = 0
            max_info_score = float('-inf')

            for i in range(len(sequence) - window_size + 1):
                window_embeddings = embeddings[0, i:i+window_size]
                info_score = float(torch.mean(torch.var(window_embeddings, dim=1)))
                if info_score > max_info_score:
                    max_info_score = info_score
                    max_info_start = i

            return {
                'start': max_info_start,
                'end': max_info_start + window_size,
                'score': float(max_info_score),
                'type': 'sequence_embeddings',
                'embeddings': embeddings.cpu().numpy()
            }
        except Exception as e:
            logger.error(f"Error getting sequence embeddings: {e}")
            return {
                'start': 0,
                'end': min(15, len(sequence)),
                'score': 0.0,
                'type': 'sequence_embeddings_error',
                'error': str(e)
            }

    def get_attention_maps(self, sequence: str) -> Dict:
        """Get attention maps for a protein sequence."""
        try:
            batch_converter = self.alphabet.get_batch_converter()
            data = [("protein", sequence)]
            _, _, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
            attention_maps = results["attentions"]

            # Find region with highest attention
            window_size = 15
            max_attention_start = 0
            max_attention_score = float('-inf')
            attention_matrix = attention_maps[0].mean(dim=0).cpu().numpy()

            for i in range(len(sequence) - window_size + 1):
                window_attention = attention_matrix[i:i+window_size, i:i+window_size]
                attention_score = float(window_attention.mean())
                if attention_score > max_attention_score:
                    max_attention_score = attention_score
                    max_attention_start = i

            return {
                'start': max_attention_start,
                'end': max_attention_start + window_size,
                'score': float(max_attention_score),
                'type': 'attention_analysis',
                'attention_maps': attention_maps[0].cpu().numpy(),
                'attention_matrix': attention_matrix
            }
        except Exception as e:
            logger.error(f"Error getting attention maps: {e}")
            return {
                'start': 0,
                'end': min(15, len(sequence)),
                'score': 0.0,
                'type': 'attention_analysis_error',
                'error': str(e)
            }

    def analyze_sequence_windows(self, sequence: str, window_size: int) -> Dict:
        """Analyze protein sequence using sliding windows."""
        try:
            embeddings = self.get_sequence_embeddings(sequence)['embeddings']
            windows = []
            max_score = float('-inf')
            best_start = 0

            for i in range(len(sequence) - window_size + 1):
                window_embeddings = embeddings[0, i:i+window_size]
                score = float(torch.mean(window_embeddings).item())
                windows.append({
                    'start': i,
                    'end': i + window_size,
                    'score': score,
                    'type': 'sequence_window',
                    'sequence': sequence[i:i+window_size],
                    'properties': {
                        'start': i,
                        'end': i + window_size,
                        'score': score,
                        'type': 'window_properties',
                        'embeddings': {
                            'start': i,
                            'end': i + window_size,
                            'score': score,
                            'type': 'window_embeddings'
                        }
                    }
                })
                if score > max_score:
                    max_score = score
                    best_start = i

            return {
                'start': best_start,
                'end': best_start + window_size,
                'score': float(max_score),
                'type': 'sequence_window_analysis',
                'windows': windows,
                'best_sequence': {
                    'start': best_start,
                    'end': best_start + window_size,
                    'score': float(max_score),
                    'type': 'best_window',
                    'sequence': sequence[best_start:best_start + window_size]
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing sequence windows: {e}")
            return {
                'start': 0,
                'end': min(window_size, len(sequence)),
                'score': 0.0,
                'type': 'sequence_window_analysis_error',
                'error': str(e)
            }

    def compare_sequences(self, seq1: str, seq2: str) -> Dict:
        """Compare two protein sequences using their embeddings."""
        try:
            emb1 = self.get_sequence_embeddings(seq1)['embeddings']
            emb2 = self.get_sequence_embeddings(seq2)['embeddings']

            # Find region with highest similarity
            window_size = min(15, min(len(seq1), len(seq2)))
            max_sim_start1 = 0
            max_sim_start2 = 0
            max_similarity = float('-inf')

            for i in range(len(seq1) - window_size + 1):
                for j in range(len(seq2) - window_size + 1):
                    sim = float(torch.nn.functional.cosine_similarity(
                        emb1[0, i:i+window_size].mean(dim=0, keepdim=True),
                        emb2[0, j:j+window_size].mean(dim=0, keepdim=True)
                    ).item())
                    if sim > max_similarity:
                        max_similarity = sim
                        max_sim_start1 = i
                        max_sim_start2 = j

            return {
                'start': max_sim_start1,
                'end': max_sim_start1 + window_size,
                'score': float(max_similarity),
                'type': 'sequence_comparison',
                'sequence2': {
                    'start': max_sim_start2,
                    'end': max_sim_start2 + window_size,
                    'score': float(max_similarity),
                    'type': 'sequence2_match'
                },
                'metadata': {
                    'start': 0,
                    'end': max(len(seq1), len(seq2)),
                    'score': float(max_similarity),
                    'type': 'comparison_metadata',
                    'seq1_length': len(seq1),
                    'seq2_length': len(seq2)
                }
            }
        except Exception as e:
            logger.error(f"Error comparing sequences: {e}")
            return {
                'start': 0,
                'end': min(15, len(seq1)),
                'score': 0.0,
                'type': 'sequence_comparison_error',
                'error': str(e)
            }

    def calculate_confidence_scores(self, sequence: str) -> Dict:
        """Calculate confidence scores for each position in the sequence."""
        try:
            embeddings = self.get_sequence_embeddings(sequence)['embeddings']
            confidence_scores = self._calculate_confidence(embeddings)

            # Find region with highest confidence
            window_size = 15
            max_conf_start = 0
            max_conf_score = float('-inf')

            for i in range(len(sequence) - window_size + 1):
                window_score = sum(confidence_scores[i:i+window_size]) / window_size
                if window_score > max_conf_score:
                    max_conf_score = window_score
                    max_conf_start = i

            # Convert confidence scores to properly structured dictionaries
            structured_scores = []
            for i, score in enumerate(confidence_scores):
                structured_scores.append({
                    'start': i,
                    'end': i + 1,
                    'score': float(score),
                    'type': 'position_confidence'
                })

            return {
                'start': max_conf_start,
                'end': max_conf_start + window_size,
                'score': float(max_conf_score),
                'type': 'confidence_analysis',
                'confidence_scores': {
                    'start': 0,
                    'end': len(sequence),
                    'score': float(sum(confidence_scores) / len(confidence_scores)),
                    'type': 'confidence_scores',
                    'scores': structured_scores
                },
                'sequence_info': {
                    'start': 0,
                    'end': len(sequence),
                    'score': float(max_conf_score),
                    'type': 'sequence_metadata',
                    'length': len(sequence)
                }
            }
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {e}")
            return {
                'start': 0,
                'end': min(15, len(sequence)),
                'score': 0.0,
                'type': 'confidence_analysis_error',
                'error': str(e)
            }
