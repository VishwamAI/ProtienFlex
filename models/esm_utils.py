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

    def predict_structure(self, sequence: str) -> Tuple[torch.Tensor, List[float]]:
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

            return token_representations, confidence_scores

        except Exception as e:
            logger.error(f"Error in structure prediction: {e}")
            raise

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

    def get_sequence_embeddings(self, sequence: str) -> torch.Tensor:
        """Get embeddings for a protein sequence."""
        try:
            batch_converter = self.alphabet.get_batch_converter()
            data = [("protein", sequence)]
            _, _, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33])
            return results["representations"][33]
        except Exception as e:
            logger.error(f"Error getting sequence embeddings: {e}")
            raise

    def get_attention_maps(self, sequence: str) -> torch.Tensor:
        """Get attention maps for a protein sequence."""
        try:
            batch_converter = self.alphabet.get_batch_converter()
            data = [("protein", sequence)]
            _, _, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
            return results["attentions"]
        except Exception as e:
            logger.error(f"Error getting attention maps: {e}")
            raise

    def analyze_sequence_windows(self, sequence: str, window_size: int) -> List[Dict]:
        """Analyze protein sequence using sliding windows."""
        try:
            embeddings = self.get_sequence_embeddings(sequence)
            windows = []
            for i in range(len(sequence) - window_size + 1):
                window_embeddings = embeddings[0, i:i+window_size]
                windows.append({
                    'start': i,
                    'end': i + window_size,
                    'sequence': sequence[i:i+window_size],
                    'score': float(torch.mean(window_embeddings).item())
                })
            return windows
        except Exception as e:
            logger.error(f"Error analyzing sequence windows: {e}")
            raise

    def compare_sequences(self, seq1: str, seq2: str) -> Dict:
        """Compare two protein sequences using their embeddings."""
        try:
            emb1 = self.get_sequence_embeddings(seq1)
            emb2 = self.get_sequence_embeddings(seq2)
            similarity = torch.nn.functional.cosine_similarity(
                emb1.mean(dim=1), emb2.mean(dim=1)
            ).item()
            return {
                'similarity_score': similarity,
                'seq1_length': len(seq1),
                'seq2_length': len(seq2)
            }
        except Exception as e:
            logger.error(f"Error comparing sequences: {e}")
            raise

    def calculate_confidence_scores(self, sequence: str) -> List[float]:
        """Calculate confidence scores for each position in the sequence."""
        try:
            embeddings = self.get_sequence_embeddings(sequence)
            return self._calculate_confidence(embeddings)
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {e}")
            raise
