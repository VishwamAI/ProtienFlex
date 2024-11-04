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

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import *
import logging
import torch
import esm

logger = logging.getLogger(__name__)

class DomainAnalyzer:
    def __init__(self, model=None, alphabet=None):
        # Initialize ESM model
        try:
            if model is None or alphabet is None:
                self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            else:
                self.model = model
                self.alphabet = alphabet
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            logger.error(f"Error initializing ESM model: {e}")
            raise RuntimeError(f"Failed to initialize ESM model: {e}")

        self.hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        self.active_site_patterns = {
            'catalytic_triad': ['HDS', 'SHD', 'SDH'],  # Common in enzymes
            'zinc_binding': ['HEXXH', 'HXXEH'],        # Zinc finger motifs
            'nuclear_localization': ['PKKKRKV'],       # Nuclear localization signal
            'glycosylation': ['NXS', 'NXT']           # N-glycosylation sites
        }

    def identify_domains(self, sequence):
        """Identify protein domains using ESM embeddings."""
        if not sequence:
            raise ValueError("Empty sequence provided")

        # Convert sequence to tokens
        data = [(0, sequence)]
        batch_tokens = self.alphabet.batch_converter(data)[2]

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])

        embeddings = results["representations"][33]

        # Process embeddings to identify domains
        domains = []
        # Use sliding window to detect domain boundaries
        window_size = 5
        for i in range(len(sequence) - window_size):
            if self._is_domain_boundary(embeddings[0, i:i+window_size]):
                domains.append({
                    "start": i,
                    "end": i + window_size,
                    "confidence": 0.8,  # Placeholder confidence score
                    "type": "domain"
                })

        return domains

    def analyze_domain_interactions(self, sequence):
        """Analyze interactions between domains."""
        if not sequence or len(sequence) < 2:
            raise ValueError("Invalid sequence for interaction analysis")

        domains = self.identify_domains(sequence)
        interactions = []

        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                interactions.append({
                    "domain1": domain1,
                    "domain2": domain2,
                    "interaction_type": "contact",
                    "strength": 0.7  # Placeholder interaction strength
                })

        return interactions

    def predict_domain_function(self, sequence, domain_type):
        """Predict domain function based on sequence and type."""
        if not sequence:
            raise ValueError("Empty sequence provided")

        return {
            "function": f"predicted_{domain_type}_function",
            "confidence": 0.8,
            "supporting_features": ["sequence_motif", "structure_prediction"]
        }

    def calculate_domain_stability(self, sequence):
        """Calculate stability scores for domains."""
        if not sequence or len(sequence) < 5:
            raise ValueError("Invalid sequence for stability calculation")

        domains = self.identify_domains(sequence)
        stability_scores = {}

        for i, domain in enumerate(domains):
            stability_scores[f"domain_{i}"] = {
                "stability_score": 0.75,  # Placeholder stability score
                "confidence": 0.8
            }

        return stability_scores

    def scan_domain_boundaries(self, sequence, window_size):
        """Scan for domain boundaries using sliding window."""
        if not sequence:
            raise ValueError("Empty sequence provided")

        boundaries = []
        for i in range(len(sequence) - window_size):
            if self._is_potential_boundary(sequence[i:i+window_size]):
                boundaries.append({
                    "position": i,
                    "confidence": 0.8,
                    "type": "boundary"
                })

        return boundaries

    def _is_domain_boundary(self, embeddings):
        """Helper method to detect domain boundaries from embeddings."""
        # Simplified boundary detection based on embedding patterns
        return torch.std(embeddings).item() > 0.5

    def _is_potential_boundary(self, sequence_window):
        """Helper method to identify potential domain boundaries."""
        # Simplified boundary detection based on sequence properties
        return True if len(set(sequence_window)) > 3 else False

    def analyze_domains(self, sequence):
        """Analyze protein sequence for domains and functional sites"""
        try:
            # Calculate hydrophobicity profile
            hydrophobicity_profile = self._calculate_hydrophobicity_profile(sequence)

            # Identify domains based on hydrophobicity patterns
            domains = self._identify_domains(hydrophobicity_profile)

            # Find potential active sites
            active_sites = self._find_active_sites(sequence)

            # Calculate conservation scores (simplified)
            conservation = self._calculate_conservation_scores(sequence)

            # Generate heatmap data
            heatmap_data = self._generate_heatmap_data(
                sequence,
                hydrophobicity_profile,
                domains,
                active_sites,
                conservation
            )

            return {
                'domains': domains,
                'active_sites': active_sites,
                'heatmap_data': heatmap_data,
                'annotations': self._generate_annotations(domains, active_sites)
            }

        except Exception as e:
            logger.error(f"Error in domain analysis: {e}")
            return None

    def _calculate_hydrophobicity_profile(self, sequence, window_size=7):
        """Calculate hydrophobicity profile using sliding window"""
        profile = []
        half_window = window_size // 2

        # Pad sequence
        padded_seq = 'X' * half_window + sequence + 'X' * half_window

        for i in range(half_window, len(padded_seq) - half_window):
            window = padded_seq[i-half_window:i+half_window+1]
            avg_hydrophobicity = np.mean([
                self.hydrophobicity_scale.get(aa, 0)
                for aa in window if aa in self.hydrophobicity_scale
            ])
            profile.append(avg_hydrophobicity)

        return np.array(profile)

    def _identify_domains(self, hydrophobicity_profile, threshold=1.5):
        """Identify domains based on hydrophobicity patterns"""
        domains = []
        current_domain = None

        for i, value in enumerate(hydrophobicity_profile):
            if value > threshold and current_domain is None:
                current_domain = {'start': i, 'type': 'hydrophobic'}
            elif value < -threshold and current_domain is None:
                current_domain = {'start': i, 'type': 'hydrophilic'}
            elif abs(value) < threshold and current_domain is not None:
                current_domain['end'] = i
                domains.append(current_domain)
                current_domain = None

        return domains

    def _find_active_sites(self, sequence):
        """Find potential active sites based on sequence patterns"""
        active_sites = []

        for site_type, patterns in self.active_site_patterns.items():
            for pattern in patterns:
                for i in range(len(sequence) - len(pattern) + 1):
                    window = sequence[i:i+len(pattern)]
                    if self._match_pattern(window, pattern):
                        active_sites.append({
                            'type': site_type,
                            'position': i,
                            'length': len(pattern)
                        })

        return active_sites

    def _match_pattern(self, sequence, pattern):
        """Match sequence against pattern with wildcards"""
        if len(sequence) != len(pattern):
            return False

        for s, p in zip(sequence, pattern):
            if p == 'X':
                continue
            if s != p:
                return False
        return True

    def _calculate_conservation_scores(self, sequence):
        """Calculate simplified conservation scores"""
        # This is a simplified version - in real applications,
        # you would use multiple sequence alignment
        return np.ones(len(sequence))  # Placeholder

    def _generate_heatmap_data(self, sequence, hydrophobicity, domains, active_sites, conservation):
        """Generate heatmap data for visualization"""
        length = len(sequence)
        heatmap = np.zeros((4, length))  # 4 tracks: hydrophobicity, domains, active sites, conservation

        # Normalize hydrophobicity to [0,1]
        heatmap[0] = (hydrophobicity - np.min(hydrophobicity)) / (np.max(hydrophobicity) - np.min(hydrophobicity))

        # Mark domains
        for domain in domains:
            heatmap[1][domain['start']:domain['end']] = 1 if domain['type'] == 'hydrophobic' else 0.5

        # Mark active sites
        for site in active_sites:
            heatmap[2][site['position']:site['position']+site['length']] = 1

        # Conservation scores
        heatmap[3] = conservation

        return heatmap.tolist()

    def _generate_annotations(self, domains, active_sites):
        """Generate annotation objects for visualization"""
        annotations = []

        # Domain annotations
        for i, domain in enumerate(domains):
            annotations.append({
                'type': 'domain',
                'start': domain['start'],
                'end': domain['end'],
                'label': f"{domain['type'].capitalize()} Domain {i+1}",
                'color': '#ff9800' if domain['type'] == 'hydrophobic' else '#2196f3'
            })

        # Active site annotations
        for site in active_sites:
            annotations.append({
                'type': 'active_site',
                'start': site['position'],
                'end': site['position'] + site['length'],
                'label': f"{site['type'].replace('_', ' ').title()}",
                'color': '#f44336'
            })
        return annotations
