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
        window_size = 15  # Increased window size for better domain detection
        stride = 5  # Step size for sliding window

        # Calculate embedding distances between adjacent windows
        for i in range(0, len(sequence) - window_size, stride):
            window_embeddings = embeddings[0, i:i+window_size]
            next_window = embeddings[0, min(i+window_size, len(sequence)-window_size):min(i+2*window_size, len(sequence))]

            # Calculate local structure features
            hydrophobicity = sum(self.hydrophobicity_scale.get(aa, 0) for aa in sequence[i:i+window_size]) / window_size
            embedding_variance = torch.var(window_embeddings, dim=0).mean().item()

            # Calculate sequence conservation in window
            conservation_score = self._calculate_conservation_scores(sequence[i:i+window_size])

            # Analyze embedding patterns for domain characteristics
            if len(next_window) > 0:
                embedding_diff = torch.norm(torch.mean(window_embeddings, dim=0) - torch.mean(next_window, dim=0)).item()
            else:
                embedding_diff = 0

            # Detect domain boundaries using multiple features
            if (embedding_variance > 0.3 and  # Moderate variance indicates domain
                abs(hydrophobicity) > 1.5 and # Significant hydrophobicity change
                conservation_score > 0.4):     # Reasonable conservation

                # Calculate confidence based on multiple features
                confidence = min(0.95, (
                    embedding_variance / 2.0 +
                    abs(hydrophobicity) / 4.0 +
                    conservation_score / 2.0 +
                    embedding_diff / 2.0
                ) / 2.0)

                # Determine domain type based on sequence features
                domain_type = "binding" if hydrophobicity < 0 and conservation_score > 0.6 else "catalytic"

                domains.append({
                    "start": i,
                    "end": i + window_size,
                    "score": float(confidence),
                    "type": domain_type,
                    "properties": {
                        "hydrophobicity": hydrophobicity,
                        "conservation": conservation_score,
                        "embedding_variance": embedding_variance
                    }
                })

        # Return standardized dictionary format
        return {
            "start": 0,
            "end": len(sequence),
            "score": max([d["score"] for d in domains]) if domains else 0.0,
            "type": "domain_analysis",
            "domains": domains
        }

    def analyze_domain_interactions(self, sequence):
        """Analyze interactions between domains."""
        if not sequence or len(sequence) < 2:
            raise ValueError("Invalid sequence for interaction analysis")

        domains = self.identify_domains(sequence)["domains"]  # Updated to access domains list from dictionary
        interactions = []

        # Get sequence embeddings for interaction analysis
        data = [(0, sequence)]
        batch_tokens = self.alphabet.batch_converter(data)[2]

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
        embeddings = results["representations"][33][0]
        attention_maps = results["attentions"][0]  # Use attention for interaction strength

        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                # Calculate embedding similarity between domains
                domain1_emb = embeddings[domain1["start"]:domain1["end"]].mean(dim=0)
                domain2_emb = embeddings[domain2["start"]:domain2["end"]].mean(dim=0)
                similarity = torch.nn.functional.cosine_similarity(
                    domain1_emb.unsqueeze(0),
                    domain2_emb.unsqueeze(0)
                ).item()

                # Calculate attention-based interaction strength
                domain1_attention = attention_maps[:, domain1["start"]:domain1["end"],
                                              domain2["start"]:domain2["end"]].mean()
                interaction_strength = (similarity + float(domain1_attention)) / 2

                # Determine interaction type based on embeddings and attention
                interaction_type = "strong_contact" if interaction_strength > 0.7 else "weak_contact"

                interactions.append({
                    "start": domain1["start"],
                    "end": domain2["end"],
                    "score": max(0.0, min(1.0, float(interaction_strength))),
                    "type": interaction_type,
                    "properties": {
                        "domain1_type": domain1["type"],
                        "domain2_type": domain2["type"],
                        "interaction_type": interaction_type,
                        "strength": max(0.0, min(1.0, float(interaction_strength)))
                    }
                })

        # Return standardized dictionary format
        return {
            "start": 0,
            "end": len(sequence),
            "score": max([i["score"] for i in interactions]) if interactions else 0.0,
            "type": "domain_interactions",
            "interactions": interactions
        }

    def predict_domain_function(self, sequence, domain_type):
        """Predict domain function based on sequence and type."""
        if not sequence:
            raise ValueError("Empty sequence provided")

        # Get sequence embeddings for function prediction
        data = [(0, sequence)]
        batch_tokens = self.alphabet.batch_converter(data)[2]

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
        embeddings = results["representations"][33][0]

        # Calculate sequence features
        conservation_scores = self._calculate_conservation_scores(sequence)
        active_sites = self._find_active_sites(sequence)

        # Calculate confidence based on multiple features
        feature_confidence = 0.0
        supporting_features = []

        if domain_type == "binding":
            feature_confidence = self._analyze_binding_features(sequence, embeddings, active_sites)
            if feature_confidence > 0.6:
                supporting_features.append("binding_site_pattern")
        elif domain_type == "catalytic":
            feature_confidence = self._analyze_catalytic_features(sequence, embeddings, active_sites)
            if feature_confidence > 0.6:
                supporting_features.append("catalytic_site_pattern")

        # Add structural and conservation evidence
        if torch.mean(embeddings).item() > 0.5:
            supporting_features.append("structural_motif")
            feature_confidence += 0.2
        if np.mean(conservation_scores) > 0.7:
            supporting_features.append("high_conservation")
            feature_confidence += 0.1

        # Normalize confidence score
        feature_confidence = max(0.0, min(1.0, feature_confidence))

        # Return standardized dictionary format
        return {
            "start": 0,
            "end": len(sequence),
            "score": feature_confidence,
            "type": "domain_function",
            "properties": {
                "domain_type": domain_type,
                "supporting_features": supporting_features,
                "active_sites": active_sites,
                "conservation": float(np.mean(conservation_scores))
            }
        }

    def calculate_domain_stability(self, sequence):
        """Calculate stability scores for domains."""
        if not sequence or len(sequence) < 5:
            raise ValueError("Invalid sequence for stability calculation")

        # Get sequence embeddings for stability calculation
        data = [(0, sequence)]
        batch_tokens = self.alphabet.batch_converter(data)[2]

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
        embeddings = results["representations"][33][0]

        # Calculate sequence features
        hydrophobicity_profile = self._calculate_hydrophobicity_profile(sequence)
        conservation_scores = self._calculate_conservation_scores(sequence)

        # Calculate stability metrics
        embedding_stability = torch.mean(torch.std(embeddings, dim=0)).item()
        hydrophobic_stability = np.mean(np.abs(hydrophobicity_profile))
        conservation_stability = np.mean(conservation_scores)

        # Calculate overall stability score
        stability_score = (
            (1.0 - embedding_stability) * 0.4 +  # Lower variance indicates stability
            hydrophobic_stability * 0.3 +        # Higher hydrophobicity contributes to stability
            conservation_stability * 0.3         # Higher conservation indicates stability
        )

        # Normalize stability score to [0, 1]
        stability_score = max(0.0, min(1.0, stability_score))

        # Analyze stability components
        stability_components = []
        window_size = 10
        for i in range(0, len(sequence) - window_size + 1):
            window_score = (
                (1.0 - float(torch.std(embeddings[i:i+window_size], dim=0).mean())) * 0.4 +
                float(np.mean(np.abs(hydrophobicity_profile[i:i+window_size]))) * 0.3 +
                float(np.mean(conservation_scores[i:i+window_size])) * 0.3
            )
            stability_components.append({
                "start": i,
                "end": i + window_size,
                "score": max(0.0, min(1.0, window_score)),
                "type": "stability_window",
                "properties": {
                    "embedding_stability": float(1.0 - torch.std(embeddings[i:i+window_size], dim=0).mean()),
                    "hydrophobic_stability": float(np.mean(np.abs(hydrophobicity_profile[i:i+window_size]))),
                    "conservation_stability": float(np.mean(conservation_scores[i:i+window_size]))
                }
            })

        # Return standardized dictionary format
        return {
            "start": 0,
            "end": len(sequence),
            "score": stability_score,
            "type": "domain_stability",
            "stability_components": stability_components,
            "properties": {
                "embedding_stability": float(1.0 - embedding_stability),
                "hydrophobic_stability": float(hydrophobic_stability),
                "conservation_stability": float(conservation_stability)
            }
        }

    def scan_domain_boundaries(self, sequence, window_size):
        """Scan for domain boundaries using sliding window."""
        if not sequence:
            raise ValueError("Empty sequence provided")

        # Get sequence embeddings
        data = [(0, sequence)]
        batch_tokens = self.alphabet.batch_converter(data)[2]

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
        embeddings = results["representations"][33][0]

        boundaries = []
        for i in range(len(sequence) - window_size):
            window_embeddings = embeddings[i:i+window_size]
            if self._is_domain_boundary(window_embeddings):
                # Calculate confidence based on embedding patterns
                embedding_grad = torch.gradient(window_embeddings.mean(dim=-1))[0]
                local_structure = torch.nn.functional.cosine_similarity(
                    window_embeddings[:-1], window_embeddings[1:], dim=-1
                )

                # Calculate boundary confidence using multiple features
                grad_score = float(torch.sigmoid(torch.abs(embedding_grad.mean())).item())
                struct_score = float(1 - local_structure.mean().item())
                hydrophobicity_change = abs(
                    sum(self.hydrophobicity_scale.get(sequence[i+j], 0) for j in range(window_size//2)) -
                    sum(self.hydrophobicity_scale.get(sequence[i+j], 0) for j in range(window_size//2, window_size))
                ) / window_size

                confidence = min(0.95, (grad_score + struct_score + hydrophobicity_change) / 3)

                # Determine boundary type based on sequence features
                boundary_type = "linker" if hydrophobicity_change > 0.5 else "domain"

                boundaries.append({
                    "start": i,
                    "end": i + window_size,
                    "score": confidence,
                    "type": boundary_type,
                    "position": i + window_size // 2,  # Center of the window
                    "confidence": confidence
                })

        # Return standardized dictionary format
        return {
            "start": 0,
            "end": len(sequence),
            "score": max([b["score"] for b in boundaries]) if boundaries else 0.0,
            "type": "domain_boundaries",
            "boundaries": boundaries
        }

    def _is_domain_boundary(self, embeddings):
        """Helper method to detect domain boundaries from embeddings."""
        # Simplified boundary detection based on embedding patterns
        # Calculate embedding gradient and local structure changes
        embedding_grad = torch.gradient(embeddings.mean(dim=-1))[0]
        local_structure = torch.nn.functional.cosine_similarity(
            embeddings[:-1], embeddings[1:], dim=-1
        )
        # Combine gradient and structure signals
        boundary_signal = (abs(embedding_grad) > 0.5) & (local_structure < 0.8)
        return boundary_signal.any().item()

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
            raw_domains = self._identify_domains(hydrophobicity_profile)

            # Find potential active sites
            active_sites = self._find_active_sites(sequence)

            # Calculate conservation scores (simplified)
            conservation = self._calculate_conservation_scores(sequence)

            # Process domains to include required fields
            domains = []
            for domain in raw_domains:
                confidence = min(0.95, (
                    abs(hydrophobicity_profile[domain['start']:domain['end']].mean()) / 5.0 +
                    np.mean(conservation[domain['start']:domain['end']]) +
                    0.5  # Base confidence
                ))
                domains.append({
                    'start': domain['start'],
                    'end': domain['end'],
                    'score': float(confidence),
                    'type': domain['type']
                })

            # Process active sites to include required fields
            processed_sites = []
            for site in active_sites:
                processed_sites.append({
                    'start': site['position'],
                    'end': site['position'] + site['length'],
                    'score': 0.9,  # High confidence for pattern matches
                    'type': site['type']
                })

            # Generate heatmap data
            heatmap_data = self._generate_heatmap_data(
                sequence,
                hydrophobicity_profile,
                domains,
                processed_sites,
                conservation
            )

            return {
                'start': 0,
                'end': len(sequence),
                'score': float(np.mean([d['score'] for d in domains])) if domains else 0.5,
                'type': 'protein_analysis',
                'domains': domains,
                'active_sites': processed_sites,
                'heatmap_data': heatmap_data,
                'annotations': self._generate_annotations(domains, processed_sites)
            }

        except Exception as e:
            logger.error(f"Error in domain analysis: {e}")
            return {
                'start': 0,
                'end': 0,
                'score': 0.0,
                'type': 'error'
            }

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

    def _analyze_binding_features(self, sequence, embeddings, active_sites):
        """Analyze features related to binding function"""
        # Check for binding-related patterns
        binding_confidence = 0.0

        # Check for binding site patterns
        if any(site['type'] == 'zinc_binding' for site in active_sites):
            binding_confidence += 0.4

        # Analyze embedding patterns characteristic of binding domains
        embedding_mean = torch.mean(embeddings, dim=0)
        if torch.norm(embedding_mean) > 0.5:  # Strong structural signal
            binding_confidence += 0.3

        # Check hydrophobicity patterns typical of binding sites
        hydrophobic_regions = sum(1 for aa in sequence if self.hydrophobicity_scale.get(aa, 0) > 2.0)
        if hydrophobic_regions / len(sequence) > 0.3:  # Significant hydrophobic regions
            binding_confidence += 0.3

        return min(binding_confidence, 0.95)

    def _analyze_catalytic_features(self, sequence, embeddings, active_sites):
        """Analyze features related to catalytic function"""
        # Check for catalytic-related patterns
        catalytic_confidence = 0.0

        # Check for catalytic site patterns
        if any(site['type'] == 'catalytic_triad' for site in active_sites):
            catalytic_confidence += 0.4

        # Analyze embedding patterns characteristic of catalytic domains
        embedding_variance = torch.var(embeddings, dim=0).mean().item()
        if embedding_variance > 0.3:  # Complex structural patterns
            catalytic_confidence += 0.3

        # Check for conserved catalytic residues
        catalytic_residues = {'H', 'D', 'S', 'E', 'K', 'C'}
        conserved_catalytic = sum(1 for aa in sequence if aa in catalytic_residues)
        if conserved_catalytic / len(sequence) > 0.15:  # Significant catalytic residues
            catalytic_confidence += 0.3

        return min(catalytic_confidence, 0.95)

    def _calculate_conservation_scores(self, sequence):
        """Calculate simplified conservation scores"""
        # This is a simplified version - in real applications,
        # you would use multiple sequence alignment
        try:
            # Convert sequence to tokens and get embeddings
            data = [(0, sequence)]
            batch_tokens = self.alphabet.batch_converter(data)[2]

            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[33])

            # Use attention scores as a proxy for conservation
            attn_weights = results["attentions"][0]  # First batch item
            conservation_scores = torch.mean(attn_weights, dim=(0, 1))  # Average across heads and positions

            # Normalize scores to [0,1]
            normalized_scores = (conservation_scores - conservation_scores.min()) / (conservation_scores.max() - conservation_scores.min())
            return normalized_scores.cpu().numpy()
        except Exception as e:
            logger.error(f"Error calculating conservation scores: {e}")
            return np.ones(len(sequence))  # Fallback to placeholder if error occurs

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
