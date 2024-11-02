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

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import *
import logging

logger = logging.getLogger(__name__)

class DomainAnalyzer:
    def __init__(self):
        """Initialize DomainAnalyzer with hydrophobicity scales and active site patterns"""
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

    def _calculate_hydrophobicity_profile(self, sequence, window_size=7):
        """Calculate hydrophobicity profile using sliding window"""
        self._validate_sequence(sequence)
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

        # Handle domain that extends to the end of the sequence
        if current_domain is not None:
            current_domain['end'] = len(hydrophobicity_profile)
            domains.append(current_domain)

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

    def predict_domain_interactions(self, sequence):
        """Predict interactions between protein domains"""
        domains = self._identify_domains(self._calculate_hydrophobicity_profile(sequence))
        n_domains = len(domains)
        interaction_matrix = np.zeros((n_domains, n_domains))
        domain_pairs = []

        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                score = self._calculate_interaction_score(domains[i], domains[j])
                interaction_matrix[i, j] = score
                interaction_matrix[j, i] = score
                if score > 0.5:  # Threshold for significant interactions
                    domain_pairs.append((i, j, score))

        return {
            'interaction_matrix': interaction_matrix,
            'domain_pairs': domain_pairs
        }

    def _calculate_interaction_score(self, domain1, domain2):
        """Calculate interaction score between two domains"""
        base_score = 0.5
        if domain1['type'] == domain2['type']:
            base_score += 0.2
        if abs(domain1['end'] - domain2['start']) < 10:
            base_score += 0.3
        return min(base_score, 1.0)

    def calculate_domain_stability(self, sequence):
        """Calculate stability scores for protein domains"""
        domains = self._identify_domains(self._calculate_hydrophobicity_profile(sequence))
        stability_scores = {}

        for i, domain in enumerate(domains):
            stability = self._calculate_domain_stability_score(sequence[domain['start']:domain['end']])
            stability_scores[f"domain_{i+1}"] = stability

        return {
            'stability_scores': stability_scores,
            'average_stability': np.mean(list(stability_scores.values()))
        }

    def _calculate_domain_stability_score(self, sequence):
        """Calculate stability score for a single domain"""
        try:
            analyzer = ProteinAnalysis(sequence)
            instability_index = analyzer.instability_index() / 100  # Normalize to 0-1
            hydrophobicity = np.mean([self.hydrophobicity_scale.get(aa, 0) for aa in sequence])
            stability = 1 - (0.7 * instability_index + 0.3 * abs(hydrophobicity) / 5)
            return max(0, min(1, stability))
        except Exception as e:
            logger.error(f"Error calculating stability score: {e}")
            return 0.5  # Default score

    def identify_binding_sites(self, sequence):
        """Identify potential binding sites in the protein sequence"""
        binding_sites = []
        domains = self._identify_domains(self._calculate_hydrophobicity_profile(sequence))
        
        for i, domain in enumerate(domains):
            domain_seq = sequence[domain['start']:domain['end']]
            for j in range(len(domain_seq) - 3):
                motif = domain_seq[j:j+4]
                if sum(1 for aa in motif if aa in 'AILMFWYV') >= 2 or sum(1 for aa in motif if aa in 'DEKR') >= 2:
                    binding_sites.append({
                        'domain_id': i,
                        'position': domain['start'] + j,
                        'motif': motif,
                        'score': max(0, min(1, (sum(self.hydrophobicity_scale.get(aa, 0) for aa in motif) + 4) / 8))
                    })
        return binding_sites

    def analyze_domain_flexibility(self, sequence):
        """Analyze flexibility of protein domains"""
        flexibility_scale = {'G':1.0, 'S':0.9, 'D':0.9, 'N':0.8, 'P':0.8, 'K':0.7, 'R':0.7, 'Q':0.7, 
                           'E':0.7, 'T':0.6, 'A':0.5, 'H':0.5, 'M':0.4, 'C':0.3, 'Y':0.3, 'V':0.2, 
                           'I':0.2, 'L':0.2, 'F':0.2, 'W':0.1}
        domains = self._identify_domains(self._calculate_hydrophobicity_profile(sequence))
        return {f"domain_{i+1}": {
            'score': sum(flexibility_scale.get(aa, 0.5) for aa in sequence[d['start']:d['end']]) / (d['end']-d['start']),
            'start': d['start'],
            'end': d['end']
        } for i, d in enumerate(domains)}

    def detect_domain_boundaries(self, sequence):
        """Detect precise domain boundaries using multiple features"""
        hydrophobicity = self._calculate_hydrophobicity_profile(sequence)
        domains = self._identify_domains(hydrophobicity)
        window = 5
        
        def refine_boundary(pos, direction):
            while 0 <= pos < len(sequence):
                if pos-window < 0 or pos+window >= len(sequence):
                    break
                left = sum(self.hydrophobicity_scale.get(aa, 0) for aa in sequence[pos-window:pos]) / window
                right = sum(self.hydrophobicity_scale.get(aa, 0) for aa in sequence[pos:pos+window]) / window
                if abs(left - right) > 1.0:
                    return pos
                pos += direction
            return pos

        return [{
            'start': refine_boundary(d['start'], -1),
            'end': refine_boundary(d['end'], 1),
            'type': d['type'],
            'confidence': 0.8  # Simplified confidence score
        } for d in domains]

    def analyze_domain_conservation(self, sequences):
        """Analyze conservation patterns across multiple sequences"""
        if not sequences or not all(len(seq) == len(sequences[0]) for seq in sequences):
            raise ValueError("All sequences must have the same length")

        # Calculate position-specific amino acid frequencies
        seq_length = len(sequences[0])
        position_frequencies = [{} for _ in range(seq_length)]
        for pos in range(seq_length):
            for seq in sequences:
                aa = seq[pos]
                if not aa.isalpha():
                    raise ValueError(f"Invalid amino acid character found at position {pos}")
                position_frequencies[pos][aa] = position_frequencies[pos].get(aa, 0) + 1

        # Calculate conservation scores
        conservation_scores = []
        for pos_freq in position_frequencies:
            total = sum(pos_freq.values())
            entropy = -sum((count/total) * np.log2(count/total) for count in pos_freq.values())
            conservation_scores.append(1 - (entropy / 4.32))  # Normalize by max possible entropy

        # Analyze conservation by domain
        domains = self._identify_domains(self._calculate_hydrophobicity_profile(sequences[0]))
        domain_conservation = {}
        for i, domain in enumerate(domains):
            domain_scores = conservation_scores[domain['start']:domain['end']]
            domain_conservation[f"domain_{i+1}"] = {
                'mean_conservation': np.mean(domain_scores),
                'peak_conservation': max(domain_scores),
                'conserved_regions': self._identify_conserved_regions(domain_scores, domain['start'])
            }

        return {
            'overall_conservation': np.mean(conservation_scores),
            'position_scores': conservation_scores,
            'domain_conservation': domain_conservation
        }

    def _identify_conserved_regions(self, scores, offset, threshold=0.8):
        """Identify highly conserved regions within a domain"""
        regions = []
        start = None
        for i, score in enumerate(scores):
            if score >= threshold and start is None:
                start = i + offset
            elif score < threshold and start is not None:
                regions.append({'start': start, 'end': i + offset, 'score': np.mean(scores[start-offset:i])})
                start = None
        if start is not None:
            regions.append({'start': start, 'end': len(scores) + offset, 'score': np.mean(scores[start-offset:])})
        return regions

    def generate_interaction_network(self, sequence):
        """Generate a network representation of domain interactions"""
        # Validate sequence
        if not sequence or not all(aa.isalpha() for aa in sequence):
            raise ValueError("Invalid sequence: must contain only valid amino acid characters")

        # Get domain interactions
        interaction_data = self.predict_domain_interactions(sequence)
        domain_pairs = interaction_data['domain_pairs']
        interaction_matrix = interaction_data['interaction_matrix']

        # Get domain properties
        domains = self._identify_domains(self._calculate_hydrophobicity_profile(sequence))
        flexibility = self.analyze_domain_flexibility(sequence)
        binding_sites = self.identify_binding_sites(sequence)

        # Build network representation
        network = {
            'nodes': [{
                'id': i,
                'type': domain['type'],
                'start': domain['start'],
                'end': domain['end'],
                'flexibility': flexibility[f"domain_{i+1}"]['score'],
                'binding_sites': [site for site in binding_sites if site['domain_id'] == i]
            } for i, domain in enumerate(domains)],
            'edges': [{
                'source': i,
                'target': j,
                'weight': score
            } for i, j, score in domain_pairs],
            'matrix': interaction_matrix.tolist()
        }

        return network

    def _validate_sequence(self, sequence):
        """Validate protein sequence"""
        if not sequence:
            raise ValueError("Empty sequence provided")
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_chars = set(sequence) - valid_aas
        if invalid_chars:
            raise ValueError(f"Invalid amino acid characters found: {','.join(invalid_chars)}")
        return True

    def analyze_domains(self, sequence):
        """Analyze protein domains and their properties"""
        try:
            self._validate_sequence(sequence)
            hydrophobicity_profile = self._calculate_hydrophobicity_profile(sequence)

            # Identify domains based on hydrophobicity patterns
            domains = self._identify_domains(hydrophobicity_profile)

            # Find potential active sites
            active_sites = self._find_active_sites(sequence)

            # Calculate conservation scores (simplified)
            conservation = self._calculate_conservation_scores(sequence)

            # Generate heatmap data
            heatmap = self._generate_heatmap_data(
                sequence,
                hydrophobicity_profile,
                domains,
                active_sites,
                conservation
            )

            return {
                'domains': domains,
                'active_sites': active_sites,
                'conservation': conservation.tolist(),
                'heatmap': heatmap,
                'annotations': self._generate_annotations(domains, active_sites)
            }
        except Exception as e:
            logger.error(f"Error in analyze_domains: {e}")
            return None

    def _calculate_hydrophobicity_profile(self, sequence, window_size=7):
        """Calculate hydrophobicity profile"""
        self._validate_sequence(sequence)
        profile = []
        half_window = window_size // 2

        # Pad sequence
        padded_seq = 'X' * half_window + sequence + 'X' * half_window

        for i in range(half_window, len(padded_seq) - half_window):
            window = padded_seq[i-half_window:i+half_window+1]
            avg_hydrophobicity = sum(self.hydrophobicity_scale.get(aa, 0) for aa in window) / window_size
            profile.append(avg_hydrophobicity)

        return np.array(profile)