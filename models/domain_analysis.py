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

logger = logging.getLogger(__name__)

class DomainAnalyzer:
    def __init__(self):
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
