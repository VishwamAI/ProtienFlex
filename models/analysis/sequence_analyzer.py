from Bio.SeqUtils.ProtParam import ProteinAnalysis
from typing import List, Tuple
import re

class SequenceAnalyzer:
    def __init__(self, sequence: str):
        """Initialize analyzer with protein sequence."""
        self.sequence = sequence
        self.analysis = ProteinAnalysis(sequence)

    def find_calcium_binding_sites(self) -> List[Tuple[int, str]]:
        """Find DxDxDG calcium binding motifs."""
        sites = []
        pattern = re.compile(r'D[A-Z]D[A-Z]DG')
        for match in pattern.finditer(self.sequence):
            start = match.start()
            sites.append((start + 1, self.sequence[start:start + 6]))
        return sites

    def find_substrate_recognition_sites(self) -> List[Tuple[int, str]]:
        """Find aromatic substrate recognition sites."""
        sites = []
        patterns = [
            (r'W[A-Z]W', 'Type 1'),  # WxW motif
            (r'F[A-Z]Y', 'Type 2'),  # FxY motif
            (r'Y[A-Z]F', 'Type 3')   # YxF motif
        ]
        for pattern, site_type in patterns:
            for match in re.finditer(pattern, self.sequence):
                start = match.start()
                sites.append((start + 1, self.sequence[start:start + 3], site_type))
        return sites

    def analyze_beta_helix_structure(self) -> dict:
        """Analyze beta-helix core structure."""
        core_patterns = {
            'TVIGADNPG': 'Type A',
            'IVIGSDNPG': 'Type B'
        }
        results = {}
        for pattern, ptype in core_patterns.items():
            count = self.sequence.count(pattern)
            positions = []
            start = 0
            while True:
                pos = self.sequence.find(pattern, start)
                if pos == -1:
                    break
                positions.append(pos + 1)
                start = pos + 1
            results[ptype] = {
                'pattern': pattern,
                'count': count,
                'positions': positions
            }
        return results

    def analyze_sequence(self) -> None:
        """Perform comprehensive sequence analysis."""
        print("\nPectate Lyase B Sequence Analysis")
        print("=" * 50)

        # Basic properties
        print(f"\nSequence Length: {len(self.sequence)} residues")
        print(f"Molecular Weight: {self.analysis.molecular_weight():.1f} Da")
        helix, sheet, coil = self.analysis.secondary_structure_fraction()
        print(f"Secondary Structure Propensity:")
        print(f"  Alpha Helix: {helix:.3f}")
        print(f"  Beta Sheet:  {sheet:.3f}")
        print(f"  Random Coil: {coil:.3f}")

        # Calcium binding sites
        print("\nCalcium Binding Sites (DxDxDG motifs):")
        for pos, motif in self.find_calcium_binding_sites():
            print(f"  Position {pos}-{pos+5}: {motif}")

        # Substrate recognition sites
        print("\nSubstrate Recognition Sites:")
        for pos, motif, site_type in self.find_substrate_recognition_sites():
            print(f"  {site_type} at position {pos}-{pos+2}: {motif}")

        # Beta-helix structure
        print("\nBeta-helix Core Structure:")
        beta_helix = self.analyze_beta_helix_structure()
        for ptype, data in beta_helix.items():
            print(f"\n  {ptype} ({data['pattern']}):")
            print(f"    Count: {data['count']}")
            print(f"    Positions: {', '.join(map(str, data['positions']))}")

def main():
    # Generated sequence from text_to_protein_generator.py
    sequence = "TVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNDADNDGGPSGPGTSGPSGPGWNWIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPDQDEDGGPSGPGTSGPSGPGFKYTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPGTVIGADNPDADNDGGPSGPGTSGPSGPGYRFIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNPGIVIGSDNP"

    analyzer = SequenceAnalyzer(sequence)
    analyzer.analyze_sequence()

if __name__ == "__main__":
    main()
