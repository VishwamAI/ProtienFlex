from proteinflex.models.utils.lazy_imports import numpy, torch, openmm
from typing import Dict, Any, List, Optional

class MutationAnalyzer:
    """Analyzes the effects of mutations on protein sequences."""

    def __init__(self, model: Any, device: torch.device):
        """Initialize the mutation analyzer.

        Args:
            model: ESM model for protein analysis
            device: torch device (CPU/GPU)
        """
        self.model = model
        self.device = device
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    def predict_mutation_effect(self, sequence: str, position: int, mutation: str) -> Dict[str, Any]:
        """Predict the effect of a mutation on a protein sequence.

        Args:
            sequence: Original protein sequence
            position: Position to mutate (0-based)
            mutation: Amino acid to mutate to

        Returns:
            Dict containing stability impact, structural impact, conservation score,
            overall impact, and confidence scores
        """
        try:
            if not self._validate_inputs(sequence, position, mutation):
                return {"error": "Invalid inputs"}

            stability_impact = self._calculate_stability_impact(sequence, position, mutation)
            structural_impact = self._calculate_structural_impact(sequence, position)
            conservation_score = self._calculate_conservation(sequence, position)

            overall_impact = (stability_impact + structural_impact + conservation_score) / 3
            confidence = self._calculate_confidence(stability_impact, structural_impact)

            return {
                "stability_impact": stability_impact,
                "structural_impact": structural_impact,
                "conservation_score": conservation_score,
                "overall_impact": overall_impact,
                "confidence": confidence
            }
        except Exception as e:
            return {"error": str(e)}

    def _validate_inputs(self, sequence: str, position: int, mutation: str) -> bool:
        """Validate input parameters."""
        if not sequence or not isinstance(sequence, str):
            return False
        if position < 0 or position >= len(sequence):
            return False
        if mutation not in self.amino_acids:
            return False
        if not all(aa in self.amino_acids for aa in sequence):
            return False
        return True

    def _calculate_stability_impact(self, sequence: str, position: int, mutation: str) -> float:
        """Calculate the impact on protein stability."""
        with torch.no_grad():
            original_seq = torch.tensor([[self.model.alphabet.get_idx(aa) for aa in sequence]])
            mutated_seq = original_seq.clone()
            mutated_seq[0, position] = self.model.alphabet.get_idx(mutation)

            original_output = self.model(original_seq.to(self.device))
            mutated_output = self.model(mutated_seq.to(self.device))

            stability_score = torch.nn.functional.cosine_similarity(
                original_output['representations'][33].mean(1),
                mutated_output['representations'][33].mean(1)
            ).item()

            return 1 - (stability_score + 1) / 2  # Normalize to [0, 1]

    def _calculate_structural_impact(self, sequence: str, position: int) -> float:
        """Calculate the structural impact at the mutation position."""
        with torch.no_grad():
            seq_tensor = torch.tensor([[self.model.alphabet.get_idx(aa) for aa in sequence]])
            output = self.model(seq_tensor.to(self.device))

            attention_maps = output['attentions'][-1].mean(1)  # Use last layer
            position_importance = attention_maps[0, position].mean().item()

            return min(1.0, position_importance)

    def _calculate_conservation(self, sequence: str, position: int) -> float:
        """Calculate the conservation score at the mutation position."""
        with torch.no_grad():
            seq_tensor = torch.tensor([[self.model.alphabet.get_idx(aa) for aa in sequence]])
            output = self.model(seq_tensor.to(self.device))

            embeddings = output['representations'][33]
            position_embedding = embeddings[0, position]

            # Calculate similarity with other positions
            similarities = torch.nn.functional.cosine_similarity(
                position_embedding.unsqueeze(0),
                embeddings[0],
                dim=1
            )

            conservation_score = similarities.mean().item()
            return (conservation_score + 1) / 2  # Normalize to [0, 1]

    def _calculate_confidence(self, stability_impact: float, structural_impact: float) -> float:
        """Calculate confidence score based on stability and structural impacts."""
        # Higher impacts generally mean higher confidence in the prediction
        confidence = (stability_impact + structural_impact) / 2
        return confidence * 100  # Scale to percentage
