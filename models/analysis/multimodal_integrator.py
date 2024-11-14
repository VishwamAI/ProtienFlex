"""
MultiModal Protein Analyzer for ProteinFlex
Integrates sequence, structure, and function prediction into a unified system
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .enhanced_sequence_analyzer import EnhancedSequenceAnalyzer
from .structure_predictor import StructurePredictor
from .function_predictor import FunctionPredictor

class MultiModalProteinAnalyzer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Initialize component models
        self.sequence_analyzer = EnhancedSequenceAnalyzer(config)
        self.structure_predictor = StructurePredictor(config)
        self.function_predictor = FunctionPredictor(config)

        # Cross-modal attention for feature integration
        self.cross_modal_attention = CrossModalAttention(
            config.get('hidden_size', 320)  # Updated to match ESM2 dimensions
        )

        # Unified prediction head
        self.unified_predictor = UnifiedPredictor(
            config.get('hidden_size', 320)  # Updated to match ESM2 dimensions
        )

    def forward(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        # Analyze sequences
        sequence_results = self.sequence_analyzer(sequences)
        sequence_features = sequence_results['embeddings']  # Updated to match new key name

        # Predict structure
        structure_results = self.structure_predictor(sequence_features)
        structure_features = structure_results['refined_structure']

        # Integrate features using cross-modal attention
        integrated_features = self.cross_modal_attention(
            sequence_features,
            structure_features
        )

        # Predict function
        function_results = self.function_predictor(
            sequence_features,
            structure_features
        )

        # Generate unified predictions
        unified_results = self.unified_predictor(
            sequence_features,
            structure_features,
            function_results
        )

        return {
            'sequence_analysis': sequence_results,
            'structure_prediction': structure_results,
            'function_prediction': function_results,
            'unified_prediction': unified_results,
            'integrated_features': integrated_features
        }

    def analyze_protein(self, sequence: str) -> Dict[str, torch.Tensor]:
        """Comprehensive protein analysis combining all modalities"""
        return self.forward([sequence])

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.sequence_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.structure_attention = nn.MultiheadAttention(hidden_size, num_heads=8)

        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        structure_features: torch.Tensor
    ) -> torch.Tensor:
        # Cross attention between sequence and structure
        seq_attended, _ = self.sequence_attention(
            sequence_features,
            structure_features,
            structure_features
        )

        struct_attended, _ = self.structure_attention(
            structure_features,
            sequence_features,
            sequence_features
        )

        # Combine attended features
        combined = torch.cat([seq_attended, struct_attended], dim=-1)
        integrated = self.feature_combiner(combined)

        return integrated

class UnifiedPredictor(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.integration_network = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        structure_features: torch.Tensor,
        function_results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Combine all features
        combined_features = torch.cat([
            sequence_features,
            structure_features,
            function_results['go_terms']
        ], dim=-1)

        # Generate unified representation
        unified_features = self.integration_network(combined_features)

        # Estimate prediction confidence
        confidence = self.confidence_estimator(unified_features)

        return {
            'unified_features': unified_features,
            'confidence': confidence
        }

def create_multimodal_analyzer(config: Dict) -> MultiModalProteinAnalyzer:
    """Factory function to create MultiModalProteinAnalyzer instance"""
    return MultiModalProteinAnalyzer(config)
