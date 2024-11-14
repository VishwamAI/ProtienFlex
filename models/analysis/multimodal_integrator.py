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
            config.get('hidden_size', 768)  # Updated to match ESM2 dimensions
        )

        # Unified prediction head
        self.unified_predictor = UnifiedPredictor(
            config.get('hidden_size', 768)  # Updated to match ESM2 dimensions
        )

    def forward(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        # Analyze sequences
        sequence_results = self.sequence_analyzer(sequences)
        sequence_features = sequence_results['features']  # Using consistent key name

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
        self.hidden_size = hidden_size

        # Transform structure coordinates to feature space
        self.structure_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 768)  # Match ESM2 dimensions
        )

        self.sequence_attention = nn.MultiheadAttention(768, num_heads=8)
        self.structure_attention = nn.MultiheadAttention(768, num_heads=8)

        self.feature_combiner = nn.Sequential(
            nn.Linear(1536, 1024),  # 768 * 2 for concatenated features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768)  # Output matches ESM2 dimensions
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        structure_features: torch.Tensor
    ) -> torch.Tensor:
        # Transform structure coordinates to feature space
        batch_size, seq_len, _ = structure_features.shape
        structure_features = self.structure_encoder(structure_features)

        # Ensure correct shape for attention: [seq_len, batch_size, hidden_size]
        sequence_features = sequence_features.transpose(0, 1)
        structure_features = structure_features.transpose(0, 1)

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

        # Return to [batch_size, seq_len, hidden_size]
        seq_attended = seq_attended.transpose(0, 1)
        struct_attended = struct_attended.transpose(0, 1)

        # Combine attended features
        combined = torch.cat([seq_attended, struct_attended], dim=-1)
        integrated = self.feature_combiner(combined)

        return integrated

class UnifiedPredictor(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Transform function results to match feature dimensions
        self.function_encoder = nn.Sequential(
            nn.Linear(1000, 768),  # Transform GO terms to match feature dimensions
            nn.LayerNorm(768),     # Normalize features
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.integration_network = nn.Sequential(
            nn.Linear(2304, 1536),  # 3 * 768-dim features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1536, 768)  # Match ESM2 dimensions
        )

        self.confidence_estimator = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        structure_features: torch.Tensor,
        function_results: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Transform function features to match dimensions
        function_features = self.function_encoder(function_results['go_terms'])

        # Combine all features
        combined_features = torch.cat([
            sequence_features,
            structure_features,
            function_features
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
