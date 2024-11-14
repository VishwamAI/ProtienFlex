"""
Concept Bottleneck Layer for interpretable protein generation
Implements the CB-pLM architecture for enhanced protein design control
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class ConceptBottleneckLayer(nn.Module):
    """
    Implements Concept Bottleneck Layer for interpretable protein generation.
    Maps hidden states to interpretable protein concepts before generation.
    """
    def __init__(
        self,
        hidden_size: int,
        num_concepts: int = 64,
        concept_groups: int = 4,
        dropout_prob: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_concepts = num_concepts
        self.concept_groups = concept_groups

        # Concept mapping layers
        self.concept_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, num_concepts // concept_groups),
                nn.LayerNorm(num_concepts // concept_groups)
            ) for _ in range(concept_groups)
        ])

        # Concept interpretation layers
        self.concept_interpreters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_concepts // concept_groups, hidden_size // concept_groups),
                nn.GELU(),
                nn.Dropout(dropout_prob)
            ) for _ in range(concept_groups)
        ])

        # Concept groups represent different protein properties
        self.concept_groups_map = {
            0: "structure",      # Secondary/tertiary structure elements
            1: "chemistry",      # Chemical properties (hydrophobicity, charge)
            2: "function",       # Functional domains and motifs
            3: "interaction"     # Protein-protein interaction sites
        }

        # Final projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_concepts: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through concept bottleneck layer

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            return_concepts: Whether to return concept activations

        Returns:
            transformed_states: Transformed hidden states
            concepts: Optional dictionary of concept activations by group
        """
        batch_size, seq_length, _ = hidden_states.size()
        concept_outputs = []
        concept_activations = {}

        # Process each concept group
        for i, (transform, interpreter) in enumerate(zip(
            self.concept_transform, self.concept_interpreters)):

            # Map to concept space
            concept_logits = transform(hidden_states)
            concept_probs = torch.sigmoid(concept_logits)

            # Store activations if requested
            if return_concepts:
                concept_activations[self.concept_groups_map[i]] = concept_probs

            # Map back to hidden space
            concept_features = interpreter(concept_probs)
            concept_outputs.append(concept_features)

        # Combine concept group outputs
        combined_concepts = torch.cat(concept_outputs, dim=-1)

        # Final transformation
        transformed_states = self.output_projection(combined_concepts)
        transformed_states = self.layer_norm(transformed_states + hidden_states)

        if return_concepts:
            return transformed_states, concept_activations
        return transformed_states, None

class LoRALayer(nn.Module):
    """
    Implements Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
    """
    def __init__(
        self,
        hidden_size: int,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.lora_alpha = lora_alpha
        # Increase base scaling for more noticeable effect while maintaining stability
        self.scaling = lora_alpha / (lora_rank * 0.5)  # Adjusted scaling factor

        # LoRA components
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_down = nn.Linear(hidden_size, lora_rank, bias=False)
        self.lora_up = nn.Linear(lora_rank, hidden_size, bias=False)

        # Initialize weights
        nn.init.normal_(self.lora_down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_up.weight, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation with residual connection"""
        dropped_hidden = self.lora_dropout(hidden_states)
        down_hidden = self.lora_down(dropped_hidden)
        up_hidden = self.lora_up(down_hidden)
        return hidden_states + (up_hidden * self.scaling)
