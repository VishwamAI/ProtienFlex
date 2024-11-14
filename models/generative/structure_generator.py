"""
Structure-aware protein sequence generator
Implements findings from HelixProtX and LaGDif papers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .graph_attention import GraphAttentionLayer

class StructureAwareGenerator(nn.Module):
    """
    Structure-aware protein sequence generator with graph attention
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 8,
        num_layers: int = 6,
        dropout_prob: float = 0.1,
        max_position_embeddings: int = 1024
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers

        # Embeddings
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_embeddings = nn.Embedding(22, hidden_size)  # 20 AA + start/end
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        # Structure-aware attention layers
        self.attention_layers = nn.ModuleList([
            GraphAttentionLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                dropout_prob=dropout_prob,
                attention_probs_dropout_prob=dropout_prob
            )
            for _ in range(num_layers)
        ])

        # Feed-forward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout_prob)
            )
            for _ in range(num_layers)
        ])

        # Output layer
        self.output = nn.Linear(hidden_size, 22)  # 20 AA + start/end

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        distance_matrix: Optional[torch.Tensor] = None,
        angle_matrix: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with structure awareness

        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            position_ids: Position IDs [batch_size, seq_length]
            distance_matrix: Pairwise distances [batch_size, seq_length, seq_length]
            angle_matrix: Pairwise angles [batch_size, seq_length, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Dict containing:
                logits: Token logits
                hidden_states: Final hidden states
                attention_weights: Attention weights from all layers
        """
        batch_size, seq_length = input_ids.shape

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = token_embeddings + position_embeddings

        # Layer norm and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Store attention weights for visualization
        attention_weights = []

        # Process through layers
        for attention_layer, ff_layer in zip(self.attention_layers, self.ff_layers):
            # Structure-aware attention
            layer_output, attn_weights = attention_layer(
                hidden_states,
                distance_matrix=distance_matrix,
                angle_matrix=angle_matrix,
                attention_mask=attention_mask
            )
            attention_weights.append(attn_weights)

            # Feed-forward
            hidden_states = ff_layer(layer_output) + layer_output

        # Output logits
        logits = self.output(hidden_states)

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "attention_weights": attention_weights
        }

    def generate(
        self,
        start_tokens: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        distance_matrix: Optional[torch.Tensor] = None,
        angle_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate protein sequences with structure guidance

        Args:
            start_tokens: Initial tokens [batch_size, start_length]
            max_length: Maximum sequence length
            temperature: Sampling temperature
            distance_matrix: Optional distance constraints
            angle_matrix: Optional angle constraints

        Returns: Generated sequences [batch_size, max_length]
        """
        batch_size = start_tokens.shape[0]
        current_tokens = start_tokens

        for _ in range(max_length - start_tokens.shape[1]):
            # Prepare inputs
            position_ids = torch.arange(
                current_tokens.shape[1],
                device=current_tokens.device
            ).unsqueeze(0).expand(batch_size, -1)

            # Forward pass
            outputs = self.forward(
                current_tokens,
                position_ids=position_ids,
                distance_matrix=distance_matrix,
                angle_matrix=angle_matrix
            )

            # Get next token logits
            next_token_logits = outputs["logits"][:, -1, :] / temperature

            # Sample next tokens
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Concatenate with current tokens
            current_tokens = torch.cat([current_tokens, next_tokens], dim=1)

        return current_tokens
