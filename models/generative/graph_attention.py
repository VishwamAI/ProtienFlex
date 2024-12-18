"""
Graph Attention Layer for structure-aware protein generation
Implements findings from LaGDif and HelixProtX papers
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class GraphAttentionLayer(nn.Module):
    """
    Graph attention mechanism for protein structure awareness
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 8,
        dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Query, Key, Value transformations
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # Structure-aware components
        self.distance_embedding = nn.Linear(1, 1)
        self.angle_embedding = nn.Linear(1, 1)
        
        # Output
        self.output = nn.Linear(hidden_size, hidden_size)

        # Dropouts
        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(dropout_prob)

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose and reshape tensor for attention computation"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        distance_matrix: Optional[torch.Tensor] = None,
        angle_matrix: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with structure awareness

        Args:
            hidden_states: Input tensor [batch_size, seq_length, hidden_size]
            distance_matrix: Pairwise distances [batch_size, seq_length, seq_length]
            angle_matrix: Pairwise angles [batch_size, seq_length, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            output: Transformed hidden states
            attention_probs: Attention probabilities
        """
        # Linear transformations
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute base attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Add structure awareness if available
        if distance_matrix is not None:
            # Get distance embeddings [batch_size, seq_length, seq_length, 1]
            distance_embeddings = self.distance_embedding(distance_matrix.unsqueeze(-1))
            
            # Project embeddings to attention head size
            batch_size, seq_len_i, seq_len_j, embed_dim = distance_embeddings.size()
            
            # Reshape to match attention scores: [batch_size, num_heads, seq_len, seq_len]
            distance_scores = distance_embeddings.squeeze(-1)  # Remove last dimension
            distance_scores = distance_scores.unsqueeze(1)     # Add head dimension
            distance_scores = distance_scores.expand(-1, self.num_attention_heads, -1, -1)
            
            # Add to attention scores
            attention_scores = attention_scores + distance_scores

        if angle_matrix is not None:
            # Get angle embeddings [batch_size, seq_length, seq_length, 1]
            angle_embeddings = self.angle_embedding(angle_matrix.unsqueeze(-1))
            
            # Remove last dimension
            angle_scores = angle_embeddings.squeeze(-1)  # [batch_size, seq_len, seq_len]
            
            # Add head dimension and expand
            angle_scores = angle_scores.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            angle_scores = angle_scores.expand(-1, self.num_attention_heads, -1, -1)
            
            # Add to attention scores
            attention_scores = attention_scores + angle_scores

        # Before applying the attention mask, reshape it to match attention_scores dimensions
        if attention_mask is not None:
            # Reshape attention_mask from [batch_size, seq_length] to [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0

        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output transformation
        output = self.output(context_layer)
        output = self.output_dropout(output)
        output = self.layer_norm(output + hidden_states)


        return output, attention_probs
