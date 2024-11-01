"""
Protein Generator Module - Implements generative AI for protein sequence and structure prediction
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class ProteinGenerativeConfig(PretrainedConfig):
    """Configuration class for protein generation model"""
    model_type = "protein_generator"

    def __init__(
        self,
        vocab_size: int = 30,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            layer_norm_eps=layer_norm_eps,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, config: ProteinGenerativeConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass"""
        batch_size, seq_length, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Ensure attention_mask has correct shape [batch_size, 1, seq_length, seq_length]
            if attention_mask.dim() > 4:
                attention_mask = attention_mask.squeeze(1).squeeze(1)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,)
        if output_attentions:
            outputs = outputs + (attention_probs,)

        return outputs

class ProteinGenerativeLayer(nn.Module):
    """Transformer layer for protein generation"""
    def __init__(self, config: ProteinGenerativeConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass"""
        # Self-attention
        attention_output = self.attention(
            self.layernorm1(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # Add & Norm
        hidden_states = hidden_states + self.dropout(attention_output[0])
        hidden_states = self.layernorm1(hidden_states)

        # Feed-forward
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layernorm2(hidden_states + ff_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attention_output[1],)

        return outputs

class ProteinGenerativeModel(PreTrainedModel):
    """Main protein generative model"""
    def __init__(self, config: ProteinGenerativeConfig):
        super().__init__(config)
        self.config = config

        # Initialize embeddings with proper padding
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.layers = nn.ModuleList(
            [ProteinGenerativeLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize output projection for token prediction
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights
        self.init_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings = value

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        """Prunes heads of the model"""
        for layer, heads in heads_to_prune.items():
            self.layers[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        batch_size, seq_length = input_ids.size()
        device = input_ids.device

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Ensure input_ids are within vocab range
        input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)

        # Get embeddings
        inputs_embeds = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Add position embeddings
        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Create causal attention mask [batch_size, seq_length, seq_length]
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=device) * -10000.0,
            diagonal=1
        )
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask * causal_mask.unsqueeze(0)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Get logits
        logits = self.output_projection(hidden_states)

        return {
            'last_hidden_state': hidden_states,
            'logits': logits,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.95,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate protein sequences"""
        batch_size = input_ids.size(0)
        current_length = input_ids.size(1)
        device = input_ids.device

        # Ensure input_ids are within vocab range
        input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)

        # Initialize sequence storage
        generated = input_ids

        # Initialize or extend attention mask
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, current_length), device=device)

        # Generate tokens up to max_length
        for _ in range(current_length, max_length):
            # Get model predictions
            outputs = self.forward(
                input_ids=generated,
                attention_mask=attention_mask,
            )
            next_token_logits = outputs['logits'][:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                # Limit top_k to vocabulary size
                top_k = min(top_k, self.config.vocab_size)
                # Get top k logits and indices
                top_k_logits, _ = torch.topk(next_token_logits, top_k)
                # Create mask for values below threshold
                indices_to_remove = next_token_logits < top_k_logits[..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                # Sort logits and get probabilities
                sorted_logits, _ = torch.sort(next_token_logits, dim=-1, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p

                # Set logits below threshold to -inf
                next_token_logits_sorted = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))
                # Get indices to sort back to original order
                _, orig_indices = torch.sort(torch.sort(next_token_logits, dim=-1, descending=True)[1], dim=-1)
                # Restore original order
                next_token_logits = torch.gather(next_token_logits_sorted, -1, orig_indices)

            # Sample next token
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append next token to sequence
            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device)
            ], dim=1)

        return generated
