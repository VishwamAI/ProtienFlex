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

"""
Protein Generator Module - Implements generative AI for protein sequence and structure prediction
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import os
import google.generativeai as genai
import asyncio
from .concept_bottleneck import ConceptBottleneckLayer, LoRALayer

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
        # Concept Bottleneck parameters
        num_concepts: int = 64,
        concept_groups: int = 4,
        # LoRA parameters
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.1,
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
        # Concept Bottleneck parameters
        self.num_concepts = num_concepts
        self.concept_groups = concept_groups
        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with structural awareness"""
    def __init__(self, config: ProteinGenerativeConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Standard attention components
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Structural attention components
        self.structure_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.structure_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.structure_value = nn.Linear(config.hidden_size, self.all_head_size)

        # Attention weights
        self.attention_weights = nn.Parameter(torch.ones(2) / 2)
        self.structure_dropout = nn.Dropout(config.attention_probs_dropout_prob)
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
        """Forward pass with structural attention"""
        batch_size, seq_length, _ = hidden_states.size()

        # Standard attention
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Structural attention
        struct_query = self.transpose_for_scores(self.structure_query(hidden_states))
        struct_key = self.transpose_for_scores(self.structure_key(hidden_states))
        struct_value = self.transpose_for_scores(self.structure_value(hidden_states))

        # Standard attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Structural attention scores
        struct_scores = torch.matmul(struct_query, struct_key.transpose(-1, -2))
        struct_scores = struct_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            if attention_mask.dim() > 4:
                attention_mask = attention_mask.squeeze(1).squeeze(1)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + attention_mask
            struct_scores = struct_scores + attention_mask

        # Normalize attention scores
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        struct_probs = nn.functional.softmax(struct_scores, dim=-1)

        # Apply dropouts
        attention_probs = self.dropout(attention_probs)
        struct_probs = self.structure_dropout(struct_probs)

        # Compute context layers
        context_layer = torch.matmul(attention_probs, value_layer)
        struct_context = torch.matmul(struct_probs, struct_value)

        # Combine attention mechanisms with learned weights
        weights = F.softmax(self.attention_weights, dim=0)
        combined_context = weights[0] * context_layer + weights[1] * struct_context

        # Reshape output
        combined_context = combined_context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = combined_context.size()[:-2] + (self.all_head_size,)
        combined_context = combined_context.view(*new_context_shape)

        outputs = (combined_context,)
        if output_attentions:
            outputs = outputs + ((attention_probs, struct_probs),)

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
    """Main protein generative model with structural awareness and concept bottleneck"""
    def __init__(self, config: ProteinGenerativeConfig):
        super().__init__(config)
        self.config = config
        self.register_buffer("_device_buffer", torch.zeros(1), persistent=False)

        # Initialize embeddings with proper padding
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )

        # Enhanced position embeddings with protein-specific features
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Protein-specific structural embeddings
        self.structural_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Amino acid property embeddings
        self.property_embeddings = nn.Embedding(
            20, config.hidden_size  # 20 standard amino acids
        )

        # Gradient checkpointing for memory efficiency
        self.gradient_checkpointing = True

        # Split layers into encoder and decoder with concept bottleneck
        num_encoder_layers = config.num_hidden_layers // 2
        num_decoder_layers = config.num_hidden_layers - num_encoder_layers

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            ProteinGenerativeLayer(config) for _ in range(num_encoder_layers)
        ])

        # Concept bottleneck layer
        self.concept_bottleneck = ConceptBottleneckLayer(
            hidden_size=config.hidden_size,
            num_concepts=config.num_concepts,
            concept_groups=config.concept_groups,
            dropout_prob=config.hidden_dropout_prob
        )

        # Decoder layers with LoRA optimization
        self.decoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'base': ProteinGenerativeLayer(config),
                'lora': LoRALayer(
                    hidden_size=config.hidden_size,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout
                )
            }) for _ in range(num_decoder_layers)
        ])

        # Enhanced normalization and regularization
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Structural validation layers
        self.structure_validator = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, 3)  # phi, psi, omega angles
        )

        # Initialize output projection for token prediction
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)

        # Add amino acid mappings with properties
        self.aa_properties = {
            'A': {'hydrophobic': 1, 'size': 0, 'charge': 0},
            'C': {'hydrophobic': 1, 'size': 0, 'charge': 0},
            'D': {'hydrophobic': 0, 'size': 0, 'charge': -1},
            'E': {'hydrophobic': 0, 'size': -1, 'charge': -1},
            'F': {'hydrophobic': 1, 'size': 2, 'charge': 0},
            'G': {'hydrophobic': 1, 'size': 0, 'charge': 0},
            'H': {'hydrophobic': 0, 'size': 1, 'charge': 1},
            'I': {'hydrophobic': 1, 'size': 1, 'charge': 0},
            'K': {'hydrophobic': 0, 'size': 1, 'charge': 1},
            'L': {'hydrophobic': 1, 'size': 1, 'charge': 0},
            'M': {'hydrophobic': 1, 'size': 1, 'charge': 0},
            'N': {'hydrophobic': 0, 'size': 0, 'charge': 0},
            'P': {'hydrophobic': 0, 'size': 0, 'charge': 0},
            'Q': {'hydrophobic': 0, 'size': 1, 'charge': 0},
            'R': {'hydrophobic': 0, 'size': 2, 'charge': 1},
            'S': {'hydrophobic': 0, 'size': 0, 'charge': 0},
            'T': {'hydrophobic': 0, 'size': 0, 'charge': 0},
            'V': {'hydrophobic': 1, 'size': 1, 'charge': 0},
            'W': {'hydrophobic': 1, 'size': 2, 'charge': 0},
            'Y': {'hydrophobic': 1, 'size': 2, 'charge': 0},
        }

        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.aa_properties.keys())}
        self.idx_to_aa = {v: k for k, v in self.aa_to_idx.items()}

        # Initialize weights
        self.init_weights()

        # Track concept interpretations
        self.concept_activations = {}

    @property
    def device(self) -> torch.device:
        return self._device_buffer.device

    def to(self, device: torch.device) -> 'ProteinGenerativeModel':
        return super().to(device)

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_concepts: Optional[bool] = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with structural awareness, concept bottleneck, and LoRA optimization"""
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

        # Get embeddings with protein-specific features
        inputs_embeds = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        structural_embeddings = self.structural_embeddings(position_ids)

        # Get amino acid properties
        aa_indices = torch.tensor([[self.aa_to_idx[self.idx_to_aa[id.item()]]
                                  for id in seq] for seq in input_ids], device=device)
        property_embeddings = self.property_embeddings(aa_indices)

        # Combine all embeddings with residual connections
        hidden_states = inputs_embeds + position_embeddings + structural_embeddings + property_embeddings
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Create causal attention mask
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=device) * -10000.0,
            diagonal=1
        )
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask * causal_mask.unsqueeze(0)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_structural_angles = []
        concept_outputs = {} if return_concepts else None

        # Encoder layers
        for i, layer in enumerate(self.encoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Apply gradient checkpointing if enabled
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # Structural validation after encoder layer
            structural_angles = self.structure_validator(hidden_states)
            all_structural_angles.append(structural_angles)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Apply concept bottleneck between encoder and decoder
        hidden_states, concept_acts = self.concept_bottleneck(
            hidden_states,
            return_concepts=return_concepts
        )
        if return_concepts:
            concept_outputs.update(concept_acts)

        # Decoder layers with LoRA optimization
        for i, layer_dict in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Apply base decoder layer
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_dict['base']),
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_dict['base'](
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            # Apply LoRA optimization
            lora_output = layer_dict['lora'](hidden_states)
            hidden_states = hidden_states + lora_output

            # Structural validation after decoder layer
            structural_angles = self.structure_validator(hidden_states)
            all_structural_angles.append(structural_angles)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Get logits
        logits = self.output_projection(hidden_states)

        # Prepare outputs
        outputs = {
            'last_hidden_state': hidden_states,
            'logits': logits,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
            'structural_angles': torch.stack(all_structural_angles, dim=1),
        }
        if return_concepts:
            outputs['concepts'] = concept_outputs

        return outputs

    def generate(
        self,
        prompt_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        template_sequence: Optional[str] = None,
        structural_guidance: bool = True,
        concept_guidance: bool = True,
        target_concepts: Optional[Dict[str, float]] = None,
        batch_size: int = 4,
    ) -> List[str]:
        """Generate protein sequences with concept guidance and structural validation.

        Args:
            prompt_text: Text description of desired protein
            max_length: Maximum sequence length
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeated tokens
            template_sequence: Optional template sequence for guided generation
            structural_guidance: Whether to use structural validation
            concept_guidance: Whether to use concept bottleneck guidance
            target_concepts: Optional target concept activations
            batch_size: Batch size for parallel generation
        """
        device = next(self.parameters()).device

        # Encode prompt text
        encoded = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]

        # Initialize template embeddings if provided
        template_embeddings = None
        if template_sequence:
            template_ids = self.tokenizer(template_sequence, return_tensors="pt").to(device)["input_ids"]
            template_embeddings = self.embeddings(template_ids)

        # Generate sequences in parallel batches
        all_sequences = []
        for batch_idx in range(0, num_return_sequences, batch_size):
            batch_size_actual = min(batch_size, num_return_sequences - batch_idx)

            # Expand inputs for batch generation
            batch_input_ids = input_ids.expand(batch_size_actual, -1)

            # Initialize sequence buffers
            current_length = batch_input_ids.shape[1]
            batch_sequences = batch_input_ids.clone()

            while current_length < max_length:
                # Get model predictions with concept interpretations
                outputs = self.forward(
                    input_ids=batch_sequences,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_concepts=concept_guidance
                )

                next_token_logits = outputs["logits"][:, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply repetition penalty
                for seq_idx in range(batch_size_actual):
                    for prev_token in batch_sequences[seq_idx]:
                        next_token_logits[seq_idx, prev_token] /= repetition_penalty

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply nucleus (top-p) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply structural guidance if enabled
                if structural_guidance and "structural_angles" in outputs:
                    structural_scores = self._evaluate_structural_validity(outputs["structural_angles"])
                    next_token_logits += structural_scores.unsqueeze(-1)

                # Apply concept guidance if enabled
                if concept_guidance and "concepts" in outputs:
                    concept_scores = self._evaluate_concept_alignment(
                        outputs["concepts"],
                        target_concepts
                    )
                    next_token_logits += concept_scores.unsqueeze(-1)

                # Apply template guidance if provided
                if template_embeddings is not None:
                    template_scores = self._compute_template_similarity(
                        outputs["last_hidden_state"],
                        template_embeddings,
                        current_length
                    )
                    next_token_logits += template_scores.unsqueeze(-1)

                # Sample next tokens
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

                # Append to sequences
                batch_sequences = torch.cat([batch_sequences, next_tokens], dim=1)
                current_length += 1

                # Check for end of sequence token
                if (batch_sequences == self.tokenizer.eos_token_id).any(dim=1).all():
                    break

            # Decode generated sequences
            for seq in batch_sequences:
                protein_sequence = self.tokenizer.decode(seq, skip_special_tokens=True)
                all_sequences.append(protein_sequence)

        return all_sequences[:num_return_sequences]

    def _evaluate_structural_validity(self, structural_angles: torch.Tensor) -> torch.Tensor:
        """Evaluate structural validity of generated angles"""
        # Ensure angles are on correct device
        structural_angles = structural_angles.to(self.device)

        # Convert angles to radians
        angles_rad = structural_angles * math.pi / 180.0

        # Check Ramachandran plot constraints
        phi = angles_rad[..., 0]
        psi = angles_rad[..., 1]
        omega = angles_rad[..., 2]

        # Score based on allowed regions in Ramachandran plot
        allowed_score = torch.where(
            (phi >= -math.pi/3) & (phi <= math.pi/3) &
            (psi >= -math.pi/3) & (psi <= math.pi/3),
            torch.ones_like(phi, device=self.device),
            torch.zeros_like(phi, device=self.device)
        )

        # Score planarity of peptide bond
        planarity_score = torch.cos(omega - math.pi)

        return (allowed_score + planarity_score) / 2.0

    def _evaluate_concept_alignment(
        self,
        current_concepts: Dict[str, torch.Tensor],
        target_concepts: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """Evaluate alignment between current and target concepts"""
        if target_concepts is None:
            return torch.zeros(
                current_concepts[list(current_concepts.keys())[0]].size(0),
                device=self.device
            )

        alignment_scores = []
        for concept_type, target_value in target_concepts.items():
            if concept_type in current_concepts:
                current_value = current_concepts[concept_type].to(self.device)
                target_tensor = torch.tensor(target_value, device=self.device)
                # Calculate similarity between current and target concept values
                diff = torch.abs(current_value - target_tensor)
                alignment_score = 1.0 - diff.mean(dim=-1)
                alignment_scores.append(alignment_score)

        if not alignment_scores:
            return torch.zeros(
                current_concepts[list(current_concepts.keys())[0]].size(0),
                device=self.device
            )

        return torch.stack(alignment_scores).mean(dim=0)

    def _compute_template_similarity(
        self,
        hidden_states: torch.Tensor,
        template_embeddings: torch.Tensor,
        current_length: int
    ) -> torch.Tensor:
        """Compute similarity between current hidden states and template"""
        template_length = template_embeddings.size(1)
        if current_length >= template_length:
            return torch.zeros(hidden_states.size(0), device=hidden_states.device)

        current_embeddings = hidden_states[:, -1, :]
        template_embedding = template_embeddings[:, current_length, :]

        similarity = torch.cosine_similarity(current_embeddings, template_embedding, dim=-1)
        return similarity
