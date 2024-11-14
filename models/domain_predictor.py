import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdb

class DomainPredictor(nn.Module):
    """Advanced protein domain prediction using transformer architecture."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 6,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        # Embedding layers
        self.position_embeddings = nn.Embedding(1024, hidden_size)
        self.aa_embeddings = nn.Embedding(22, hidden_size)  # 20 AAs + start/end
        self.structure_embeddings = nn.Linear(3, hidden_size)  # For structural features

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=4*hidden_size,
                dropout=dropout_prob,
                batch_first=True
            ) for _ in range(num_hidden_layers)
        ])

        # Domain prediction heads
        self.domain_start_predictor = nn.Linear(hidden_size, 2)  # Binary classification
        self.domain_end_predictor = nn.Linear(hidden_size, 2)
        self.domain_type_predictor = nn.Linear(hidden_size, 10)  # 10 common domain types

        # Confidence scoring
        self.confidence_scorer = nn.Linear(hidden_size, 1)

        # Loss functions
        self.boundary_loss = nn.CrossEntropyLoss()
        self.type_loss = nn.CrossEntropyLoss()
        self.confidence_loss = nn.MSELoss()

    def forward(
        self,
        sequence_ids: torch.Tensor,
        structure_coords: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        domain_labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for domain prediction."""
        batch_size, seq_length = sequence_ids.shape
        device = sequence_ids.device

        # Generate position embeddings
        position_ids = torch.arange(seq_length, device=device).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        # Generate amino acid embeddings
        aa_embeddings = self.aa_embeddings(sequence_ids)

        # Combine embeddings
        embeddings = aa_embeddings + position_embeddings

        # Add structural information if available
        if structure_coords is not None:
            structure_embeddings = self.structure_embeddings(structure_coords)
            embeddings = embeddings + structure_embeddings

        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=~attention_mask if attention_mask is not None else None)

        # Predict domain boundaries and types
        start_logits = self.domain_start_predictor(hidden_states)
        end_logits = self.domain_end_predictor(hidden_states)
        type_logits = self.domain_type_predictor(hidden_states)
        confidence_scores = self.confidence_scorer(hidden_states).squeeze(-1)

        outputs = {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "type_logits": type_logits,
            "confidence_scores": confidence_scores,
            "hidden_states": hidden_states
        }

        # Calculate loss if labels provided
        if domain_labels is not None:
            loss = self._calculate_loss(
                start_logits=start_logits,
                end_logits=end_logits,
                type_logits=type_logits,
                confidence_scores=confidence_scores,
                labels=domain_labels
            )
            outputs["loss"] = loss

        return outputs

    def _calculate_loss(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        type_logits: torch.Tensor,
        confidence_scores: torch.Tensor,
        labels: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate combined loss for domain prediction."""
        start_loss = self.boundary_loss(
            start_logits.view(-1, 2),
            labels["start_positions"].view(-1)
        )
        end_loss = self.boundary_loss(
            end_logits.view(-1, 2),
            labels["end_positions"].view(-1)
        )
        type_loss = self.type_loss(
            type_logits.view(-1, 10),
            labels["domain_types"].view(-1)
        )
        confidence_loss = self.confidence_loss(
            confidence_scores,
            labels["confidence_scores"]
        )

        return start_loss + end_loss + type_loss + confidence_loss

    def predict_domains(
        self,
        sequence: str,
        structure: Optional[AtomArray] = None,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Union[int, str, float]]]:
        """Predict protein domains from sequence and optional structure."""
        # Convert sequence to tensor
        sequence_ids = torch.tensor(
            [self.aa_to_id[aa] for aa in sequence],
            device=next(self.parameters()).device
        ).unsqueeze(0)

        # Process structure if provided
        structure_coords = None
        if structure is not None:
            coords = structure.coord
            structure_coords = torch.tensor(coords, device=sequence_ids.device).unsqueeze(0)

        # Get predictions
        with torch.no_grad():
            outputs = self.forward(sequence_ids, structure_coords)

        # Process predictions
        start_probs = torch.softmax(outputs["start_logits"][0], dim=-1)[:, 1]
        end_probs = torch.softmax(outputs["end_logits"][0], dim=-1)[:, 1]
        type_probs = torch.softmax(outputs["type_logits"][0], dim=-1)
        confidence_scores = torch.sigmoid(outputs["confidence_scores"][0])

        # Find domain boundaries
        domains = []
        seq_length = len(sequence)

        for start_idx in range(seq_length):
            if start_probs[start_idx] > confidence_threshold:
                for end_idx in range(start_idx + 10, min(start_idx + 300, seq_length)):
                    if end_probs[end_idx] > confidence_threshold:
                        # Get domain type
                        domain_type_idx = type_probs[start_idx:end_idx+1].mean(0).argmax()
                        confidence = confidence_scores[start_idx:end_idx+1].mean()

                        if confidence > confidence_threshold:
                            domains.append({
                                "start": start_idx,
                                "end": end_idx,
                                "type": self.id_to_domain_type[domain_type_idx.item()],
                                "confidence": confidence.item()
                            })

        # Merge overlapping domains
        domains = self._merge_overlapping_domains(domains)
        return domains

    def _merge_overlapping_domains(
        self,
        domains: List[Dict[str, Union[int, str, float]]],
        overlap_threshold: float = 0.5
    ) -> List[Dict[str, Union[int, str, float]]]:
        """Merge overlapping domain predictions."""
        if not domains:
            return domains

        # Sort domains by start position
        domains.sort(key=lambda x: x["start"])
        merged = []
        current = domains[0]

        for next_domain in domains[1:]:
            overlap = min(current["end"], next_domain["end"]) - max(current["start"], next_domain["start"])
            length = min(current["end"] - current["start"], next_domain["end"] - next_domain["start"])

            if overlap / length > overlap_threshold:
                # Merge domains
                if next_domain["confidence"] > current["confidence"]:
                    current = next_domain
            else:
                merged.append(current)
                current = next_domain

        merged.append(current)
        return merged
