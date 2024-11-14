"""
Graph-Based Sampling implementation for ProteinFlex.
Implements message passing and local structure preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        node_dim: int = 768,
        edge_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """Initialize message passing layer."""
        super().__init__()
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

        self.edge_update = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim)
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        adjacency: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for message passing.

        Args:
            nodes: Node features [batch_size, num_nodes, node_dim]
            edges: Edge features [batch_size, num_nodes, num_nodes, edge_dim]
            adjacency: Adjacency matrix [batch_size, num_nodes, num_nodes]

        Returns:
            Updated node and edge features
        """
        B, N, D = nodes.shape
        _, _, _, E = edges.shape

        # Aggregate messages for nodes
        edges_adj = edges * adjacency.unsqueeze(-1)  # [B, N, N, E]
        nodes_expanded = nodes.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]

        # Compute messages
        messages = torch.cat([edges_adj, nodes_expanded], dim=-1)  # [B, N, N, E+D]
        messages = messages.sum(dim=2)  # [B, N, E+D]

        # Update nodes
        nodes_updated = nodes + self.node_update(messages)

        # Update edges
        nodes_i = nodes.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        nodes_j = nodes.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        edge_input = torch.cat([edges, nodes_i, nodes_j], dim=-1)  # [B, N, N, E+2D]
        edges_updated = edges + self.edge_update(edge_input)

        return nodes_updated, edges_updated

class GraphBasedSampler(nn.Module):
    def __init__(
        self,
        node_dim: int = 768,
        edge_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        """
        Initialize Graph-Based Sampler.

        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension
            num_layers: Number of message passing layers
            dropout: Dropout rate
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Edge feature initialization
        self.edge_embedding = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.ReLU()
        )

        # Message passing layers
        self.layers = nn.ModuleList([
            MessagePassingLayer(node_dim, edge_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Structure preservation network
        self.structure_preserving = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3D coordinates
        )

    def initialize_edges(
        self,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize edge features from coordinates."""
        B, N = coords.shape[:2]
        device = coords.device

        # Compute pairwise distances
        coords_i = coords.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = coords.unsqueeze(1)  # [B, 1, N, 3]
        diff = coords_i - coords_j  # [B, N, N, 3]
        distances = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]

        # Create adjacency matrix (connect nearby residues)
        adjacency = (distances.squeeze(-1) < 10.0).float()  # [B, N, N]

        # Initialize edge features
        edges = self.edge_embedding(distances)  # [B, N, N, edge_dim]

        return edges, adjacency

    def forward(
        self,
        nodes: torch.Tensor,
        coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            nodes: Node features [batch_size, num_nodes, node_dim]
            coords: Optional 3D coordinates [batch_size, num_nodes, 3]

        Returns:
            Updated node features and predicted coordinates
        """
        if coords is None:
            coords = self.structure_preserving(nodes)

        # Initialize graph structure
        edges, adjacency = self.initialize_edges(coords)

        # Apply message passing layers
        current_nodes = nodes
        current_edges = edges

        for layer in self.layers:
            current_nodes, current_edges = layer(
                current_nodes,
                current_edges,
                adjacency
            )

        # Predict coordinates
        coords_pred = self.structure_preserving(current_nodes)

        return current_nodes, coords_pred

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate protein features using graph-based sampling.

        Args:
            batch_size: Number of samples to generate
            seq_len: Sequence length
            device: Device to generate on
            temperature: Sampling temperature

        Returns:
            Tuple of (node features, 3D coordinates)
        """
        # Initialize random node features
        nodes = torch.randn(
            batch_size, seq_len, self.node_dim,
            device=device
        ) * temperature

        # Generate initial coordinates
        coords = self.structure_preserving(nodes)

        # Refine through message passing
        nodes_refined, coords_refined = self.forward(nodes, coords)

        return nodes_refined, coords_refined

    def compute_loss(
        self,
        pred_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        pred_coords: torch.Tensor,
        target_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            pred_nodes: Predicted node features
            target_nodes: Target node features
            pred_coords: Predicted coordinates
            target_coords: Target coordinates

        Returns:
            Combined loss value
        """
        # Node feature reconstruction loss
        node_loss = F.mse_loss(pred_nodes, target_nodes)

        # Coordinate prediction loss
        coord_loss = F.mse_loss(pred_coords, target_coords)

        # Distance matrix consistency loss
        pred_dist = torch.cdist(pred_coords, pred_coords)
        target_dist = torch.cdist(target_coords, target_coords)
        dist_loss = F.mse_loss(pred_dist, target_dist)

        return node_loss + coord_loss + dist_loss
