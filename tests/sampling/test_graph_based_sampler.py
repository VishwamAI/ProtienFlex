"""
Tests for the Graph-Based Sampler.
"""

import torch
import pytest
from models.sampling.graph_based_sampler import GraphBasedSampler, MessagePassingLayer

@pytest.fixture
def sampler():
    return GraphBasedSampler(
        node_dim=768,
        edge_dim=64,
        hidden_dim=512,
        num_layers=3
    )

def test_sampler_initialization(sampler):
    """Test sampler initialization."""
    assert isinstance(sampler, GraphBasedSampler)
    assert sampler.node_dim == 768
    assert sampler.edge_dim == 64
    assert len(sampler.layers) == 3

def test_message_passing_layer():
    """Test message passing layer."""
    layer = MessagePassingLayer(
        node_dim=768,
        edge_dim=64,
        hidden_dim=512
    )
    batch_size, num_nodes = 2, 10
    nodes = torch.randn(batch_size, num_nodes, 768)
    edges = torch.randn(batch_size, num_nodes, num_nodes, 64)
    adjacency = torch.ones(batch_size, num_nodes, num_nodes)

    nodes_updated, edges_updated = layer(nodes, edges, adjacency)
    assert nodes_updated.shape == nodes.shape
    assert edges_updated.shape == edges.shape
    assert not torch.isnan(nodes_updated).any()
    assert not torch.isnan(edges_updated).any()

def test_edge_initialization(sampler):
    """Test edge feature initialization."""
    batch_size, num_nodes = 2, 10
    coords = torch.randn(batch_size, num_nodes, 3)

    edges, adjacency = sampler.initialize_edges(coords)
    assert edges.shape == (batch_size, num_nodes, num_nodes, sampler.edge_dim)
    assert adjacency.shape == (batch_size, num_nodes, num_nodes)
    assert not torch.isnan(edges).any()
    assert not torch.isnan(adjacency).any()

def test_forward_pass(sampler):
    """Test forward pass with and without coordinates."""
    batch_size, num_nodes = 2, 10
    nodes = torch.randn(batch_size, num_nodes, sampler.node_dim)
    coords = torch.randn(batch_size, num_nodes, 3)

    # Test with coordinates
    nodes_updated1, coords_pred1 = sampler(nodes, coords)
    assert nodes_updated1.shape == nodes.shape
    assert coords_pred1.shape == coords.shape
    assert not torch.isnan(nodes_updated1).any()
    assert not torch.isnan(coords_pred1).any()

    # Test without coordinates
    nodes_updated2, coords_pred2 = sampler(nodes)
    assert nodes_updated2.shape == nodes.shape
    assert coords_pred2.shape == coords.shape
    assert not torch.isnan(nodes_updated2).any()
    assert not torch.isnan(coords_pred2).any()

def test_sampling(sampler):
    """Test protein feature generation."""
    batch_size, seq_len = 2, 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = sampler.to(device)

    nodes, coords = sampler.sample(
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        temperature=0.8
    )

    assert nodes.shape == (batch_size, seq_len, sampler.node_dim)
    assert coords.shape == (batch_size, seq_len, 3)
    assert not torch.isnan(nodes).any()
    assert not torch.isnan(coords).any()

def test_loss_computation(sampler):
    """Test loss computation."""
    batch_size, seq_len = 2, 10
    pred_nodes = torch.randn(batch_size, seq_len, sampler.node_dim)
    target_nodes = torch.randn_like(pred_nodes)
    pred_coords = torch.randn(batch_size, seq_len, 3)
    target_coords = torch.randn_like(pred_coords)

    loss = sampler.compute_loss(pred_nodes, target_nodes, pred_coords, target_coords)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar tensor
    assert not torch.isnan(loss)
    assert loss.item() > 0

def test_structure_preservation(sampler):
    """Test structure preservation network."""
    batch_size, seq_len = 2, 10
    nodes = torch.randn(batch_size, seq_len, sampler.node_dim)
    coords = sampler.structure_preserving(nodes)

    assert coords.shape == (batch_size, seq_len, 3)
    assert not torch.isnan(coords).any()
