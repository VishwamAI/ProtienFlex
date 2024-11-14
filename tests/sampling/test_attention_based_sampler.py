"""
Tests for the Attention-Based Sampler.
"""

import torch
import pytest
from models.sampling.attention_based_sampler import AttentionBasedSampler, StructureAwareAttention

@pytest.fixture
def sampler():
    return AttentionBasedSampler(
        feature_dim=768,
        hidden_dim=512,
        num_layers=3,
        num_heads=8
    )

def test_sampler_initialization(sampler):
    """Test sampler initialization."""
    assert isinstance(sampler, AttentionBasedSampler)
    assert sampler.feature_dim == 768
    assert sampler.hidden_dim == 512
    assert len(sampler.layers) == 3

def test_structure_aware_attention():
    """Test structure-aware attention mechanism."""
    attention = StructureAwareAttention(
        feature_dim=768,
        num_heads=8
    )
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 768)
    structure_bias = torch.randn(batch_size, seq_len, 768)

    output = attention(x, structure_bias)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()

def test_forward_pass(sampler):
    """Test forward pass with and without structure info."""
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, sampler.feature_dim)
    structure_info = torch.randn_like(x)

    # Test without structure info
    output1 = sampler(x)
    assert output1.shape == x.shape
    assert not torch.isnan(output1).any()

    # Test with structure info
    output2 = sampler(x, structure_info)
    assert output2.shape == x.shape
    assert not torch.isnan(output2).any()

def test_sampling(sampler):
    """Test protein feature generation."""
    batch_size, seq_len = 2, 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = sampler.to(device)
    structure_info = torch.randn(batch_size, seq_len, sampler.feature_dim, device=device)

    # Test without structure info
    generated1 = sampler.sample(
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        temperature=0.8
    )
    assert generated1.shape == (batch_size, seq_len, sampler.feature_dim)
    assert not torch.isnan(generated1).any()

    # Test with structure info
    generated2 = sampler.sample(
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        structure_info=structure_info,
        temperature=0.8
    )
    assert generated2.shape == (batch_size, seq_len, sampler.feature_dim)
    assert not torch.isnan(generated2).any()

def test_loss_computation(sampler):
    """Test loss computation."""
    batch_size, seq_len = 2, 10
    pred = torch.randn(batch_size, seq_len, sampler.feature_dim)
    target = torch.randn_like(pred)
    structure_info = torch.randn_like(pred)

    # Test without structure info
    loss1 = sampler.compute_loss(pred, target)
    assert isinstance(loss1, torch.Tensor)
    assert loss1.ndim == 0  # Scalar tensor
    assert not torch.isnan(loss1)
    assert loss1.item() > 0

    # Test with structure info
    loss2 = sampler.compute_loss(pred, target, structure_info)
    assert isinstance(loss2, torch.Tensor)
    assert loss2.ndim == 0
    assert not torch.isnan(loss2)
    assert loss2.item() > 0

def test_structure_encoder(sampler):
    """Test structure encoder network."""
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, sampler.feature_dim)
    encoded = sampler.structure_encoder(x)

    assert encoded.shape == x.shape
    assert not torch.isnan(encoded).any()
