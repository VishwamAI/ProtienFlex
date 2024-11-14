"""
Tests for the Confidence-Guided Sampler.
"""

import torch
import pytest
from models.sampling.confidence_guided_sampler import ConfidenceGuidedSampler

@pytest.fixture
def sampler():
    return ConfidenceGuidedSampler(
        feature_dim=768,
        hidden_dim=512,
        num_steps=100
    )

def test_sampler_initialization(sampler):
    """Test sampler initialization."""
    assert isinstance(sampler, ConfidenceGuidedSampler)
    assert sampler.feature_dim == 768
    assert sampler.hidden_dim == 512
    assert sampler.num_steps == 100

def test_time_embedding(sampler):
    """Test time embedding generation."""
    t = torch.tensor([0, 50, 99])
    emb = sampler.get_time_embedding(t)
    assert emb.shape == (3, sampler.hidden_dim)
    assert not torch.isnan(emb).any()

def test_forward_pass(sampler):
    """Test forward pass with noise prediction."""
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, sampler.feature_dim)
    noise = torch.randn_like(x)

    noisy_x, pred_noise = sampler(x, noise)
    assert noisy_x.shape == x.shape
    assert pred_noise.shape == noise.shape
    assert not torch.isnan(noisy_x).any()
    assert not torch.isnan(pred_noise).any()

def test_sampling(sampler):
    """Test protein feature generation."""
    batch_size, seq_len = 2, 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = sampler.to(device)

    generated = sampler.sample(
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        temperature=0.8
    )

    assert generated.shape == (batch_size, seq_len, sampler.feature_dim)
    assert not torch.isnan(generated).any()

def test_loss_computation(sampler):
    """Test loss computation."""
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, sampler.feature_dim)
    noise = torch.randn_like(x)
    noisy_x, pred_noise = sampler(x, noise)

    loss = sampler.compute_loss(x, noise, pred_noise)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar tensor
    assert not torch.isnan(loss)
    assert loss.item() > 0  # Loss should be positive

def test_confidence_estimation(sampler):
    """Test confidence estimation network."""
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, sampler.feature_dim)
    confidence = sampler.confidence_net(x)

    assert confidence.shape == (batch_size, seq_len, 1)
    assert (confidence >= 0).all() and (confidence <= 1).all()
    assert not torch.isnan(confidence).any()
