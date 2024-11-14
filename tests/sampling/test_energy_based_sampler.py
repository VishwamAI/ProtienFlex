"""
Tests for the Energy-Based Sampler.
"""

import torch
import pytest
from models.sampling.energy_based_sampler import EnergyBasedSampler

@pytest.fixture
def sampler():
    return EnergyBasedSampler(
        feature_dim=768,
        hidden_dim=512,
        num_steps=10  # Reduced for testing
    )

def test_sampler_initialization(sampler):
    """Test sampler initialization."""
    assert isinstance(sampler, EnergyBasedSampler)
    assert sampler.feature_dim == 768
    assert sampler.hidden_dim == 512
    assert sampler.num_steps == 10

def test_energy_computation(sampler):
    """Test energy computation."""
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, sampler.feature_dim)
    energy = sampler.compute_energy(x)

    assert energy.shape == (batch_size, seq_len)
    assert not torch.isnan(energy).any()

def test_gradient_computation(sampler):
    """Test energy gradient computation."""
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, sampler.feature_dim)
    grad = sampler.compute_gradients(x)

    assert grad.shape == x.shape
    assert not torch.isnan(grad).any()

def test_langevin_dynamics(sampler):
    """Test Langevin dynamics sampling."""
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, sampler.feature_dim)
    samples, energies = sampler.langevin_dynamics(x, num_steps=5)

    assert samples.shape == x.shape
    assert len(energies) == 5
    assert not torch.isnan(samples).any()
    assert not torch.isnan(energies).any()

def test_forward_pass(sampler):
    """Test forward pass."""
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, sampler.feature_dim)
    samples, energies = sampler(x, num_steps=5)

    assert samples.shape == x.shape
    assert len(energies) == 5
    assert not torch.isnan(samples).any()
    assert not torch.isnan(energies).any()

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
    real = torch.randn(batch_size, seq_len, sampler.feature_dim)
    fake = torch.randn(batch_size, seq_len, sampler.feature_dim)

    loss = sampler.compute_loss(real, fake)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar tensor
    assert not torch.isnan(loss)
    assert loss.item() > 0  # Loss should be positive

def test_structure_validation(sampler):
    """Test structure validation network."""
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, sampler.feature_dim)
    validity = sampler.structure_validator(x)

    assert validity.shape == (batch_size, seq_len, 1)
    assert (validity >= 0).all() and (validity <= 1).all()
    assert not torch.isnan(validity).any()
