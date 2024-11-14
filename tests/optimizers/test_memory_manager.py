import pytest
import torch
import torch.nn as nn
from models.optimizers.memory_manager import MemoryManager

class SimpleTransformer(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4
        )

    def forward(self, x, cached_attention=None):
        output = self.encoder(x)
        return {"output": output, "attention_cache": None}

@pytest.fixture
def memory_manager():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MemoryManager(device=device)

@pytest.fixture
def sample_model():
    return SimpleTransformer()

@pytest.fixture
def sample_input():
    return torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, hidden_size=64

def test_memory_manager_initialization(memory_manager):
    """Test memory manager initialization."""
    assert memory_manager is not None
    assert memory_manager.enable_amp is True
    assert memory_manager.enable_checkpoint is True
    assert memory_manager.attention_cache == {}

def test_optimize_memory_allocation(memory_manager, sample_model, sample_input):
    """Test memory allocation optimization."""
    batch_size = sample_input.size(0)
    sequence_length = sample_input.size(1)

    result = memory_manager.optimize_memory_allocation(
        sample_model,
        batch_size,
        sequence_length
    )

    assert "optimal_batch_size" in result
    assert isinstance(result["optimal_batch_size"], int)
    assert result["optimal_batch_size"] > 0

def test_forward_with_memory_optimization(memory_manager, sample_model, sample_input):
    """Test forward pass with memory optimization."""
    outputs = memory_manager.forward_with_memory_optimization(
        sample_model,
        {"x": sample_input},
        cache_key="test"
    )

    assert "output" in outputs
    assert outputs["output"].shape == sample_input.shape

def test_checkpoint_sequential(memory_manager, sample_input):
    """Test gradient checkpointing."""
    functions = [
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64)
    ]

    output = memory_manager.checkpoint_sequential(
        functions,
        segments=2,
        input_tensor=sample_input
    )

    assert output.shape == sample_input.shape

def test_clear_cache(memory_manager):
    """Test cache clearing functionality."""
    # Add some test data to caches
    memory_manager.attention_cache["test"] = torch.randn(2, 10, 64)
    memory_manager.gradient_cache["test"] = torch.randn(2, 10, 64)
    memory_manager.activation_cache["test"] = torch.randn(2, 10, 64)

    # Clear specific cache
    memory_manager.clear_cache("attention")
    assert len(memory_manager.attention_cache) == 0
    assert len(memory_manager.gradient_cache) > 0

    # Clear all caches
    memory_manager.clear_cache()
    assert len(memory_manager.attention_cache) == 0
    assert len(memory_manager.gradient_cache) == 0
    assert len(memory_manager.activation_cache) == 0

def test_optimize_for_inference(memory_manager, sample_model):
    """Test inference optimization."""
    optimized_model = memory_manager.optimize_for_inference(
        sample_model,
        quantize=True,
        optimize_graph=False  # Set to False for testing
    )

    assert isinstance(optimized_model, nn.Module)

def test_adaptive_batch_size(memory_manager, sample_model):
    """Test adaptive batch size adjustment."""
    initial_batch_size = 32
    sequence_length = 10

    new_batch_size = memory_manager.adaptive_batch_size(
        sample_model,
        initial_batch_size,
        sequence_length
    )

    assert isinstance(new_batch_size, int)
    assert new_batch_size > 0
    assert new_batch_size <= initial_batch_size * 2

def test_memory_stats(memory_manager, sample_model, sample_input):
    """Test memory statistics tracking."""
    # Perform some operations to generate memory stats
    _ = memory_manager.forward_with_memory_optimization(
        sample_model,
        {"x": sample_input},
        cache_key="test"
    )

    stats = memory_manager.get_memory_stats()
    assert "peak_allocated" in stats
    assert "current_allocated" in stats
    assert "cached_memory" in stats

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(memory_manager, sample_model, batch_size):
    """Test memory manager with different batch sizes."""
    input_tensor = torch.randn(batch_size, 10, 64)

    result = memory_manager.optimize_memory_allocation(
        sample_model,
        batch_size,
        sequence_length=10
    )

    assert result["optimal_batch_size"] > 0
    assert result["optimal_batch_size"] <= batch_size

@pytest.mark.parametrize("enable_amp", [True, False])
def test_mixed_precision_settings(enable_amp):
    """Test memory manager with different mixed precision settings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = MemoryManager(device=device, enable_amp=enable_amp)

    assert manager.enable_amp == enable_amp
    assert manager.scaler.is_enabled() == enable_amp
