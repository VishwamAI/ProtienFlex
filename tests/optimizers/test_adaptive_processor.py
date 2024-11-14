import pytest
import torch
import torch.nn as nn
from models.optimizers.adaptive_processor import AdaptiveProcessor

class SimpleModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(self, x):
        return self.layers(x)

@pytest.fixture
def adaptive_processor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return AdaptiveProcessor(device=device)

@pytest.fixture
def sample_model():
    return SimpleModel()

@pytest.fixture
def sample_input():
    return torch.randn(2, 10, 64)  # batch_size=2, seq_len=10, hidden_size=64

def test_adaptive_processor_initialization(adaptive_processor):
    """Test adaptive processor initialization."""
    assert adaptive_processor is not None
    assert adaptive_processor.target_latency == 1.0
    assert adaptive_processor.enable_profiling is True
    assert isinstance(adaptive_processor.hardware_info, dict)

def test_hardware_detection(adaptive_processor):
    """Test hardware detection functionality."""
    info = adaptive_processor.hardware_info
    assert "device_type" in info
    assert "cores" in info
    assert isinstance(info["cores"], int)
    assert info["cores"] > 0

def test_optimize_for_hardware(adaptive_processor, sample_model, sample_input):
    """Test hardware-specific optimization."""
    optimized_model = adaptive_processor.optimize_for_hardware(
        sample_model,
        sample_input.shape,
        target_accuracy=0.95
    )

    assert isinstance(optimized_model, nn.Module)
    # Test model can still process inputs
    with torch.no_grad():
        output = optimized_model(sample_input)
    assert output.shape == sample_input.shape

def test_profile_execution(adaptive_processor, sample_model, sample_input):
    """Test execution profiling."""
    metrics = adaptive_processor.profile_execution(
        sample_model,
        sample_input,
        num_iterations=10
    )

    assert "average_latency" in metrics
    assert "p95_latency" in metrics
    assert "throughput" in metrics
    assert metrics["average_latency"] > 0
    assert metrics["throughput"] > 0

def test_adapt_processing(adaptive_processor, sample_model):
    """Test processing adaptation based on metrics."""
    current_latency = 1.5  # Above target
    current_accuracy = 0.96

    adapted_model, changes = adaptive_processor.adapt_processing(
        sample_model,
        current_latency,
        current_accuracy
    )

    assert isinstance(adapted_model, nn.Module)
    assert isinstance(changes, dict)
    assert "batch_size" in changes or "precision" in changes

@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_mixed_precision(adaptive_processor, sample_model, precision):
    """Test mixed precision support."""
    if precision == "fp16" and not torch.cuda.is_available():
        pytest.skip("CUDA required for fp16 testing")

    model = adaptive_processor._enable_mixed_precision(sample_model) if precision == "fp16" else sample_model
    assert isinstance(model, nn.Module)

    if precision == "fp16":
        assert any(p.dtype == torch.float16 for p in model.parameters())

def test_memory_access_optimization(adaptive_processor, sample_model):
    """Test memory access pattern optimization."""
    optimized_model = adaptive_processor._optimize_memory_access(sample_model)
    assert isinstance(optimized_model, nn.Module)

    # Verify weights are contiguous
    for param in optimized_model.parameters():
        assert param.is_contiguous()

def test_operation_fusion(adaptive_processor, sample_model):
    """Test operation fusion optimization."""
    if not torch.jit.is_scripting():
        optimized_model = adaptive_processor._fuse_operations(sample_model)
        assert isinstance(optimized_model, (nn.Module, torch.jit.ScriptModule))

def test_performance_metrics(adaptive_processor, sample_model, sample_input):
    """Test performance metrics tracking."""
    # Profile execution to generate metrics
    _ = adaptive_processor.profile_execution(
        sample_model,
        sample_input,
        num_iterations=10
    )

    metrics = adaptive_processor.get_performance_metrics()
    assert "inference_times" in metrics
    assert "throughput" in metrics
    assert "accuracy" in metrics
    assert "latency" in metrics

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_different_batch_sizes(adaptive_processor, sample_model, batch_size):
    """Test adaptive processor with different batch sizes."""
    input_tensor = torch.randn(batch_size, 10, 64)

    metrics = adaptive_processor.profile_execution(
        sample_model,
        input_tensor,
        num_iterations=10
    )

    assert metrics["average_latency"] > 0
    assert metrics["throughput"] > 0

@pytest.mark.parametrize("target_latency", [0.1, 0.5, 1.0])
def test_different_latency_targets(target_latency):
    """Test adaptive processor with different latency targets."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AdaptiveProcessor(device=device, target_latency=target_latency)

    assert processor.target_latency == target_latency

def test_layer_optimization(adaptive_processor, sample_model):
    """Test layer-wise optimization."""
    current_latency = 1.5  # Above target
    optimized_model = adaptive_processor._optimize_layer_configuration(
        sample_model,
        current_latency
    )

    assert isinstance(optimized_model, nn.Module)
