import pytest
import torch
import torch.nn as nn
from models.optimizers.memory_manager import MemoryManager
from models.optimizers.adaptive_processor import AdaptiveProcessor
from models.optimizers.performance_monitor import PerformanceMonitor
import tempfile
from pathlib import Path

class ProteinGenerationModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4
        )
        self.decoder = nn.Linear(hidden_size, 20)  # 20 amino acids

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, size=100, seq_length=10):
        self.data = torch.randn(size, seq_length, 64)  # Input embeddings
        self.labels = torch.randint(0, 20, (size, seq_length))  # Amino acid labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def optimization_components(temp_log_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "memory_manager": MemoryManager(device=device),
        "adaptive_processor": AdaptiveProcessor(device=device),
        "performance_monitor": PerformanceMonitor(
            target_latency=1.0,
            target_accuracy=0.95,
            target_memory_efficiency=0.8,
            log_dir=temp_log_dir
        )
    }

@pytest.fixture
def model():
    return ProteinGenerationModel()

@pytest.fixture
def dataloader():
    dataset = ProteinDataset()
    return torch.utils.data.DataLoader(dataset, batch_size=16)

@pytest.fixture
def sample_input():
    return torch.randn(32, 10, 64)  # batch_size=32, seq_len=10, hidden_size=64

def test_complete_optimization_pipeline(
    optimization_components,
    model,
    dataloader,
    sample_input
):
    """Test the complete optimization pipeline."""
    memory_manager = optimization_components["memory_manager"]
    adaptive_processor = optimization_components["adaptive_processor"]
    performance_monitor = optimization_components["performance_monitor"]

    # Step 1: Memory Optimization
    batch_size = sample_input.size(0)
    sequence_length = sample_input.size(1)
    memory_config = memory_manager.optimize_memory_allocation(
        model,
        batch_size,
        sequence_length
    )
    assert "optimal_batch_size" in memory_config

    # Step 2: Hardware Optimization
    optimized_model = adaptive_processor.optimize_for_hardware(
        model,
        sample_input.shape,
        target_accuracy=0.95
    )
    assert isinstance(optimized_model, nn.Module)

    # Step 3: Performance Validation
    criterion = nn.CrossEntropyLoss()
    meets_targets, metrics = performance_monitor.validate_performance(
        optimized_model,
        dataloader,
        criterion,
        sample_input
    )

    assert isinstance(meets_targets, bool)
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in [
        "inference_time",
        "accuracy",
        "memory_usage",
        "throughput",
        "latency_p95"
    ])

def test_pipeline_adaptation(
    optimization_components,
    model,
    dataloader,
    sample_input
):
    """Test pipeline adaptation based on performance metrics."""
    adaptive_processor = optimization_components["adaptive_processor"]
    performance_monitor = optimization_components["performance_monitor"]

    # Initial optimization
    optimized_model = adaptive_processor.optimize_for_hardware(
        model,
        sample_input.shape
    )

    # Get initial performance metrics
    criterion = nn.CrossEntropyLoss()
    _, initial_metrics = performance_monitor.validate_performance(
        optimized_model,
        dataloader,
        criterion,
        sample_input
    )

    # Adapt based on metrics
    adapted_model, changes = adaptive_processor.adapt_processing(
        optimized_model,
        initial_metrics["inference_time"],
        initial_metrics["accuracy"]
    )

    # Validate adaptation
    _, adapted_metrics = performance_monitor.validate_performance(
        adapted_model,
        dataloader,
        criterion,
        sample_input
    )

    assert isinstance(changes, dict)
    assert adapted_metrics["inference_time"] > 0

def test_pipeline_memory_efficiency(
    optimization_components,
    model,
    sample_input
):
    """Test memory efficiency of the complete pipeline."""
    memory_manager = optimization_components["memory_manager"]
    performance_monitor = optimization_components["performance_monitor"]

    # Track initial memory usage
    initial_memory = memory_manager.get_memory_stats()

    # Run optimization pipeline
    optimized_model = memory_manager.optimize_for_inference(model)
    _ = memory_manager.forward_with_memory_optimization(
        optimized_model,
        {"x": sample_input},
        cache_key="test"
    )

    # Check final memory usage
    final_memory = memory_manager.get_memory_stats()
    assert final_memory["peak_allocated"] >= 0
    assert final_memory["cached_memory"] >= 0

def test_pipeline_performance_tracking(
    optimization_components,
    model,
    dataloader,
    sample_input
):
    """Test performance tracking across the pipeline."""
    adaptive_processor = optimization_components["adaptive_processor"]
    performance_monitor = optimization_components["performance_monitor"]

    # Track performance across multiple iterations
    metrics_history = []
    criterion = nn.CrossEntropyLoss()

    for _ in range(3):  # Test multiple iterations
        # Optimize and adapt
        optimized_model = adaptive_processor.optimize_for_hardware(
            model,
            sample_input.shape
        )

        # Measure performance
        _, metrics = performance_monitor.validate_performance(
            optimized_model,
            dataloader,
            criterion,
            sample_input
        )
        metrics_history.append(metrics)

    assert len(metrics_history) == 3
    assert all(isinstance(m, dict) for m in metrics_history)

def test_pipeline_error_handling(
    optimization_components,
    model,
    sample_input
):
    """Test error handling in the optimization pipeline."""
    memory_manager = optimization_components["memory_manager"]
    adaptive_processor = optimization_components["adaptive_processor"]

    # Test with invalid batch size
    with pytest.raises(ValueError):
        memory_manager.optimize_memory_allocation(model, -1, 10)

    # Test with invalid input shape
    with pytest.raises(ValueError):
        adaptive_processor.optimize_for_hardware(model, (-1, 10, 64))

@pytest.mark.parametrize("batch_size", [8, 16, 32])
def test_pipeline_different_batch_sizes(
    optimization_components,
    model,
    batch_size
):
    """Test pipeline with different batch sizes."""
    input_tensor = torch.randn(batch_size, 10, 64)
    memory_manager = optimization_components["memory_manager"]
    adaptive_processor = optimization_components["adaptive_processor"]

    # Optimize for different batch sizes
    memory_config = memory_manager.optimize_memory_allocation(
        model,
        batch_size,
        sequence_length=10
    )
    assert memory_config["optimal_batch_size"] > 0

    optimized_model = adaptive_processor.optimize_for_hardware(
        model,
        input_tensor.shape
    )
    assert isinstance(optimized_model, nn.Module)

@pytest.mark.parametrize("sequence_length", [10, 20, 30])
def test_pipeline_different_sequence_lengths(
    optimization_components,
    model,
    sequence_length
):
    """Test pipeline with different sequence lengths."""
    input_tensor = torch.randn(16, sequence_length, 64)
    memory_manager = optimization_components["memory_manager"]
    adaptive_processor = optimization_components["adaptive_processor"]

    # Optimize for different sequence lengths
    memory_config = memory_manager.optimize_memory_allocation(
        model,
        batch_size=16,
        sequence_length=sequence_length
    )
    assert memory_config["optimal_batch_size"] > 0

    optimized_model = adaptive_processor.optimize_for_hardware(
        model,
        input_tensor.shape
    )
    assert isinstance(optimized_model, nn.Module)
