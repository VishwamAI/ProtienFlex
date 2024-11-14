import pytest
import torch
import torch.nn as nn
from models.optimizers.performance_monitor import PerformanceMonitor, PerformanceMetrics
import tempfile
from pathlib import Path

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        return self.linear(x)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 64)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def performance_monitor(temp_log_dir):
    return PerformanceMonitor(
        target_latency=1.0,
        target_accuracy=0.95,
        target_memory_efficiency=0.8,
        log_dir=temp_log_dir
    )

@pytest.fixture
def sample_model():
    return SimpleModel()

@pytest.fixture
def sample_dataloader():
    dataset = SimpleDataset()
    return torch.utils.data.DataLoader(dataset, batch_size=16)

@pytest.fixture
def sample_input():
    return torch.randn(32, 64)

def test_performance_monitor_initialization(performance_monitor):
    """Test performance monitor initialization."""
    assert performance_monitor.target_latency == 1.0
    assert performance_monitor.target_accuracy == 0.95
    assert performance_monitor.target_memory_efficiency == 0.8
    assert isinstance(performance_monitor.metrics_history, dict)

def test_measure_inference_time(performance_monitor, sample_model, sample_input):
    """Test inference time measurement."""
    metrics = performance_monitor.measure_inference_time(
        sample_model,
        sample_input,
        num_runs=10,
        warmup_runs=2
    )

    assert "average_inference_time" in metrics
    assert "p95_latency" in metrics
    assert "throughput" in metrics
    assert metrics["average_inference_time"] > 0
    assert metrics["throughput"] > 0

def test_measure_accuracy(performance_monitor, sample_model, sample_dataloader):
    """Test accuracy measurement."""
    criterion = nn.CrossEntropyLoss()
    metrics = performance_monitor.measure_accuracy(
        sample_model,
        sample_dataloader,
        criterion
    )

    assert "accuracy" in metrics
    assert "validation_loss" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["validation_loss"] >= 0

def test_measure_memory_usage(performance_monitor, sample_model, sample_input):
    """Test memory usage measurement."""
    metrics = performance_monitor.measure_memory_usage(sample_model, sample_input)

    if torch.cuda.is_available():
        assert "peak_memory_mb" in metrics
        assert "current_memory_mb" in metrics
        assert "memory_efficiency" in metrics
    else:
        assert "memory_usage_mb" in metrics

def test_validate_performance(
    performance_monitor,
    sample_model,
    sample_dataloader,
    sample_input
):
    """Test performance validation."""
    criterion = nn.CrossEntropyLoss()
    meets_targets, metrics = performance_monitor.validate_performance(
        sample_model,
        sample_dataloader,
        criterion,
        sample_input
    )

    assert isinstance(meets_targets, bool)
    assert isinstance(metrics, dict)
    assert "inference_time" in metrics
    assert "accuracy" in metrics
    assert "memory_usage" in metrics
    assert "throughput" in metrics
    assert "latency_p95" in metrics

def test_metrics_history_update(performance_monitor, sample_model, sample_dataloader, sample_input):
    """Test metrics history updating."""
    criterion = nn.CrossEntropyLoss()
    _ = performance_monitor.validate_performance(
        sample_model,
        sample_dataloader,
        criterion,
        sample_input
    )

    history = performance_monitor.metrics_history
    assert len(history["inference_times"]) > 0
    assert len(history["accuracies"]) > 0
    assert len(history["memory_usage"]) > 0
    assert len(history["throughput"]) > 0
    assert len(history["latency_p95"]) > 0

def test_save_metrics_history(performance_monitor, temp_log_dir):
    """Test saving metrics history."""
    # Add some test metrics
    performance_monitor.metrics_history["inference_times"].append(0.1)
    performance_monitor.metrics_history["accuracies"].append(0.95)

    filename = "test_metrics.json"
    performance_monitor.save_metrics_history(filename)

    saved_file = Path(temp_log_dir) / filename
    assert saved_file.exists()
    assert saved_file.stat().st_size > 0

def test_get_optimization_suggestions(
    performance_monitor,
    sample_model,
    sample_dataloader,
    sample_input
):
    """Test optimization suggestions generation."""
    criterion = nn.CrossEntropyLoss()
    _ = performance_monitor.validate_performance(
        sample_model,
        sample_dataloader,
        criterion,
        sample_input
    )

    suggestions = performance_monitor.get_optimization_suggestions()
    assert isinstance(suggestions, list)
    assert all(isinstance(s, str) for s in suggestions)

@pytest.mark.parametrize("target_latency", [0.1, 0.5, 1.0])
def test_different_latency_targets(target_latency, temp_log_dir):
    """Test performance monitor with different latency targets."""
    monitor = PerformanceMonitor(
        target_latency=target_latency,
        target_accuracy=0.95,
        target_memory_efficiency=0.8,
        log_dir=temp_log_dir
    )
    assert monitor.target_latency == target_latency

@pytest.mark.parametrize("target_accuracy", [0.9, 0.95, 0.99])
def test_different_accuracy_targets(target_accuracy, temp_log_dir):
    """Test performance monitor with different accuracy targets."""
    monitor = PerformanceMonitor(
        target_latency=1.0,
        target_accuracy=target_accuracy,
        target_memory_efficiency=0.8,
        log_dir=temp_log_dir
    )
    assert monitor.target_accuracy == target_accuracy

def test_log_file_creation(performance_monitor, temp_log_dir):
    """Test log file creation and writing."""
    log_file = Path(temp_log_dir) / "performance.log"
    assert log_file.exists()

    # Generate some logs
    performance_monitor._log_metrics("test", {"value": 1.0})

    # Check log file content
    assert log_file.stat().st_size > 0
