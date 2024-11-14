import pytest
import torch
import torch.nn as nn
import time
from pathlib import Path
import json
import numpy as np
from models.optimizers.memory_manager import MemoryManager
from models.optimizers.adaptive_processor import AdaptiveProcessor
from models.optimizers.performance_monitor import PerformanceMonitor

class BenchmarkProteinModel(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4
        )
        self.decoder = nn.Linear(hidden_size, 20)  # 20 amino acids

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

@pytest.fixture
def optimization_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "memory_manager": MemoryManager(device=device),
        "adaptive_processor": AdaptiveProcessor(device=device),
        "performance_monitor": PerformanceMonitor(
            target_latency=1.0,
            target_accuracy=0.95,
            target_memory_efficiency=0.8
        )
    }

@pytest.fixture
def benchmark_model():
    return BenchmarkProteinModel()

@pytest.fixture
def benchmark_data():
    """Generate benchmark data with various sizes."""
    return {
        "small": (torch.randn(16, 128, 256), torch.randint(0, 20, (16, 128))),
        "medium": (torch.randn(32, 256, 256), torch.randint(0, 20, (32, 256))),
        "large": (torch.randn(64, 512, 256), torch.randint(0, 20, (64, 512)))
    }

class BenchmarkResults:
    def __init__(self, save_dir="benchmark_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "latency": {},
            "throughput": {},
            "memory_usage": {},
            "accuracy": {}
        }

    def add_result(self, category, size, value):
        self.results[category][size] = value

    def save_results(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.save_dir / f"benchmark_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""

    def test_latency_benchmarks(
        self,
        optimization_pipeline,
        benchmark_model,
        benchmark_data
    ):
        """Benchmark latency across different input sizes."""
        results = BenchmarkResults()
        adaptive_processor = optimization_pipeline["adaptive_processor"]

        for size, (inputs, _) in benchmark_data.items():
            # Optimize model for current size
            optimized_model = adaptive_processor.optimize_for_hardware(
                benchmark_model,
                inputs.shape
            )

            # Measure latency
            latencies = []
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = optimized_model(inputs)

                # Benchmark
                for _ in range(100):
                    start_time = time.perf_counter()
                    _ = optimized_model(inputs)
                    latencies.append(time.perf_counter() - start_time)

            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)

            results.add_result("latency", size, {
                "average": avg_latency,
                "p95": p95_latency
            })

            # Assert performance targets
            assert avg_latency < 1.0, f"Average latency for {size} exceeds 1.0s"
            assert p95_latency < 2.0, f"P95 latency for {size} exceeds 2.0s"

        results.save_results()

    def test_throughput_benchmarks(
        self,
        optimization_pipeline,
        benchmark_model,
        benchmark_data
    ):
        """Benchmark throughput across different batch sizes."""
        results = BenchmarkResults()
        adaptive_processor = optimization_pipeline["adaptive_processor"]

        for size, (inputs, _) in benchmark_data.items():
            optimized_model = adaptive_processor.optimize_for_hardware(
                benchmark_model,
                inputs.shape
            )

            # Measure throughput
            batch_size = inputs.size(0)
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(50):  # Process 50 batches
                    _ = optimized_model(inputs)
            end_time = time.perf_counter()


            total_time = end_time - start_time
            samples_per_second = (50 * batch_size) / total_time

            results.add_result("throughput", size, {
                "samples_per_second": samples_per_second
            })

            # Assert minimum throughput
            min_throughput = 100  # Minimum 100 samples per second
            assert samples_per_second > min_throughput, \
                f"Throughput for {size} below {min_throughput} samples/s"

        results.save_results()

    def test_memory_efficiency_benchmarks(
        self,
        optimization_pipeline,
        benchmark_model,
        benchmark_data
    ):
        """Benchmark memory efficiency across different input sizes."""
        results = BenchmarkResults()
        memory_manager = optimization_pipeline["memory_manager"]

        for size, (inputs, _) in benchmark_data.items():
            # Measure memory usage
            initial_stats = memory_manager.get_memory_stats()

            # Run model with memory optimization
            optimized_model = memory_manager.optimize_for_inference(benchmark_model)
            _ = memory_manager.forward_with_memory_optimization(
                optimized_model,
                {"x": inputs},
                cache_key=f"benchmark_{size}"
            )

            final_stats = memory_manager.get_memory_stats()
            memory_efficiency = (
                final_stats["current_allocated"] /
                final_stats["peak_allocated"]
                if final_stats["peak_allocated"] > 0 else 1.0
            )

            results.add_result("memory_usage", size, {
                "efficiency": memory_efficiency,
                "peak_mb": final_stats["peak_allocated"] / (1024 * 1024),
                "current_mb": final_stats["current_allocated"] / (1024 * 1024)
            })

            # Assert memory efficiency
            assert memory_efficiency > 0.7, \
                f"Memory efficiency for {size} below 70%"

        results.save_results()

    def test_accuracy_benchmarks(
        self,
        optimization_pipeline,
        benchmark_model,
        benchmark_data
    ):
        """Benchmark prediction accuracy across different input sizes."""
        results = BenchmarkResults()
        performance_monitor = optimization_pipeline["performance_monitor"]

        criterion = nn.CrossEntropyLoss()
        for size, (inputs, targets) in benchmark_data.items():
            # Create dataloader for current size
            dataset = torch.utils.data.TensorDataset(inputs, targets)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

            # Measure accuracy
            metrics = performance_monitor.measure_accuracy(
                benchmark_model,
                dataloader,
                criterion
            )

            results.add_result("accuracy", size, {
                "accuracy": metrics["accuracy"],
                "validation_loss": metrics["validation_loss"]
            })

            # Assert minimum accuracy
            assert metrics["accuracy"] > 0.95, \
                f"Accuracy for {size} below 95%"

        results.save_results()

    @pytest.mark.parametrize("optimization_level", ["O1", "O2", "O3"])
    def test_optimization_levels(
        self,
        optimization_pipeline,
        benchmark_model,
        benchmark_data,
        optimization_level
    ):
        """Benchmark different optimization levels."""
        adaptive_processor = optimization_pipeline["adaptive_processor"]
        results = BenchmarkResults()

        for size, (inputs, _) in benchmark_data.items():
            # Configure optimization level
            adaptive_processor.optimization_level = optimization_level

            # Optimize and measure performance
            optimized_model = adaptive_processor.optimize_for_hardware(
                benchmark_model,
                inputs.shape
            )

            # Measure latency with current optimization level
            latencies = []
            with torch.no_grad():
                for _ in range(50):
                    start_time = time.perf_counter()
                    _ = optimized_model(inputs)
                    latencies.append(time.perf_counter() - start_time)

            results.add_result(f"optimization_{optimization_level}", size, {
                "average_latency": np.mean(latencies),
                "p95_latency": np.percentile(latencies, 95)
            })

        results.save_results()
