import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import time
from dataclasses import dataclass
import json
import logging
from pathlib import Path

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    inference_time: float
    accuracy: float
    memory_usage: float
    throughput: float
    latency_p95: float

class PerformanceMonitor:
    """Monitor and validate model performance metrics."""

    def __init__(
        self,
        target_latency: float = 1.0,  # seconds
        target_accuracy: float = 0.95,
        target_memory_efficiency: float = 0.8,
        log_dir: Optional[str] = None
    ):
        self.target_latency = target_latency
        self.target_accuracy = target_accuracy
        self.target_memory_efficiency = target_memory_efficiency

        # Initialize metrics storage
        self.metrics_history = {
            "inference_times": [],
            "accuracies": [],
            "memory_usage": [],
            "throughput": [],
            "latency_p95": []
        }

        # Setup logging
        self.log_dir = Path(log_dir) if log_dir else Path("performance_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for performance monitoring."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "performance.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("PerformanceMonitor")

    def measure_inference_time(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Measure model inference time statistics."""
        model.eval()
        times = []

        with torch.no_grad():
            # Warmup runs
            for _ in range(warmup_runs):
                _ = model(input_data)

            # Actual measurements
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(input_data)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        throughput = 1.0 / avg_time

        metrics = {
            "average_inference_time": avg_time,
            "p95_latency": p95_time,
            "throughput": throughput
        }

        self._log_metrics("inference", metrics)
        return metrics

    def measure_accuracy(
        self,
        model: nn.Module,
        validation_data: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Measure model accuracy on validation data."""
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in validation_data:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == targets).sum().item()
                total_samples += targets.size(0)

        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(validation_data)

        metrics = {
            "accuracy": accuracy,
            "validation_loss": avg_loss
        }

        self._log_metrics("accuracy", metrics)
        return metrics

    def measure_memory_usage(
        self,
        model: nn.Module,
        sample_input: torch.Tensor
    ) -> Dict[str, float]:
        """Measure model memory usage statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

            # Run inference to measure memory
            with torch.no_grad():
                _ = model(sample_input)

            peak_memory = torch.cuda.max_memory_allocated()
            current_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()

            metrics = {
                "peak_memory_mb": peak_memory / 1024**2,
                "current_memory_mb": current_memory / 1024**2,
                "reserved_memory_mb": reserved_memory / 1024**2,
                "memory_efficiency": current_memory / reserved_memory if reserved_memory > 0 else 1.0
            }
        else:
            import psutil
            process = psutil.Process()
            metrics = {
                "memory_usage_mb": process.memory_info().rss / 1024**2
            }

        self._log_metrics("memory", metrics)
        return metrics

    def validate_performance(
        self,
        model: nn.Module,
        validation_data: torch.utils.data.DataLoader,
        criterion: nn.Module,
        sample_input: torch.Tensor
    ) -> Tuple[bool, Dict[str, float]]:
        """Validate if model meets performance targets."""
        # Measure all metrics
        inference_metrics = self.measure_inference_time(model, sample_input)
        accuracy_metrics = self.measure_accuracy(model, validation_data, criterion)
        memory_metrics = self.measure_memory_usage(model, sample_input)

        # Combine metrics
        performance_metrics = PerformanceMetrics(
            inference_time=inference_metrics["average_inference_time"],
            accuracy=accuracy_metrics["accuracy"],
            memory_usage=memory_metrics.get("memory_efficiency", 1.0),
            throughput=inference_metrics["throughput"],
            latency_p95=inference_metrics["p95_latency"]
        )

        # Update history
        self._update_metrics_history(performance_metrics)

        # Check if meets targets
        meets_targets = (
            performance_metrics.inference_time < self.target_latency and
            performance_metrics.accuracy > self.target_accuracy and
            performance_metrics.memory_usage > self.target_memory_efficiency
        )

        # Log validation results
        self._log_validation_results(meets_targets, performance_metrics)

        return meets_targets, self._get_current_metrics()

    def _update_metrics_history(self, metrics: PerformanceMetrics) -> None:
        """Update metrics history."""
        self.metrics_history["inference_times"].append(metrics.inference_time)
        self.metrics_history["accuracies"].append(metrics.accuracy)
        self.metrics_history["memory_usage"].append(metrics.memory_usage)
        self.metrics_history["throughput"].append(metrics.throughput)
        self.metrics_history["latency_p95"].append(metrics.latency_p95)

    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            "inference_time": self.metrics_history["inference_times"][-1],
            "accuracy": self.metrics_history["accuracies"][-1],
            "memory_usage": self.metrics_history["memory_usage"][-1],
            "throughput": self.metrics_history["throughput"][-1],
            "latency_p95": self.metrics_history["latency_p95"][-1]
        }

    def _log_metrics(self, metric_type: str, metrics: Dict[str, float]) -> None:
        """Log performance metrics."""
        self.logger.info(f"{metric_type.capitalize()} metrics: {json.dumps(metrics, indent=2)}")

    def _log_validation_results(
        self,
        meets_targets: bool,
        metrics: PerformanceMetrics
    ) -> None:
        """Log validation results."""
        status = "PASSED" if meets_targets else "FAILED"
        self.logger.info(f"Performance validation {status}")
        self.logger.info(f"Metrics: {json.dumps(metrics.__dict__, indent=2)}")

    def save_metrics_history(self, filename: Optional[str] = None) -> None:
        """Save metrics history to file."""
        if filename is None:
            filename = f"metrics_history_{time.strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        self.logger.info(f"Saved metrics history to {filepath}")

    def get_optimization_suggestions(self) -> List[str]:
        """Get suggestions for performance optimization."""
        suggestions = []
        current_metrics = self._get_current_metrics()

        if current_metrics["inference_time"] >= self.target_latency:
            suggestions.append(
                "Consider enabling mixed precision training or reducing model size"
            )

        if current_metrics["accuracy"] <= self.target_accuracy:
            suggestions.append(
                "Consider increasing model capacity or adjusting learning rate"
            )

        if current_metrics["memory_usage"] <= self.target_memory_efficiency:
            suggestions.append(
                "Consider implementing gradient checkpointing or reducing batch size"
            )

        return suggestions
