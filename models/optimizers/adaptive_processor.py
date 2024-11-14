import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from torch.cuda import amp
import torch.distributed as dist
import psutil
import time

class AdaptiveProcessor:
    """Hardware-aware adaptive processing for optimal performance."""

    def __init__(
        self,
        device: torch.device,
        target_latency: float = 1.0,  # Target latency in seconds
        enable_profiling: bool = True
    ):
        self.device = device
        self.target_latency = target_latency
        self.enable_profiling = enable_profiling

        # Hardware detection
        self.hardware_info = self._detect_hardware()

        # Performance tracking
        self.performance_metrics = {
            "inference_times": [],
            "throughput": [],
            "accuracy": [],
            "latency": []
        }

        # Optimization state
        self.current_precision = "fp32"
        self.current_batch_size = 1
        self.layer_configs = {}

    def _detect_hardware(self) -> Dict[str, Union[str, int]]:
        """Detect available hardware and capabilities."""
        info = {
            "device_type": "cpu",
            "compute_capability": None,
            "memory_bandwidth": None,
            "cores": psutil.cpu_count(logical=False)
        }

        if torch.cuda.is_available():
            info["device_type"] = "cuda"
            device_props = torch.cuda.get_device_properties(self.device)
            info["compute_capability"] = f"{device_props.major}.{device_props.minor}"
            info["memory_bandwidth"] = device_props.memory_clock_rate * device_props.memory_bus_width / 8
            info["cores"] = device_props.multi_processor_count

        return info

    def optimize_for_hardware(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        target_accuracy: float = 0.95
    ) -> nn.Module:
        """Optimize model for specific hardware."""
        device_type = self.hardware_info["device_type"]

        if device_type == "cuda":
            model = self._optimize_for_gpu(model, input_shape, target_accuracy)
        else:
            model = self._optimize_for_cpu(model, input_shape, target_accuracy)

        return model

    def _optimize_for_gpu(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        target_accuracy: float
    ) -> nn.Module:
        """GPU-specific optimizations."""
        # Enable mixed precision for newer GPUs
        if self.hardware_info["compute_capability"] >= "7.0":
            model = self._enable_mixed_precision(model)

        # Optimize memory access patterns
        model = self._optimize_memory_access(model)

        # Fuse operations where possible
        model = self._fuse_operations(model)

        # Apply tensor cores if available
        if self.hardware_info["compute_capability"] >= "7.0":
            model = self._enable_tensor_cores(model)

        return model

    def _optimize_for_cpu(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        target_accuracy: float
    ) -> nn.Module:
        """CPU-specific optimizations."""
        # Quantize model for CPU
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )

        # Enable vectorization
        model = self._enable_vectorization(model)

        # Optimize memory layout
        model = self._optimize_memory_layout(model)

        return model

    def profile_execution(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """Profile model execution performance."""
        model.eval()
        times = []

        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(sample_input)

            # Profile
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(sample_input)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

        # Calculate statistics
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        throughput = 1.0 / avg_time

        return {
            "average_latency": avg_time,
            "p95_latency": p95_time,
            "throughput": throughput
        }

    def adapt_processing(
        self,
        model: nn.Module,
        current_latency: float,
        current_accuracy: float
    ) -> Tuple[nn.Module, Dict[str, Union[str, float]]]:
        """Adapt processing based on performance metrics."""
        changes = {}

        # Adjust precision if needed
        if current_latency > self.target_latency and current_accuracy > 0.95:
            if self.current_precision == "fp32":
                model = self._enable_mixed_precision(model)
                self.current_precision = "fp16"
                changes["precision"] = "fp16"

        # Adjust batch size
        if current_latency > self.target_latency:
            self.current_batch_size = max(1, self.current_batch_size // 2)
            changes["batch_size"] = self.current_batch_size
        elif current_latency < self.target_latency * 0.5:
            self.current_batch_size *= 2
            changes["batch_size"] = self.current_batch_size

        # Layer-wise optimization
        model = self._optimize_layer_configuration(model, current_latency)

        return model, changes

    def _enable_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Enable mixed precision training/inference."""
        model = model.to(dtype=torch.float16)
        return model

    def _optimize_memory_access(self, model: nn.Module) -> nn.Module:
        """Optimize memory access patterns."""
        # Implement memory access optimization
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.contiguous()
        return model

    def _fuse_operations(self, model: nn.Module) -> nn.Module:
        """Fuse operations for better performance."""
        model = torch.jit.freeze(torch.jit.script(model))
        return model

    def _enable_tensor_cores(self, model: nn.Module) -> nn.Module:
        """Enable tensor cores for compatible operations."""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.data = module.weight.data.to(dtype=torch.float16)
        return model

    def _enable_vectorization(self, model: nn.Module) -> nn.Module:
        """Enable vectorized operations for CPU."""
        torch.backends.mkldnn.enabled = True
        return model

    def _optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """Optimize memory layout for CPU execution."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.contiguous(memory_format=torch.contiguous_format)
        return model

    def _optimize_layer_configuration(
        self,
        model: nn.Module,
        current_latency: float
    ) -> nn.Module:
        """Optimize layer-wise configuration."""
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                # Adjust attention heads based on latency
                if current_latency > self.target_latency:
                    module.self_attn.num_heads = max(1, module.self_attn.num_heads // 2)
                self.layer_configs[name] = {"num_heads": module.self_attn.num_heads}

        return model

    def get_performance_metrics(self) -> Dict[str, List[float]]:
        """Get current performance metrics."""
        return self.performance_metrics

    def update_metrics(
        self,
        inference_time: float,
        accuracy: float,
        batch_size: int
    ) -> None:
        """Update performance metrics."""
        self.performance_metrics["inference_times"].append(inference_time)
        self.performance_metrics["accuracy"].append(accuracy)
        self.performance_metrics["latency"].append(inference_time / batch_size)
        self.performance_metrics["throughput"].append(batch_size / inference_time)
