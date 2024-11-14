import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from torch.cuda import amp
import torch.distributed as dist
import psutil

class MemoryManager:
    """Advanced memory management for efficient model execution."""

    def __init__(
        self,
        device: torch.device,
        max_memory: Optional[int] = None,
        enable_amp: bool = True,
        enable_checkpoint: bool = True
    ):
        self.device = device
        self.max_memory = max_memory or self._get_available_memory()
        self.enable_amp = enable_amp
        self.enable_checkpoint = enable_checkpoint

        # Initialize memory pools
        self.attention_cache = {}
        self.gradient_cache = {}
        self.activation_cache = {}

        # Setup mixed precision training
        self.scaler = amp.GradScaler(enabled=enable_amp)

        # Initialize memory tracking
        self.memory_stats = {
            "peak_allocated": 0,
            "current_allocated": 0,
            "cached_memory": 0
        }

    def _get_available_memory(self) -> int:
        """Get available GPU memory."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(self.device).total_memory
        return psutil.virtual_memory().available

    def optimize_memory_allocation(
        self,
        model: nn.Module,
        batch_size: int,
        sequence_length: int
    ) -> Dict[str, int]:
        """Optimize memory allocation based on model and input size."""
        # Calculate memory requirements
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        input_memory = batch_size * sequence_length * model.hidden_size * 4  # Float32

        # Calculate attention cache size
        num_layers = len([m for m in model.modules() if isinstance(m, nn.TransformerEncoderLayer)])
        attention_memory = batch_size * num_layers * sequence_length * sequence_length * 4

        # Adjust batch size if needed
        if (param_memory + input_memory + attention_memory) > self.max_memory:
            new_batch_size = self._calculate_optimal_batch_size(
                param_memory, input_memory, attention_memory, batch_size
            )
            return {"optimal_batch_size": new_batch_size}

        return {"optimal_batch_size": batch_size}

    def _calculate_optimal_batch_size(
        self,
        param_memory: int,
        input_memory_per_batch: int,
        attention_memory_per_batch: int,
        current_batch_size: int
    ) -> int:
        """Calculate optimal batch size based on memory constraints."""
        total_memory_per_batch = input_memory_per_batch + attention_memory_per_batch
        available_memory = self.max_memory - param_memory
        optimal_batch_size = max(1, int(available_memory / total_memory_per_batch))
        return min(optimal_batch_size, current_batch_size)

    @torch.cuda.amp.autocast()
    def forward_with_memory_optimization(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        cache_key: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with memory optimization."""
        # Check cache
        if cache_key and cache_key in self.attention_cache:
            cached_attention = self.attention_cache[cache_key]
        else:
            cached_attention = None

        # Run forward pass with automatic mixed precision
        with amp.autocast(enabled=self.enable_amp):
            outputs = model(
                **inputs,
                cached_attention=cached_attention
            )

        # Update cache
        if cache_key:
            self.attention_cache[cache_key] = outputs.get("attention_cache", None)

        # Update memory stats
        self._update_memory_stats()

        return outputs

    def checkpoint_sequential(
        self,
        functions: List[nn.Module],
        segments: int,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Gradient checkpointing for memory efficiency."""
        if not self.enable_checkpoint:
            return nn.Sequential(*functions)(input_tensor)

        # Split into segments for checkpointing
        segment_size = len(functions) // segments
        segment_functions = [
            nn.Sequential(*functions[i:i + segment_size])
            for i in range(0, len(functions), segment_size)
        ]

        # Apply checkpointing
        def create_checkpoint_function(function):
            def forward(*inputs):
                return function(*inputs)
            return forward

        result = input_tensor
        for function in segment_functions:
            result = torch.utils.checkpoint.checkpoint(
                create_checkpoint_function(function),
                result
            )

        return result

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """Clear specified or all cache types."""
        if cache_type == "attention" or cache_type is None:
            self.attention_cache.clear()
        if cache_type == "gradient" or cache_type is None:
            self.gradient_cache.clear()
        if cache_type == "activation" or cache_type is None:
            self.activation_cache.clear()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _update_memory_stats(self) -> None:
        """Update memory usage statistics."""
        if torch.cuda.is_available():
            self.memory_stats.update({
                "peak_allocated": torch.cuda.max_memory_allocated(self.device),
                "current_allocated": torch.cuda.memory_allocated(self.device),
                "cached_memory": torch.cuda.memory_reserved(self.device)
            })

    def optimize_for_inference(
        self,
        model: nn.Module,
        quantize: bool = True,
        optimize_graph: bool = True
    ) -> nn.Module:
        """Optimize model for inference."""
        if quantize:
            # Quantize model to int8
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

        if optimize_graph and torch.jit.is_tracing():
            # Optimize computation graph
            model = torch.jit.script(model)
            model = torch.jit.freeze(model)

        return model

    def get_memory_stats(self) -> Dict[str, int]:
        """Get current memory statistics."""
        self._update_memory_stats()
        return self.memory_stats

    def adaptive_batch_size(
        self,
        model: nn.Module,
        initial_batch_size: int,
        sequence_length: int,
        target_memory_usage: float = 0.8
    ) -> int:
        """Dynamically adjust batch size based on memory usage."""
        stats = self.get_memory_stats()
        current_usage = stats["current_allocated"] / self.max_memory

        if current_usage > target_memory_usage:
            # Decrease batch size
            return max(1, int(initial_batch_size * (target_memory_usage / current_usage)))
        elif current_usage < target_memory_usage * 0.7:
            # Increase batch size if memory usage is significantly below target
            return min(
                initial_batch_size * 2,
                int(initial_batch_size * (target_memory_usage / current_usage))
            )

        return initial_batch_size
