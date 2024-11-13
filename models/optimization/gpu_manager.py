"""GPU resource management for protein analysis"""
import torch
import logging
from typing import Optional, Dict, Any
import os

logger = logging.getLogger(__name__)

class GPUManager:
    """Manage GPU resources for protein analysis computations"""

    def __init__(self):
        self.device = self._get_device()
        self.memory_allocated = 0
        self.max_memory = self._get_max_memory()

    def _get_device(self) -> torch.device:
        """Get available device (GPU/CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def _get_max_memory(self) -> int:
        """Get maximum available GPU memory"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 0

    def allocate_memory(self, size_mb: int) -> bool:
        """Attempt to allocate GPU memory"""
        if self.memory_allocated + size_mb * 1024 * 1024 <= self.max_memory:
            self.memory_allocated += size_mb * 1024 * 1024
            return True
        return False

    def free_memory(self, size_mb: int):
        """Free allocated GPU memory"""
        self.memory_allocated = max(0, self.memory_allocated - size_mb * 1024 * 1024)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status"""
        status = {
            "device": str(self.device),
            "total_memory_mb": self.max_memory // (1024 * 1024),
            "allocated_memory_mb": self.memory_allocated // (1024 * 1024),
            "available_memory_mb": (self.max_memory - self.memory_allocated) // (1024 * 1024)
        }
        return status

    def optimize_batch_size(self, min_batch: int, max_batch: int,
                          sample_input_size: int) -> int:
        """Optimize batch size based on available memory"""
        if not torch.cuda.is_available():
            return min_batch

        for batch_size in range(max_batch, min_batch - 1, -1):
            try:
                # Test memory allocation
                sample = torch.randn(batch_size, sample_input_size,
                                   device=self.device)
                del sample
                torch.cuda.empty_cache()
                return batch_size
            except RuntimeError:
                continue

        return min_batch
