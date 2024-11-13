"""
GPU Manager Module

Handles GPU resource allocation, optimization, and multi-GPU support for
protein structure prediction and molecular dynamics simulations.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import tensorflow as tf
import jax
import jax.numpy as jnp

class GPUManager:
    """Manages GPU resources for optimal performance."""

    def __init__(self,
                 required_memory: Dict[str, int] = None,
                 prefer_single_gpu: bool = False):
        """Initialize GPU manager.

        Args:
            required_memory: Dictionary of memory requirements per component
                           (e.g., {'prediction': 16, 'dynamics': 8} in GB)
            prefer_single_gpu: If True, prefer using a single GPU even when
                             multiple are available
        """
        self.required_memory = required_memory or {
            'prediction': 16,  # AlphaFold3 typically needs ~16GB
            'dynamics': 8      # Molecular dynamics typically needs ~8GB
        }
        self.prefer_single_gpu = prefer_single_gpu
        self.logger = logging.getLogger(__name__)

        # Initialize frameworks
        self._setup_frameworks()

    def _setup_frameworks(self):
        """Setup ML frameworks for GPU usage."""
        # TensorFlow setup
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                self.logger.warning(f"Memory growth setup failed: {str(e)}")

        # JAX setup
        if len(jax.devices('gpu')) > 0:
            # Enable 32-bit matrix multiplication
            jax.config.update('jax_enable_x64', True)

        # PyTorch setup
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def get_available_gpus(self) -> List[Dict[str, any]]:
        """Get list of available GPUs with their properties."""
        available_gpus = []

        # Check PyTorch GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_props = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'memory_free': self._get_gpu_free_memory(i),
                    'framework': 'pytorch'
                }
                available_gpus.append(gpu_props)

        return available_gpus

    def _get_gpu_free_memory(self, device_index: int) -> float:
        """Get free memory for given GPU in GB."""
        try:
            torch.cuda.set_device(device_index)
            free_memory = torch.cuda.memory_reserved(device_index) - torch.cuda.memory_allocated(device_index)
            return free_memory / (1024**3)
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory: {str(e)}")
            return 0.0

    def allocate_gpus(self, task: str) -> List[int]:
        """Allocate GPUs for specific task based on requirements.

        Args:
            task: Task type ('prediction' or 'dynamics')

        Returns:
            List of GPU indices to use
        """
        available_gpus = self.get_available_gpus()
        required_memory = self.required_memory.get(task, 0)

        if not available_gpus:
            self.logger.warning("No GPUs available, falling back to CPU")
            return []

        # Filter GPUs with sufficient memory
        suitable_gpus = [
            gpu for gpu in available_gpus
            if gpu['memory_free'] >= required_memory
        ]

        if not suitable_gpus:
            self.logger.warning(
                f"No GPUs with sufficient memory ({required_memory}GB) found"
            )
            return []

        # If preferring single GPU, return the one with most free memory
        if self.prefer_single_gpu:
            best_gpu = max(suitable_gpus, key=lambda x: x['memory_free'])
            return [best_gpu['index']]

        # Otherwise, return all suitable GPUs
        return [gpu['index'] for gpu in suitable_gpus]

    def optimize_memory_usage(self, task: str, gpu_indices: List[int]):
        """Optimize memory usage for given task and GPUs.

        Args:
            task: Task type ('prediction' or 'dynamics')
            gpu_indices: List of GPU indices to optimize
        """
        if not gpu_indices:
            return

        if task == 'prediction':
            self._optimize_prediction_memory(gpu_indices)
        elif task == 'dynamics':
            self._optimize_dynamics_memory(gpu_indices)

    def _optimize_prediction_memory(self, gpu_indices: List[int]):
        """Optimize memory usage for prediction task."""
        # Set PyTorch to use GPU(s)
        if torch.cuda.is_available():
            if len(gpu_indices) == 1:
                torch.cuda.set_device(gpu_indices[0])
            else:
                # Setup for multi-GPU
                torch.cuda.set_device(gpu_indices[0])
                # Enable gradient checkpointing for memory efficiency
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

        # JAX optimization
        if len(jax.devices('gpu')) > 0:
            # Enable memory defragmentation
            jax.config.update('jax_enable_x64', True)
            # Set default device
            jax.config.update('jax_platform_name', 'gpu')

    def _optimize_dynamics_memory(self, gpu_indices: List[int]):
        """Optimize memory usage for dynamics task."""
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_indices[0])
            # Use mixed precision for dynamics
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def monitor_memory_usage(self, gpu_indices: List[int]) -> Dict[int, Dict]:
        """Monitor memory usage of specified GPUs.

        Args:
            gpu_indices: List of GPU indices to monitor

        Returns:
            Dictionary of memory statistics per GPU
        """
        memory_stats = {}
        for idx in gpu_indices:
            try:
                torch.cuda.set_device(idx)
                stats = {
                    'total': torch.cuda.get_device_properties(idx).total_memory / (1024**3),
                    'reserved': torch.cuda.memory_reserved(idx) / (1024**3),
                    'allocated': torch.cuda.memory_allocated(idx) / (1024**3),
                    'free': self._get_gpu_free_memory(idx)
                }
                memory_stats[idx] = stats
            except Exception as e:
                self.logger.error(f"Error monitoring GPU {idx}: {str(e)}")
                memory_stats[idx] = {'error': str(e)}

        return memory_stats

    def cleanup(self, gpu_indices: List[int]):
        """Clean up GPU memory after task completion.

        Args:
            gpu_indices: List of GPU indices to clean up
        """
        if torch.cuda.is_available():
            try:
                for idx in gpu_indices:
                    torch.cuda.set_device(idx)
                    torch.cuda.empty_cache()
            except Exception as e:
                self.logger.error(f"Error cleaning up GPU memory: {str(e)}")

        # Force garbage collection
        import gc
        gc.collect()

    def get_optimal_batch_size(self, task: str, gpu_indices: List[int]) -> int:
        """Calculate optimal batch size based on available GPU memory.

        Args:
            task: Task type ('prediction' or 'dynamics')
            gpu_indices: List of GPU indices to use

        Returns:
            Optimal batch size
        """
        if not gpu_indices:
            return 1

        # Get minimum free memory across GPUs
        free_memory = min(
            self._get_gpu_free_memory(idx)
            for idx in gpu_indices
        )

        # Calculate batch size based on task requirements
        if task == 'prediction':
            # AlphaFold3 typically needs ~16GB per protein
            memory_per_item = self.required_memory['prediction']
            # Leave 10% memory as buffer
            usable_memory = free_memory * 0.9
            batch_size = max(1, int(usable_memory / memory_per_item))
        else:  # dynamics
            # Molecular dynamics typically needs ~8GB per simulation
            memory_per_item = self.required_memory['dynamics']
            # Leave 20% memory as buffer for dynamics
            usable_memory = free_memory * 0.8
            batch_size = max(1, int(usable_memory / memory_per_item))

        return batch_size

    def get_device_mapping(self, task: str) -> Dict[str, str]:
        """Get framework-specific device mapping for task.

        Args:
            task: Task type ('prediction' or 'dynamics')

        Returns:
            Dictionary of framework-specific device strings
        """
        gpu_indices = self.allocate_gpus(task)
        if not gpu_indices:
            return {
                'pytorch': 'cpu',
                'tensorflow': '/CPU:0',
                'jax': 'cpu'
            }

        primary_gpu = gpu_indices[0]
        return {
            'pytorch': f'cuda:{primary_gpu}',
            'tensorflow': f'/GPU:{primary_gpu}',
            'jax': f'gpu:{primary_gpu}'
        }
