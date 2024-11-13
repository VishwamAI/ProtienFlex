"""
Data Handler Module

Manages efficient data transfer, caching, and memory optimization between
pipeline components for protein structure and dynamics analysis.
"""

import os
import logging
import tempfile
import shutil
from typing import Dict, Any, Optional, Union, List
import numpy as np
import h5py
import pickle
from pathlib import Path
import json
import hashlib
from datetime import datetime

class DataHandler:
    """Handles efficient data management between pipeline components."""

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 max_cache_size: float = 100.0,  # GB
                 enable_compression: bool = True):
        """Initialize data handler.

        Args:
            cache_dir: Directory for caching data
            max_cache_size: Maximum cache size in GB
            enable_compression: Whether to enable data compression
        """
        self.cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(),
            'proteinflex_cache'
        )
        self.max_cache_size = max_cache_size * (1024**3)  # Convert to bytes
        self.enable_compression = enable_compression
        self.logger = logging.getLogger(__name__)

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize cache tracking
        self._init_cache_tracking()

    def _init_cache_tracking(self):
        """Initialize cache tracking system."""
        self.cache_index_file = os.path.join(self.cache_dir, 'cache_index.json')
        if os.path.exists(self.cache_index_file):
            with open(self.cache_index_file, 'r') as f:
                self.cache_index = json.load(f)
        else:
            self.cache_index = {
                'entries': {},
                'total_size': 0
            }

    def _generate_cache_key(self, data_id: str, metadata: Dict) -> str:
        """Generate unique cache key based on data ID and metadata."""
        # Create string representation of metadata
        meta_str = json.dumps(metadata, sort_keys=True)
        # Combine with data_id and hash
        combined = f"{data_id}_{meta_str}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def store_structure(self,
                       structure_data: Dict[str, Any],
                       data_id: str,
                       metadata: Optional[Dict] = None) -> str:
        """Store structure data efficiently.

        Args:
            structure_data: Dictionary containing structure information
            data_id: Unique identifier for the data
            metadata: Optional metadata for caching

        Returns:
            Cache key for stored data
        """
        metadata = metadata or {}
        cache_key = self._generate_cache_key(data_id, metadata)
        file_path = os.path.join(self.cache_dir, f"{cache_key}_structure.h5")

        try:
            with h5py.File(file_path, 'w') as f:
                # Store atomic positions
                if 'positions' in structure_data:
                    if self.enable_compression:
                        f.create_dataset('positions',
                                       data=structure_data['positions'],
                                       compression='gzip',
                                       compression_opts=4)
                    else:
                        f.create_dataset('positions',
                                       data=structure_data['positions'])

                # Store confidence metrics
                if 'plddt' in structure_data:
                    f.create_dataset('plddt', data=structure_data['plddt'])
                if 'pae' in structure_data:
                    f.create_dataset('pae', data=structure_data['pae'])

                # Store metadata
                f.attrs['data_id'] = data_id
                f.attrs['timestamp'] = datetime.now().isoformat()
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        f.attrs[key] = value

            # Update cache index
            file_size = os.path.getsize(file_path)
            self._update_cache_index(cache_key, file_path, file_size)

            return cache_key

        except Exception as e:
            self.logger.error(f"Error storing structure data: {str(e)}")
            raise

    def store_trajectory(self,
                        trajectory_data: Dict[str, Any],
                        data_id: str,
                        metadata: Optional[Dict] = None) -> str:
        """Store trajectory data efficiently.

        Args:
            trajectory_data: Dictionary containing trajectory information
            data_id: Unique identifier for the data
            metadata: Optional metadata for caching

        Returns:
            Cache key for stored data
        """
        metadata = metadata or {}
        cache_key = self._generate_cache_key(data_id, metadata)
        file_path = os.path.join(self.cache_dir, f"{cache_key}_trajectory.h5")

        try:
            with h5py.File(file_path, 'w') as f:
                # Store trajectory frames
                if 'frames' in trajectory_data:
                    if self.enable_compression:
                        f.create_dataset('frames',
                                       data=trajectory_data['frames'],
                                       compression='gzip',
                                       compression_opts=4)
                    else:
                        f.create_dataset('frames',
                                       data=trajectory_data['frames'])

                # Store additional trajectory data
                for key, value in trajectory_data.items():
                    if key != 'frames' and isinstance(value, np.ndarray):
                        f.create_dataset(key, data=value)

                # Store metadata
                f.attrs['data_id'] = data_id
                f.attrs['timestamp'] = datetime.now().isoformat()
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        f.attrs[key] = value

            # Update cache index
            file_size = os.path.getsize(file_path)
            self._update_cache_index(cache_key, file_path, file_size)

            return cache_key

        except Exception as e:
            self.logger.error(f"Error storing trajectory data: {str(e)}")
            raise

    def load_data(self, cache_key: str) -> Dict[str, Any]:
        """Load data from cache.

        Args:
            cache_key: Cache key for stored data

        Returns:
            Dictionary containing stored data
        """
        if cache_key not in self.cache_index['entries']:
            raise KeyError(f"Cache key {cache_key} not found")

        entry = self.cache_index['entries'][cache_key]
        file_path = entry['file_path']

        try:
            with h5py.File(file_path, 'r') as f:
                # Load all datasets
                data = {}
                for key in f.keys():
                    data[key] = f[key][:]

                # Load metadata from attributes
                metadata = dict(f.attrs)
                data['metadata'] = metadata

            return data

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def _update_cache_index(self,
                           cache_key: str,
                           file_path: str,
                           file_size: int):
        """Update cache index with new entry."""
        # Add new entry
        self.cache_index['entries'][cache_key] = {
            'file_path': file_path,
            'size': file_size,
            'timestamp': datetime.now().isoformat()
        }
        self.cache_index['total_size'] += file_size

        # Check cache size and cleanup if necessary
        self._cleanup_cache_if_needed()

        # Save updated index
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f)

    def _cleanup_cache_if_needed(self):
        """Clean up oldest cache entries if size limit exceeded."""
        while self.cache_index['total_size'] > self.max_cache_size:
            # Find oldest entry
            oldest_key = min(
                self.cache_index['entries'],
                key=lambda k: self.cache_index['entries'][k]['timestamp']
            )

            # Remove file and update index
            entry = self.cache_index['entries'][oldest_key]
            try:
                os.remove(entry['file_path'])
                self.cache_index['total_size'] -= entry['size']
                del self.cache_index['entries'][oldest_key]
            except Exception as e:
                self.logger.error(f"Error cleaning up cache: {str(e)}")

    def clear_cache(self):
        """Clear all cached data."""
        try:
            # Remove all files
            for entry in self.cache_index['entries'].values():
                try:
                    os.remove(entry['file_path'])
                except Exception as e:
                    self.logger.error(f"Error removing file: {str(e)}")

            # Reset cache index
            self.cache_index = {
                'entries': {},
                'total_size': 0
            }

            # Save empty index
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f)

        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            raise

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        return {
            'total_size_gb': self.cache_index['total_size'] / (1024**3),
            'max_size_gb': self.max_cache_size / (1024**3),
            'num_entries': len(self.cache_index['entries']),
            'usage_percent': (self.cache_index['total_size'] / self.max_cache_size) * 100
        }

    def optimize_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory usage of data structures.

        Args:
            data: Dictionary containing data to optimize

        Returns:
            Optimized data dictionary
        """
        optimized = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Convert to float32 for better memory usage
                if value.dtype in [np.float64, np.float128]:
                    optimized[key] = value.astype(np.float32)
                else:
                    optimized[key] = value
            else:
                optimized[key] = value
        return optimized
