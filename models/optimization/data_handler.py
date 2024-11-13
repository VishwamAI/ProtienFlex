"""Efficient data handling for protein analysis"""
import numpy as np
from typing import Dict, List, Any, Optional
import h5py
import logging
import json
import os

logger = logging.getLogger(__name__)

class DataHandler:
    """Handle data efficiently for protein analysis"""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def save_trajectory(self, traj_data: np.ndarray, metadata: Dict[str, Any],
                       filename: str):
        """Save trajectory data with compression"""
        try:
            filepath = os.path.join(self.cache_dir, filename)
            with h5py.File(filepath, 'w') as f:
                # Create compressed dataset
                f.create_dataset('trajectory', data=traj_data,
                               compression='gzip', compression_opts=9)
                # Store metadata
                f.attrs['metadata'] = json.dumps(metadata)
            logger.info(f"Saved trajectory to {filepath}")
        except Exception as e:
            logger.error(f"Error saving trajectory: {e}")
            raise

    def load_trajectory(self, filename: str) -> tuple[np.ndarray, Dict[str, Any]]:
        """Load trajectory data"""
        try:
            filepath = os.path.join(self.cache_dir, filename)
            with h5py.File(filepath, 'r') as f:
                traj_data = f['trajectory'][:]
                metadata = json.loads(f.attrs['metadata'])
            return traj_data, metadata
        except Exception as e:
            logger.error(f"Error loading trajectory: {e}")
            raise

    def clear_cache(self):
        """Clear cached data"""
        try:
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise

    def get_cache_size(self) -> int:
        """Get total size of cached data in bytes"""
        total_size = 0
        for file in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, file)
            total_size += os.path.getsize(file_path)
        return total_size
