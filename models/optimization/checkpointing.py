"""Checkpoint management for protein analysis"""
import os
import json
import pickle
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manage checkpoints for long-running computations"""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.current_checkpoint = None

    def save_checkpoint(self, state: Dict[str, Any], name: Optional[str] = None):
        """Save computation state checkpoint"""
        try:
            if name is None:
                name = datetime.now().strftime("%Y%m%d_%H%M%S")

            checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.ckpt")

            # Save state dictionary
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)

            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'name': name,
                'size': os.path.getsize(checkpoint_path)
            }

            metadata_path = os.path.join(self.checkpoint_dir, f"{name}.meta")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.current_checkpoint = name
            logger.info(f"Saved checkpoint: {name}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise

    def load_checkpoint(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint state"""
        try:
            if name is None:
                name = self._get_latest_checkpoint()

            if name is None:
                raise ValueError("No checkpoints found")

            checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.ckpt")

            with open(checkpoint_path, 'rb') as f:
                state = pickle.load(f)

            self.current_checkpoint = name
            logger.info(f"Loaded checkpoint: {name}")
            return state

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []

        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.meta'):
                with open(os.path.join(self.checkpoint_dir, file), 'r') as f:
                    metadata = json.load(f)
                    checkpoints.append(metadata)

        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get name of latest checkpoint"""
        checkpoints = self.list_checkpoints()
        return checkpoints[0]['name'] if checkpoints else None

    def remove_checkpoint(self, name: str):
        """Remove a checkpoint"""
        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.ckpt")
            metadata_path = os.path.join(self.checkpoint_dir, f"{name}.meta")

            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            if self.current_checkpoint == name:
                self.current_checkpoint = None

            logger.info(f"Removed checkpoint: {name}")

        except Exception as e:
            logger.error(f"Error removing checkpoint: {e}")
            raise
