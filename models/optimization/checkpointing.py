"""
Checkpoint Manager Module

Coordinates checkpointing across different components of the protein analysis pipeline,
ensuring consistent state saving and recovery.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import shutil
from pathlib import Path
import threading
import time
import hashlib

class CheckpointManager:
    """Manages checkpointing across pipeline components."""

    def __init__(self,
                 base_dir: str,
                 max_checkpoints: int = 5,
                 auto_cleanup: bool = True):
        """Initialize checkpoint manager.

        Args:
            base_dir: Base directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            auto_cleanup: Whether to automatically clean up old checkpoints
        """
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        self.logger = logging.getLogger(__name__)

        # Create checkpoint directory structure
        self._init_directories()

        # Lock for thread safety
        self._lock = threading.Lock()

    def _init_directories(self):
        """Initialize checkpoint directory structure."""
        # Main checkpoint directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.structure_dir = self.base_dir / 'structure'
        self.dynamics_dir = self.base_dir / 'dynamics'
        self.analysis_dir = self.base_dir / 'analysis'
        self.progress_dir = self.base_dir / 'progress'

        # Create subdirectories
        for directory in [self.structure_dir, self.dynamics_dir,
                         self.analysis_dir, self.progress_dir]:
            directory.mkdir(exist_ok=True)

    def create_checkpoint(self,
                         checkpoint_id: str,
                         component_states: Dict[str, Any]) -> str:
        """Create a new checkpoint.

        Args:
            checkpoint_id: Unique identifier for checkpoint
            component_states: Dictionary of component states to save

        Returns:
            Path to checkpoint directory
        """
        with self._lock:
            # Create checkpoint directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_dir = self.base_dir / f"checkpoint_{checkpoint_id}_{timestamp}"
            checkpoint_dir.mkdir(exist_ok=True)

            try:
                # Save component states
                self._save_component_states(checkpoint_dir, component_states)

                # Save checkpoint metadata
                self._save_metadata(checkpoint_dir, checkpoint_id, component_states)

                # Cleanup old checkpoints if needed
                if self.auto_cleanup:
                    self._cleanup_old_checkpoints()

                return str(checkpoint_dir)

            except Exception as e:
                self.logger.error(f"Failed to create checkpoint: {str(e)}")
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                raise

    def _save_component_states(self,
                             checkpoint_dir: Path,
                             component_states: Dict[str, Any]):
        """Save individual component states."""
        for component, state in component_states.items():
            component_dir = checkpoint_dir / component
            component_dir.mkdir(exist_ok=True)

            if component == 'structure':
                self._save_structure_state(component_dir, state)
            elif component == 'dynamics':
                self._save_dynamics_state(component_dir, state)
            elif component == 'analysis':
                self._save_analysis_state(component_dir, state)
            elif component == 'progress':
                self._save_progress_state(component_dir, state)

    def _save_structure_state(self, directory: Path, state: Dict):
        """Save structure prediction state."""
        with open(directory / 'structure_state.json', 'w') as f:
            json.dump({
                k: v for k, v in state.items()
                if isinstance(v, (dict, list, str, int, float, bool))
            }, f)

        # Save numpy arrays separately
        for key, value in state.items():
            if hasattr(value, 'numpy'):  # PyTorch tensors
                np.save(directory / f'{key}.npy', value.numpy())
            elif isinstance(value, np.ndarray):
                np.save(directory / f'{key}.npy', value)

    def _save_dynamics_state(self, directory: Path, state: Dict):
        """Save molecular dynamics state."""
        # Save trajectory data
        if 'trajectory' in state:
            np.save(directory / 'trajectory.npy', state['trajectory'])

        # Save other state information
        with open(directory / 'dynamics_state.json', 'w') as f:
            json.dump({
                k: v for k, v in state.items()
                if k != 'trajectory' and isinstance(v, (dict, list, str, int, float, bool))
            }, f)

    def _save_analysis_state(self, directory: Path, state: Dict):
        """Save analysis state."""
        with open(directory / 'analysis_state.json', 'w') as f:
            json.dump({
                k: v for k, v in state.items()
                if isinstance(v, (dict, list, str, int, float, bool))
            }, f)

        # Save numpy arrays
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                np.save(directory / f'{key}.npy', value)

    def _save_progress_state(self, directory: Path, state: Dict):
        """Save progress tracking state."""
        with open(directory / 'progress_state.json', 'w') as f:
            json.dump(state, f)

    def _save_metadata(self,
                      checkpoint_dir: Path,
                      checkpoint_id: str,
                      component_states: Dict[str, Any]):
        """Save checkpoint metadata."""
        metadata = {
            'checkpoint_id': checkpoint_id,
            'timestamp': datetime.now().isoformat(),
            'components': list(component_states.keys()),
            'sizes': {
                component: self._get_state_size(state)
                for component, state in component_states.items()
            }
        }

        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

    def _get_state_size(self, state: Any) -> int:
        """Calculate approximate size of state in bytes."""
        if isinstance(state, (dict, list)):
            return len(json.dumps(state).encode())
        elif isinstance(state, np.ndarray):
            return state.nbytes
        elif hasattr(state, 'numpy'):  # PyTorch tensors
            return state.numpy().nbytes
        return 0

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint from path.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary of component states
        """
        checkpoint_dir = Path(checkpoint_path)
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_path}")

        try:
            # Load metadata
            with open(checkpoint_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            # Load component states
            states = {}
            for component in metadata['components']:
                component_dir = checkpoint_dir / component
                if component == 'structure':
                    states[component] = self._load_structure_state(component_dir)
                elif component == 'dynamics':
                    states[component] = self._load_dynamics_state(component_dir)
                elif component == 'analysis':
                    states[component] = self._load_analysis_state(component_dir)
                elif component == 'progress':
                    states[component] = self._load_progress_state(component_dir)

            return states

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise

    def _load_structure_state(self, directory: Path) -> Dict:
        """Load structure prediction state."""
        # Load basic state
        with open(directory / 'structure_state.json', 'r') as f:
            state = json.load(f)

        # Load numpy arrays
        for npy_file in directory.glob('*.npy'):
            key = npy_file.stem
            if key != 'structure_state':
                state[key] = np.load(npy_file)

        return state

    def _load_dynamics_state(self, directory: Path) -> Dict:
        """Load molecular dynamics state."""
        # Load basic state
        with open(directory / 'dynamics_state.json', 'r') as f:
            state = json.load(f)

        # Load trajectory
        if (directory / 'trajectory.npy').exists():
            state['trajectory'] = np.load(directory / 'trajectory.npy')

        return state

    def _load_analysis_state(self, directory: Path) -> Dict:
        """Load analysis state."""
        # Load basic state
        with open(directory / 'analysis_state.json', 'r') as f:
            state = json.load(f)

        # Load numpy arrays
        for npy_file in directory.glob('*.npy'):
            key = npy_file.stem
            if key != 'analysis_state':
                state[key] = np.load(npy_file)

        return state

    def _load_progress_state(self, directory: Path) -> Dict:
        """Load progress tracking state."""
        with open(directory / 'progress_state.json', 'r') as f:
            return json.load(f)

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints exceeding max_checkpoints."""
        checkpoints = sorted(
            [d for d in self.base_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if len(checkpoints) > self.max_checkpoints:
            for checkpoint in checkpoints[self.max_checkpoints:]:
                try:
                    shutil.rmtree(checkpoint)
                except Exception as e:
                    self.logger.error(f"Failed to remove checkpoint {checkpoint}: {str(e)}")

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints.

        Returns:
            List of checkpoint metadata
        """
        checkpoints = []
        for checkpoint_dir in self.base_dir.iterdir():
            if checkpoint_dir.is_dir():
                metadata_file = checkpoint_dir / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            metadata['path'] = str(checkpoint_dir)
                            checkpoints.append(metadata)
                    except Exception as e:
                        self.logger.error(f"Failed to read checkpoint metadata: {str(e)}")

        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)

    def verify_checkpoint(self, checkpoint_path: str) -> bool:
        """Verify checkpoint integrity.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            True if checkpoint is valid
        """
        try:
            checkpoint_dir = Path(checkpoint_path)
            if not checkpoint_dir.exists():
                return False

            # Check metadata
            metadata_file = checkpoint_dir / 'metadata.json'
            if not metadata_file.exists():
                return False

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Verify all component directories exist
            for component in metadata['components']:
                component_dir = checkpoint_dir / component
                if not component_dir.exists():
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Checkpoint verification failed: {str(e)}")
            return False
