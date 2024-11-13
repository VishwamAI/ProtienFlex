"""Track progress of protein analysis computations"""
import time
from typing import Optional, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track and report progress of computations"""

    def __init__(self, total_steps: int, description: str = ""):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.description = description
        self.checkpoints = {}

    def start(self):
        """Start tracking progress"""
        self.start_time = time.time()
        self.current_step = 0
        logger.info(f"Started {self.description}")

    def update(self, steps: int = 1):
        """Update progress"""
        self.current_step = min(self.current_step + steps, self.total_steps)
        self._log_progress()

    def add_checkpoint(self, name: str):
        """Add a checkpoint"""
        self.checkpoints[name] = {
            'time': time.time(),
            'step': self.current_step,
            'progress': self.get_progress()
        }
        logger.info(f"Checkpoint '{name}' at {self.get_progress()['percentage']:.1f}%")

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress details"""
        if not self.start_time:
            return {'percentage': 0, 'elapsed': 0, 'remaining': 0}

        elapsed = time.time() - self.start_time
        progress = self.current_step / self.total_steps

        try:
            remaining = elapsed / progress * (1 - progress)
        except ZeroDivisionError:
            remaining = 0

        return {
            'percentage': progress * 100,
            'elapsed': elapsed,
            'remaining': remaining,
            'current_step': self.current_step,
            'total_steps': self.total_steps
        }

    def get_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """Get all checkpoints"""
        return self.checkpoints

    def _log_progress(self):
        """Log current progress"""
        progress = self.get_progress()
        logger.info(
            f"Progress: {progress['percentage']:.1f}% "
            f"({self.current_step}/{self.total_steps}) - "
            f"Elapsed: {progress['elapsed']:.1f}s, "
            f"Remaining: {progress['remaining']:.1f}s"
        )
