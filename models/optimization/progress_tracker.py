"""
Progress Tracker Module

Provides real-time progress tracking and monitoring for long-running
protein analysis operations with support for nested tasks and checkpointing.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import os
from pathlib import Path
import threading
from queue import Queue
import signal

class ProgressTracker:
    """Tracks progress of long-running operations with nested task support."""

    def __init__(self,
                 total_steps: int = 100,
                 checkpoint_dir: Optional[str] = None,
                 auto_checkpoint: bool = True,
                 checkpoint_interval: int = 300):  # 5 minutes
        """Initialize progress tracker.

        Args:
            total_steps: Total number of steps in the pipeline
            checkpoint_dir: Directory for saving checkpoints
            auto_checkpoint: Whether to automatically save checkpoints
            checkpoint_interval: Interval between checkpoints in seconds
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.checkpoint_dir = checkpoint_dir
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.logger = logging.getLogger(__name__)

        # Task tracking
        self.tasks = {}
        self.active_tasks = set()
        self.task_progress = {}
        self.task_messages = Queue()

        # Performance metrics
        self.performance_metrics = {
            'step_times': [],
            'memory_usage': [],
            'gpu_utilization': []
        }

        # Initialize checkpoint directory
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Start monitoring thread if auto-checkpointing is enabled
        if auto_checkpoint and checkpoint_dir:
            self._start_checkpoint_thread()

    def _start_checkpoint_thread(self):
        """Start background thread for automatic checkpointing."""
        self.checkpoint_thread = threading.Thread(
            target=self._checkpoint_monitor,
            daemon=True
        )
        self.checkpoint_thread.start()

    def _checkpoint_monitor(self):
        """Monitor and trigger automatic checkpoints."""
        while True:
            time.sleep(self.checkpoint_interval)
            if self.active_tasks:
                try:
                    self.save_checkpoint()
                except Exception as e:
                    self.logger.error(f"Auto-checkpoint failed: {str(e)}")

    def start(self):
        """Start progress tracking."""
        self.start_time = time.time()
        self.current_step = 0
        self.active_tasks.clear()
        self.task_progress.clear()
        self.performance_metrics = {
            'step_times': [],
            'memory_usage': [],
            'gpu_utilization': []
        }

    def update(self,
               steps: int = 1,
               message: Optional[str] = None,
               performance_metrics: Optional[Dict] = None):
        """Update progress.

        Args:
            steps: Number of steps completed
            message: Optional status message
            performance_metrics: Optional performance metrics
        """
        self.current_step = min(self.current_step + steps, self.total_steps)

        if message:
            self.task_messages.put({
                'time': datetime.now().isoformat(),
                'message': message,
                'progress': self.get_progress()
            })

        if performance_metrics:
            self._update_performance_metrics(performance_metrics)

        # Log progress
        progress = self.get_progress()
        self.logger.info(
            f"Progress: {progress['percent']:.1f}% - "
            f"Step {self.current_step}/{self.total_steps}"
        )

    def start_task(self,
                   task_id: str,
                   task_name: str,
                   total_steps: int,
                   parent_task: Optional[str] = None):
        """Start tracking a new task.

        Args:
            task_id: Unique task identifier
            task_name: Human-readable task name
            total_steps: Total steps in this task
            parent_task: Optional parent task ID
        """
        self.tasks[task_id] = {
            'name': task_name,
            'total_steps': total_steps,
            'current_step': 0,
            'start_time': time.time(),
            'parent_task': parent_task,
            'subtasks': set()
        }

        if parent_task and parent_task in self.tasks:
            self.tasks[parent_task]['subtasks'].add(task_id)

        self.active_tasks.add(task_id)
        self.task_progress[task_id] = 0.0

    def update_task(self,
                    task_id: str,
                    steps: int = 1,
                    message: Optional[str] = None):
        """Update task progress.

        Args:
            task_id: Task identifier
            steps: Number of steps completed
            message: Optional status message
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task['current_step'] = min(
            task['current_step'] + steps,
            task['total_steps']
        )

        # Update task progress
        self.task_progress[task_id] = (
            task['current_step'] / task['total_steps']
        )

        if message:
            self.task_messages.put({
                'time': datetime.now().isoformat(),
                'task_id': task_id,
                'message': message,
                'progress': self.get_task_progress(task_id)
            })

        # Update parent task progress
        if task['parent_task']:
            self._update_parent_progress(task['parent_task'])

    def _update_parent_progress(self, parent_id: str):
        """Update parent task progress based on subtasks."""
        parent = self.tasks[parent_id]
        if parent['subtasks']:
            # Average progress of all subtasks
            subtask_progress = [
                self.task_progress[subtask_id]
                for subtask_id in parent['subtasks']
            ]
            parent_progress = sum(subtask_progress) / len(subtask_progress)
            self.task_progress[parent_id] = parent_progress

            # Recursively update higher-level parents
            if parent['parent_task']:
                self._update_parent_progress(parent['parent_task'])

    def complete_task(self, task_id: str, message: Optional[str] = None):
        """Mark task as complete.

        Args:
            task_id: Task identifier
            message: Optional completion message
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task['current_step'] = task['total_steps']
        task['end_time'] = time.time()
        self.task_progress[task_id] = 1.0
        self.active_tasks.remove(task_id)

        if message:
            self.task_messages.put({
                'time': datetime.now().isoformat(),
                'task_id': task_id,
                'message': message,
                'progress': 1.0,
                'status': 'completed'
            })

    def get_progress(self) -> Dict[str, Any]:
        """Get overall progress information.

        Returns:
            Dictionary containing progress information
        """
        current_time = time.time()
        elapsed = current_time - (self.start_time or current_time)

        progress = {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'percent': (self.current_step / self.total_steps) * 100,
            'elapsed_time': elapsed,
            'active_tasks': len(self.active_tasks),
            'task_progress': self.task_progress.copy()
        }

        # Estimate remaining time
        if self.current_step > 0:
            steps_per_second = self.current_step / elapsed
            remaining_steps = self.total_steps - self.current_step
            progress['estimated_remaining'] = remaining_steps / steps_per_second
        else:
            progress['estimated_remaining'] = None

        return progress

    def get_task_progress(self, task_id: str) -> Dict[str, Any]:
        """Get detailed progress for specific task.

        Args:
            task_id: Task identifier

        Returns:
            Dictionary containing task progress information
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        current_time = time.time()
        elapsed = current_time - task['start_time']

        progress = {
            'name': task['name'],
            'current_step': task['current_step'],
            'total_steps': task['total_steps'],
            'percent': (task['current_step'] / task['total_steps']) * 100,
            'elapsed_time': elapsed,
            'parent_task': task['parent_task'],
            'subtasks': list(task['subtasks'])
        }

        # Add completion time if task is finished
        if 'end_time' in task:
            progress['completion_time'] = task['end_time'] - task['start_time']

        return progress

    def _update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics.

        Args:
            metrics: Dictionary of performance metrics
        """
        if 'step_time' in metrics:
            self.performance_metrics['step_times'].append(metrics['step_time'])
        if 'memory_usage' in metrics:
            self.performance_metrics['memory_usage'].append(metrics['memory_usage'])
        if 'gpu_utilization' in metrics:
            self.performance_metrics['gpu_utilization'].append(metrics['gpu_utilization'])

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary.

        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            'step_times': {
                'mean': np.mean(self.performance_metrics['step_times'])
                if self.performance_metrics['step_times'] else None,
                'max': max(self.performance_metrics['step_times'])
                if self.performance_metrics['step_times'] else None
            },
            'memory_usage': {
                'mean': np.mean(self.performance_metrics['memory_usage'])
                if self.performance_metrics['memory_usage'] else None,
                'max': max(self.performance_metrics['memory_usage'])
                if self.performance_metrics['memory_usage'] else None
            },
            'gpu_utilization': {
                'mean': np.mean(self.performance_metrics['gpu_utilization'])
                if self.performance_metrics['gpu_utilization'] else None,
                'max': max(self.performance_metrics['gpu_utilization'])
                if self.performance_metrics['gpu_utilization'] else None
            }
        }
        return metrics

    def save_checkpoint(self):
        """Save progress checkpoint."""
        if not self.checkpoint_dir:
            raise ValueError("Checkpoint directory not specified")

        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'progress': self.get_progress(),
            'tasks': self.tasks,
            'task_progress': self.task_progress,
            'performance_metrics': self.performance_metrics
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_{int(time.time())}.json"
        )

        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")
            raise

    def load_checkpoint(self, checkpoint_path: str):
        """Load progress from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)

            self.tasks = checkpoint_data['tasks']
            self.task_progress = checkpoint_data['task_progress']
            self.performance_metrics = checkpoint_data['performance_metrics']
            self.active_tasks = {
                task_id for task_id, task in self.tasks.items()
                if 'end_time' not in task
            }

            progress = checkpoint_data['progress']
            self.current_step = progress['current_step']
            self.start_time = time.time() - progress['elapsed_time']

            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            raise
