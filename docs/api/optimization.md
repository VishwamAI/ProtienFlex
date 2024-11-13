# Optimization Components API Documentation

## Overview

The optimization components provide GPU acceleration, efficient data handling, progress tracking, and checkpointing capabilities for the ProteinFlex pipeline. These components are designed to work together to ensure optimal performance and reliability.

## GPUManager

The `GPUManager` class handles GPU resource allocation and optimization across different computational tasks.

```python
from models.optimization import GPUManager

class GPUManager:
    def __init__(self,
                 required_memory: Dict[str, int] = None,
                 prefer_single_gpu: bool = False):
        """Initialize GPU manager.

        Args:
            required_memory: Memory requirements per component
                Example: {'prediction': 16000, 'dynamics': 8000}  # MB
            prefer_single_gpu: Prefer single GPU usage when possible
        """
        pass

    def get_available_gpus(self) -> List[Dict[str, Any]]:
        """Get list of available GPUs with their properties.

        Returns:
            List of dictionaries containing GPU information:
            [{'index': 0, 'name': 'NVIDIA A100', 'memory_total': 40000,
              'memory_free': 38000, 'compute_capability': (8, 0)}]
        """
        pass

    def allocate_gpus(self, task: str) -> List[int]:
        """Allocate GPUs for specific task based on requirements.

        Args:
            task: Task identifier ('prediction' or 'dynamics')

        Returns:
            List of allocated GPU indices
        """
        pass

    def optimize_memory_usage(self,
                            task: str,
                            gpu_indices: List[int]):
        """Optimize memory usage for given task and GPUs.

        Args:
            task: Task identifier
            gpu_indices: List of GPU indices to optimize
        """
        pass

    def get_optimal_batch_size(self,
                             task: str,
                             gpu_indices: List[int]) -> int:
        """Calculate optimal batch size based on available GPU memory.

        Args:
            task: Task identifier
            gpu_indices: List of GPU indices

        Returns:
            Optimal batch size for the task
        """
        pass
```

### Usage Example

```python
# Initialize GPU manager with memory requirements
gpu_manager = GPUManager(
    required_memory={
        'prediction': 16000,  # 16GB for structure prediction
        'dynamics': 8000      # 8GB for molecular dynamics
    }
)

# Get available GPUs
available_gpus = gpu_manager.get_available_gpus()
print(f"Available GPUs: {available_gpus}")

# Allocate GPUs for prediction
prediction_gpus = gpu_manager.allocate_gpus('prediction')
print(f"Allocated GPUs for prediction: {prediction_gpus}")

# Get optimal batch size
batch_size = gpu_manager.get_optimal_batch_size('prediction', prediction_gpus)
print(f"Optimal batch size: {batch_size}")
```

## DataHandler

The `DataHandler` class manages efficient data transfer and caching between pipeline components.

```python
from models.optimization import DataHandler

class DataHandler:
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
        pass

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
        pass

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
        pass

    def load_data(self, cache_key: str) -> Dict[str, Any]:
        """Load data from cache.

        Args:
            cache_key: Cache key for stored data

        Returns:
            Dictionary containing stored data
        """
        pass

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        pass
```

### Usage Example

```python
# Initialize data handler
data_handler = DataHandler(
    cache_dir='cache',
    max_cache_size=100.0,  # 100GB
    enable_compression=True
)

# Store structure data
structure_key = data_handler.store_structure(
    structure_data={
        'positions': coordinates,
        'plddt': confidence_scores
    },
    data_id='protein1',
    metadata={'resolution': 'high'}
)

# Store trajectory data
trajectory_key = data_handler.store_trajectory(
    trajectory_data={
        'frames': trajectory_frames,
        'time': time_steps
    },
    data_id='protein1_md',
    metadata={'timestep': 2.0}
)

# Load data
structure = data_handler.load_data(structure_key)
trajectory = data_handler.load_data(trajectory_key)

# Check cache statistics
stats = data_handler.get_cache_stats()
print(f"Cache usage: {stats['usage_percent']}%")
```

## ProgressTracker

The `ProgressTracker` class provides real-time progress tracking for long-running operations.

```python
from models.optimization import ProgressTracker

class ProgressTracker:
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
        pass

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
        pass

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
        pass

    def get_progress(self) -> Dict[str, Any]:
        """Get overall progress information.

        Returns:
            Dictionary containing progress information
        """
        pass
```

### Usage Example

```python
# Initialize progress tracker
tracker = ProgressTracker(
    total_steps=100,
    checkpoint_dir='checkpoints',
    auto_checkpoint=True
)

# Start main task
tracker.start_task(
    task_id='protein1',
    task_name='Protein Analysis',
    total_steps=3
)

# Start subtask
tracker.start_task(
    task_id='prediction',
    task_name='Structure Prediction',
    total_steps=100,
    parent_task='protein1'
)

# Update progress
tracker.update_task(
    task_id='prediction',
    steps=10,
    message='Processing MSA'
)

# Get progress
progress = tracker.get_progress()
print(f"Overall progress: {progress['percent']}%")
```

## CheckpointManager

The `CheckpointManager` class coordinates checkpointing across pipeline components.

```python
from models.optimization import CheckpointManager

class CheckpointManager:
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
        pass

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
        pass

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint from path.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary of component states
        """
        pass

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints.

        Returns:
            List of checkpoint metadata
        """
        pass
```

### Usage Example

```python
# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(
    base_dir='checkpoints',
    max_checkpoints=5
)

# Create checkpoint
checkpoint_path = checkpoint_manager.create_checkpoint(
    checkpoint_id='analysis_1',
    component_states={
        'structure': structure_state,
        'dynamics': dynamics_state,
        'analysis': analysis_state,
        'progress': progress_state
    }
)

# List checkpoints
checkpoints = checkpoint_manager.list_checkpoints()
print(f"Available checkpoints: {checkpoints}")

# Load checkpoint
states = checkpoint_manager.load_checkpoint(checkpoint_path)
```

## Performance Considerations

### GPU Memory Management
- Structure prediction requires ~16GB VRAM
- Molecular dynamics requires ~8GB VRAM
- Batch processing adjusts automatically based on available memory
- Multi-GPU support for parallel processing

### Data Handling
- Compression reduces storage requirements by ~60%
- Caching improves repeated analysis performance
- Automatic cleanup of old cache entries
- Efficient memory usage for large trajectories

### Progress Tracking
- Minimal overhead (<1% CPU usage)
- Real-time updates without blocking
- Hierarchical task tracking
- Automatic checkpointing

### Checkpointing
- Component-specific state saving
- Efficient storage format
- Automatic cleanup of old checkpoints
- Fast state recovery

## Best Practices

1. **GPU Usage**
   - Monitor memory usage with `get_cache_stats()`
   - Use multi-GPU mode for large batch processing
   - Adjust batch sizes based on available memory

2. **Data Management**
   - Enable compression for large datasets
   - Set appropriate cache size limits
   - Clean up unused cache entries regularly

3. **Progress Tracking**
   - Use hierarchical tasks for complex workflows
   - Include informative progress messages
   - Enable auto-checkpointing for long runs

4. **Checkpointing**
   - Create checkpoints at logical workflow points
   - Maintain reasonable checkpoint history
   - Verify checkpoint integrity after saving
