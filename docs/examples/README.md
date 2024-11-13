# ProteinFlex Examples

This directory contains examples demonstrating the usage of ProteinFlex for protein flexibility analysis.

## Basic Usage

### 1. Analyzing Protein Flexibility

```python
from models.flexibility import BackboneFlexibility, SidechainMobility, DomainMovements

# Initialize analyzers
backbone = BackboneFlexibility()
sidechain = SidechainMobility()
domains = DomainMovements()

# Load trajectory
trajectory_file = "protein_trajectory.dcd"
topology_file = "protein.pdb"

backbone.load_trajectory(trajectory_file, topology_file)
sidechain.load_trajectory(trajectory_file, topology_file)
domains.load_trajectory(trajectory_file, topology_file)

# Analyze flexibility
rmsd = backbone.calculate_rmsd()
rmsf = backbone.calculate_rmsf()
ss_analysis = backbone.analyze_secondary_structure()

# Analyze sidechain mobility
chi_angles = sidechain.calculate_chi_angles()
rotamer_pops = sidechain.analyze_rotamer_populations(chi_angles)

# Analyze domain movements
domain_regions = domains.identify_domains()
movements = domains.calculate_domain_movements(domain_regions)
hinges = domains.analyze_hinge_regions(domain_regions)
```

### 2. Using the Analysis Pipeline

```python
from models.pipeline import AnalysisPipeline

# Initialize pipeline
pipeline = AnalysisPipeline(gpu_required=True)

# Analyze protein sequence
results = pipeline.analyze_sequence(
    sequence="MVKVGVNG...",
    output_dir="./output"
)

# Save results
pipeline.save_results(results, "analysis_results.h5")
```

### 3. Structure Prediction

```python
from models.prediction import AlphaFoldInterface, StructureConverter

# Initialize predictors
predictor = AlphaFoldInterface()
converter = StructureConverter()

# Predict structure
prediction = predictor.predict_structure("MVKVGVNG...")

# Convert and save structure
structure = converter.convert_structure(prediction, output_format='pdb')
converter.save_structure(structure, "predicted_structure.pdb")
```

## Advanced Examples

### 1. GPU-Optimized Analysis

```python
from models.optimization import GPUManager
from models.pipeline import FlexibilityPipeline

# Initialize GPU manager
gpu_manager = GPUManager()
status = gpu_manager.get_memory_status()

# Initialize pipeline with GPU support
pipeline = FlexibilityPipeline(gpu_required=True)

# Run analysis
results = pipeline.analyze_protein(
    pdb_file="protein.pdb",
    trajectory_file="trajectory.dcd"
)
```

### 2. Progress Tracking

```python
from models.optimization import ProgressTracker, CheckpointManager

# Initialize trackers
progress = ProgressTracker(total_steps=100)
checkpoints = CheckpointManager()

# Track progress
progress.start()
for i in range(100):
    # Do work
    progress.update()
    if i % 10 == 0:
        progress.add_checkpoint(f"step_{i}")

# Save checkpoint
checkpoints.save_checkpoint(state_dict, "analysis_checkpoint")
```

## Performance Tips

1. Use GPU acceleration when available
2. Enable checkpointing for long computations
3. Optimize batch sizes for your hardware
4. Use efficient data handling for large trajectories

## Error Handling

```python
try:
    pipeline = FlexibilityPipeline()
    results = pipeline.analyze_protein(pdb_file, trajectory_file)
except Exception as e:
    logger.error(f"Analysis failed: {e}")
    # Handle error appropriately
```
