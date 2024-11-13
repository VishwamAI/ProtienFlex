# ProteinFlex Usage Examples

This directory contains examples demonstrating various use cases for protein flexibility analysis using the ProteinFlex pipeline.

## Basic Examples

### 1. Single Protein Analysis

```python
from models.pipeline import FlexibilityPipeline
from models.optimization import GPUManager

# Initialize GPU manager
gpu_manager = GPUManager(
    required_memory={
        'prediction': 16000,  # 16GB for structure prediction
        'dynamics': 8000      # 8GB for molecular dynamics
    }
)

# Initialize pipeline
pipeline = FlexibilityPipeline(
    output_dir='results',
    gpu_manager=gpu_manager
)

# Analyze single protein
results = pipeline.analyze_sequence(
    sequence='MKLLVLGLRSGSGKS',
    name='example_protein'
)

# Access flexibility metrics
print("Backbone Flexibility:")
print(f"RMSF values: {results['flexibility']['backbone_rmsf']}")
print(f"B-factors: {results['flexibility']['b_factors']}")

# Analyze domain movements
domains = results['flexibility']['domain_movements']
print("\nDomain Movements:")
for domain in domains:
    print(f"Domain {domain['id']}: {domain['movement_magnitude']} Å")

# Check side-chain mobility
sidechains = results['flexibility']['sidechain_mobility']
print("\nSide-chain Mobility:")
for residue, mobility in sidechains.items():
    print(f"Residue {residue}: {mobility:.2f}")
```

### 2. Batch Analysis with Progress Tracking

```python
from models.pipeline import AnalysisPipeline
from models.optimization import ProgressTracker, DataHandler

# Initialize components
tracker = ProgressTracker(total_steps=100)
data_handler = DataHandler(cache_dir='cache')

# Setup pipeline
pipeline = AnalysisPipeline(
    output_dir='results',
    progress_tracker=tracker,
    data_handler=data_handler
)

# Define proteins to analyze
proteins = [
    {
        'name': 'protein1',
        'sequence': 'MKLLVLGLRSGSGKS',
        'description': 'Example protein 1'
    },
    {
        'name': 'protein2',
        'sequence': 'MALWMRLLPLLALLALWGPD',
        'description': 'Example protein 2'
    }
]

# Run batch analysis
results = pipeline.analyze_proteins(
    proteins=proteins,
    checkpoint_dir='checkpoints'
)

# Process results
for protein_name, protein_results in results.items():
    print(f"\nResults for {protein_name}:")
    print("Flexibility Metrics:")
    print(f"- Average RMSF: {protein_results['flexibility']['avg_rmsf']:.2f}")
    print(f"- Flexible regions: {protein_results['flexibility']['flexible_regions']}")
    print(f"- Domain movements: {len(protein_results['flexibility']['domain_movements'])}")
```

### 3. Analysis with Experimental Validation

```python
from models.pipeline import ValidationPipeline
from models.analysis import ExperimentalValidator

# Load experimental data
experimental_data = {
    'protein1': {
        'b_factors': [15.0, 16.2, 17.1, 18.5, 19.2],
        'crystal_contacts': [(10, 15), (25, 30)],
        'temperature': 298
    }
}

# Initialize validator
validator = ExperimentalValidator()

# Setup pipeline with validation
pipeline = ValidationPipeline(
    output_dir='results',
    validator=validator
)

# Run analysis with validation
results = pipeline.analyze_sequence(
    sequence='MKLLVLGLRSGSGKS',
    name='protein1',
    experimental_data=experimental_data['protein1']
)

# Check validation results
validation = results['validation']
print("\nValidation Results:")
print(f"B-factor correlation: {validation['b_factor_correlation']:.2f}")
print(f"RMSD to crystal structure: {validation['rmsd']:.2f} Å")
print(f"Flexible region overlap: {validation['flexible_region_overlap']:.2f}")
```

## Advanced Examples

### 1. Custom Analysis Pipeline

```python
from models.pipeline import CustomPipeline
from models.analysis import (
    BackboneAnalyzer,
    SidechainAnalyzer,
    DomainAnalyzer
)

# Initialize analyzers
backbone_analyzer = BackboneAnalyzer(
    window_size=5,
    cutoff=2.0
)

sidechain_analyzer = SidechainAnalyzer(
    rotamer_library='dunbrack',
    energy_cutoff=2.0
)

domain_analyzer = DomainAnalyzer(
    algorithm='spectral',
    min_domain_size=30
)

# Create custom pipeline
pipeline = CustomPipeline(
    analyzers=[
        backbone_analyzer,
        sidechain_analyzer,
        domain_analyzer
    ],
    output_dir='results'
)

# Run analysis
results = pipeline.analyze_sequence(
    sequence='MKLLVLGLRSGSGKS',
    name='custom_analysis'
)

# Process detailed results
print("\nDetailed Analysis Results:")
print("\nBackbone Analysis:")
print(f"Flexible regions: {results['backbone']['flexible_regions']}")
print(f"Hinge points: {results['backbone']['hinge_points']}")

print("\nSide-chain Analysis:")
print(f"Rotamer distributions: {results['sidechain']['rotamer_stats']}")
print(f"Interaction networks: {results['sidechain']['interactions']}")

print("\nDomain Analysis:")
print(f"Domain boundaries: {results['domains']['boundaries']}")
print(f"Movement correlations: {results['domains']['correlations']}")
```

### 2. Enhanced Sampling Analysis

```python
from models.pipeline import EnhancedSamplingPipeline
from models.dynamics import (
    TemperatureREMD,
    Metadynamics,
    AcceleratedMD
)

# Setup enhanced sampling methods
remd = TemperatureREMD(
    temp_range=(300, 400),
    n_replicas=4
)

metad = Metadynamics(
    collective_variables=['phi', 'psi'],
    height=1.0,
    sigma=0.5
)

amd = AcceleratedMD(
    boost_potential=1.0,
    threshold_energy=-170000
)

# Initialize pipeline
pipeline = EnhancedSamplingPipeline(
    sampling_methods=[remd, metad, amd],
    output_dir='results'
)

# Run enhanced sampling
results = pipeline.analyze_sequence(
    sequence='MKLLVLGLRSGSGKS',
    name='enhanced_sampling',
    simulation_time=100  # ns
)

# Analyze sampling results
print("\nEnhanced Sampling Results:")
print("\nREMD Analysis:")
print(f"Exchange acceptance: {results['remd']['acceptance_rate']:.2f}")
print(f"Temperature distributions: {results['remd']['temp_dist']}")

print("\nMetadynamics Analysis:")
print(f"Free energy surface: {results['metad']['free_energy']}")
print(f"Convergence metric: {results['metad']['convergence']:.2f}")

print("\nAccelerated MD Analysis:")
print(f"Boost statistics: {results['amd']['boost_stats']}")
print(f"Reweighted ensembles: {results['amd']['reweighted_states']}")
```

### 3. Large-Scale Analysis with Distributed Computing

```python
from models.pipeline import DistributedPipeline
from models.optimization import (
    GPUManager,
    DataHandler,
    ProgressTracker
)

# Setup distributed components
gpu_manager = GPUManager(
    required_memory={
        'prediction': 16000,
        'dynamics': 8000
    },
    prefer_single_gpu=False
)

data_handler = DataHandler(
    cache_dir='distributed_cache',
    max_cache_size=500.0  # 500GB
)

tracker = ProgressTracker(
    total_steps=1000,
    checkpoint_interval=600  # 10 minutes
)

# Initialize distributed pipeline
pipeline = DistributedPipeline(
    output_dir='results',
    gpu_manager=gpu_manager,
    data_handler=data_handler,
    progress_tracker=tracker,
    n_workers=4
)

# Load protein dataset
proteins = load_protein_dataset('large_dataset.csv')

# Run distributed analysis
results = pipeline.analyze_proteins(
    proteins=proteins,
    batch_size=10,
    checkpoint_dir='distributed_checkpoints'
)

# Aggregate results
summary = pipeline.aggregate_results(results)
print("\nAnalysis Summary:")
print(f"Total proteins: {summary['total_proteins']}")
print(f"Average flexibility: {summary['avg_flexibility']:.2f}")
print(f"Flexibility distribution: {summary['flexibility_dist']}")
print(f"Common flexible motifs: {summary['flexible_motifs']}")
```

## Performance Tips

1. **GPU Memory Management**
   - Monitor GPU memory usage with `gpu_manager.get_memory_stats()`
   - Adjust batch sizes based on available memory
   - Use multi-GPU mode for large datasets

2. **Data Handling**
   - Enable compression for large trajectories
   - Use appropriate cache sizes
   - Clean up unused cache entries

3. **Progress Tracking**
   - Use hierarchical tasks for complex workflows
   - Enable auto-checkpointing for long runs
   - Monitor progress with detailed messages

4. **Validation**
   - Always validate against experimental data when available
   - Use multiple validation metrics
   - Consider crystal contacts in B-factor analysis

## Common Issues and Solutions

1. **Memory Issues**
   ```python
   # Solution: Adjust batch size
   batch_size = gpu_manager.get_optimal_batch_size('prediction', gpu_indices)
   ```

2. **Performance Bottlenecks**
   ```python
   # Solution: Enable data compression
   data_handler = DataHandler(enable_compression=True)
   ```

3. **Checkpoint Recovery**
   ```python
   # Solution: Use checkpoint manager
   states = checkpoint_manager.load_checkpoint(latest_checkpoint)
   ```

## Additional Resources

- [API Documentation](../api/README.md)
- [Performance Benchmarks](../benchmarks/README.md)
- [Validation Results](../benchmarks/validation.md)
