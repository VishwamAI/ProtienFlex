"""
Integration tests for the complete protein flexibility analysis pipeline.
"""

import pytest
import numpy as np
import mdtraj as md
from pathlib import Path
import os
import tempfile
import shutil
import json

from models.pipeline import FlexibilityPipeline, AnalysisPipeline
from models.dynamics import FlexibilityAnalysis
from models.prediction import structure_converter

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / 'data'
ALANINE_PDB = TEST_DATA_DIR / 'alanine-dipeptide.pdb'
EXPERIMENTAL_DATA = TEST_DATA_DIR / 'experimental_bfactors.json'

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_proteins():
    """Sample protein sequences for testing."""
    return [
        {
            'name': 'protein1',
            'sequence': 'MKLLVLGLRSGSGKS',
            'experimental_data': {
                'b_factors': [15.0, 16.2, 17.1, 18.5, 19.2,
                             20.1, 18.9, 17.8, 16.5, 15.9,
                             15.2, 14.8, 14.5, 14.2, 14.0]
            }
        },
        {
            'name': 'protein2',
            'sequence': 'MALWMRLLPLLALLALWGPD',
            'experimental_data': {
                'b_factors': [14.5, 15.2, 16.8, 18.2, 19.5,
                             20.8, 19.2, 18.1, 17.2, 16.5,
                             15.8, 15.2, 14.8, 14.5, 14.2,
                             14.0, 13.8, 13.6, 13.5, 13.4]
            }
        }
    ]

@pytest.fixture
def pipeline(temp_dir):
    """Create FlexibilityPipeline instance."""
    return FlexibilityPipeline(
        alphafold_model_dir='/path/to/models',
        output_dir=temp_dir
    )

@pytest.fixture
def analysis_pipeline(temp_dir):
    """Create AnalysisPipeline instance."""
    return AnalysisPipeline(
        alphafold_model_dir='/path/to/models',
        output_dir=temp_dir,
        n_workers=2
    )

def test_end_to_end_single_protein(pipeline, sample_proteins, temp_dir):
    """Test complete pipeline for single protein analysis."""
    protein = sample_proteins[0]

    # Run complete analysis
    results = pipeline.analyze_sequence(
        sequence=protein['sequence'],
        name=protein['name'],
        experimental_data=protein['experimental_data']
    )

    # Check results structure
    assert 'structure_prediction' in results
    assert 'dynamics' in results
    assert 'flexibility' in results
    assert 'validation' in results

    # Check prediction results
    pred = results['structure_prediction']
    assert 'positions' in pred
    assert 'plddt' in pred
    assert 'mean_plddt' in pred

    # Check dynamics results
    dyn = results['dynamics']
    assert 'trajectory' in dyn
    assert 'energies' in dyn
    assert len(dyn['trajectory']) > 0

    # Check flexibility results
    flex = results['flexibility']
    assert 'rmsf' in flex
    assert 'ss_flexibility' in flex
    assert 'domain_movements' in flex

    # Check validation results
    val = results['validation']
    assert 'stability' in val
    assert 'sampling' in val
    if 'experimental' in val:
        assert 'b_factor_correlation' in val['experimental']

    # Check output files
    assert os.path.exists(os.path.join(temp_dir, f"{protein['name']}_results.json"))
    assert os.path.exists(os.path.join(temp_dir, f"{protein['name']}_trajectory.h5"))

def test_batch_analysis(analysis_pipeline, sample_proteins):
    """Test batch analysis of multiple proteins."""
    # Run batch analysis
    results = analysis_pipeline.analyze_proteins(
        proteins=[
            {
                'name': p['name'],
                'sequence': p['sequence']
            }
            for p in sample_proteins
        ],
        experimental_data={
            p['name']: p['experimental_data']
            for p in sample_proteins
        }
    )

    # Check results
    assert 'individual' in results
    assert 'aggregated' in results

    # Check individual results
    individual = results['individual']
    assert len(individual) == len(sample_proteins)
    for protein in sample_proteins:
        assert protein['name'] in individual

    # Check aggregated results
    aggregated = results['aggregated']
    assert 'flexibility_stats' in aggregated
    assert 'validation_stats' in aggregated
    assert 'performance_stats' in aggregated

def test_experimental_validation(pipeline, sample_proteins):
    """Test validation against experimental B-factors."""
    protein = sample_proteins[0]

    # Run analysis with experimental data
    results = pipeline.analyze_sequence(
        sequence=protein['sequence'],
        name=protein['name'],
        experimental_data=protein['experimental_data']
    )

    # Check experimental validation
    assert 'experimental' in results['validation']
    exp_val = results['validation']['experimental']

    assert 'b_factor_correlation' in exp_val
    assert 'rmsd_to_experimental' in exp_val
    assert 'relative_error' in exp_val

    # Check correlation coefficient
    assert -1.0 <= exp_val['b_factor_correlation'] <= 1.0

def test_pipeline_checkpointing(pipeline, sample_proteins, temp_dir):
    """Test pipeline checkpointing and resumption."""
    protein = sample_proteins[0]

    # Run with checkpointing
    results = pipeline.analyze_sequence(
        sequence=protein['sequence'],
        name=protein['name'],
        checkpoint_dir=os.path.join(temp_dir, 'checkpoints')
    )

    # Check checkpoint files
    checkpoint_dir = os.path.join(temp_dir, 'checkpoints', protein['name'])
    assert os.path.exists(checkpoint_dir)
    assert os.path.exists(os.path.join(checkpoint_dir, 'prediction.pkl'))
    assert os.path.exists(os.path.join(checkpoint_dir, 'trajectory.h5'))

    # Test resumption from checkpoint
    resumed_results = pipeline.analyze_sequence(
        sequence=protein['sequence'],
        name=protein['name'],
        checkpoint_dir=os.path.join(temp_dir, 'checkpoints'),
        resume=True
    )

    # Check resumed results match original
    assert resumed_results['structure_prediction']['mean_plddt'] == \
           results['structure_prediction']['mean_plddt']

def test_error_handling_and_recovery(pipeline, sample_proteins):
    """Test error handling and recovery in pipeline."""
    # Test with invalid sequence
    with pytest.raises(ValueError):
        pipeline.analyze_sequence(
            sequence="INVALID123",
            name="invalid_protein"
        )

    # Test with missing experimental data
    results = pipeline.analyze_sequence(
        sequence=sample_proteins[0]['sequence'],
        name=sample_proteins[0]['name'],
        experimental_data=None  # Missing experimental data
    )
    assert 'experimental' not in results['validation']

def test_performance_metrics(analysis_pipeline, sample_proteins):
    """Test performance metrics collection."""
    # Run batch analysis with performance monitoring
    results = analysis_pipeline.analyze_proteins(
        proteins=[
            {
                'name': p['name'],
                'sequence': p['sequence']
            }
            for p in sample_proteins
        ],
        collect_performance_metrics=True
    )

    # Check performance metrics
    assert 'performance_stats' in results['aggregated']
    perf = results['aggregated']['performance_stats']

    assert 'success_rate' in perf
    assert 'processing_times' in perf
    assert 'error_types' in perf

def test_result_serialization(pipeline, sample_proteins, temp_dir):
    """Test result serialization and deserialization."""
    protein = sample_proteins[0]

    # Run analysis
    results = pipeline.analyze_sequence(
        sequence=protein['sequence'],
        name=protein['name']
    )

    # Save results
    output_file = os.path.join(temp_dir, f"{protein['name']}_results.json")
    pipeline.save_results(results, output_file)

    # Load results
    loaded_results = pipeline.load_results(output_file)

    # Check loaded results match original
    assert loaded_results['structure_prediction']['mean_plddt'] == \
           results['structure_prediction']['mean_plddt']
    assert np.allclose(
        loaded_results['flexibility']['rmsf'],
        results['flexibility']['rmsf']
    )

@pytest.mark.parametrize("n_workers", [1, 2, 4])
def test_parallel_processing(temp_dir, sample_proteins, n_workers):
    """Test parallel processing with different numbers of workers."""
    # Create pipeline with different worker counts
    pipeline = AnalysisPipeline(
        alphafold_model_dir='/path/to/models',
        output_dir=temp_dir,
        n_workers=n_workers
    )

    # Run batch analysis
    results = pipeline.analyze_proteins(
        proteins=[
            {
                'name': p['name'],
                'sequence': p['sequence']
            }
            for p in sample_proteins
        ]
    )

    # Check results
    assert len(results['individual']) == len(sample_proteins)
    assert all(p['name'] in results['individual'] for p in sample_proteins)
