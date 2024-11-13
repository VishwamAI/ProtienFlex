"""
Unit tests for flexibility analysis components.
"""

import pytest
import numpy as np
import mdtraj as md
from pathlib import Path
import os

from models.flexibility import backbone_flexibility, sidechain_mobility, domain_movements
from models.dynamics import FlexibilityAnalysis

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / 'data'
ALANINE_PDB = TEST_DATA_DIR / 'alanine-dipeptide.pdb'

@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    # Load alanine dipeptide trajectory
    traj = md.load(str(ALANINE_PDB))
    # Create a small trajectory with multiple frames
    frames = [traj.xyz[0] + np.random.normal(0, 0.01, traj.xyz[0].shape)
             for _ in range(10)]
    multi_frame = md.Trajectory(
        xyz=np.array(frames),
        topology=traj.topology,
        time=np.arange(len(frames))
    )
    return multi_frame

@pytest.fixture
def flexibility_analyzer(sample_trajectory):
    """Create FlexibilityAnalysis instance."""
    return FlexibilityAnalysis(sample_trajectory)

def test_rmsf_calculation(flexibility_analyzer):
    """Test RMSF calculation."""
    rmsf = flexibility_analyzer.calculate_rmsf()

    # Basic checks
    assert isinstance(rmsf, np.ndarray)
    assert len(rmsf) > 0
    assert np.all(rmsf >= 0)  # RMSF should be non-negative

    # Test with alignment
    rmsf_aligned = flexibility_analyzer.calculate_rmsf(align=True)
    assert np.allclose(rmsf_aligned, rmsf_aligned)  # Should be reproducible

    # Test specific atom selection
    ca_indices = flexibility_analyzer.topology.select('name CA')
    rmsf_ca = flexibility_analyzer.calculate_rmsf(atom_indices=ca_indices)
    assert len(rmsf_ca) == len(ca_indices)

def test_secondary_structure_flexibility(flexibility_analyzer):
    """Test secondary structure flexibility analysis."""
    ss_flex = flexibility_analyzer.analyze_secondary_structure_flexibility()

    # Check structure types
    assert all(ss_type in ss_flex for ss_type in ['H', 'E', 'C'])

    # Check values
    for ss_type, value in ss_flex.items():
        assert isinstance(value, float)
        assert value >= 0  # Flexibility measure should be non-negative

def test_residue_correlations(flexibility_analyzer):
    """Test residue correlation calculation."""
    # Test linear correlations
    corr_linear = flexibility_analyzer.calculate_residue_correlations(method='linear')
    assert isinstance(corr_linear, np.ndarray)
    assert corr_linear.shape[0] == corr_linear.shape[1]  # Should be square matrix
    assert np.allclose(corr_linear, corr_linear.T)  # Should be symmetric
    assert np.all(np.abs(corr_linear) <= 1.0)  # Correlations should be in [-1, 1]

    # Test mutual information
    corr_mi = flexibility_analyzer.calculate_residue_correlations(method='mutual_information')
    assert isinstance(corr_mi, np.ndarray)
    assert corr_mi.shape == corr_linear.shape
    assert np.all(corr_mi >= 0)  # MI should be non-negative

def test_flexible_regions_identification(flexibility_analyzer):
    """Test identification of flexible regions."""
    regions = flexibility_analyzer.identify_flexible_regions(percentile=90.0)

    # Check format
    assert isinstance(regions, list)
    for start, end in regions:
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start <= end

    # Test different percentiles
    regions_strict = flexibility_analyzer.identify_flexible_regions(percentile=95.0)
    regions_loose = flexibility_analyzer.identify_flexible_regions(percentile=80.0)
    assert len(regions_strict) <= len(regions)  # Stricter threshold should find fewer regions
    assert len(regions_loose) >= len(regions)  # Looser threshold should find more regions

def test_domain_movements(flexibility_analyzer):
    """Test domain movement analysis."""
    results = flexibility_analyzer.analyze_domain_movements()

    # Check result structure
    assert 'domain_centers' in results
    assert 'domain_movements' in results
    assert 'domain_assignments' in results

    # Check domain assignments
    assignments = results['domain_assignments']
    assert len(assignments) == len(flexibility_analyzer.topology.select('name CA'))
    assert len(np.unique(assignments)) >= 1  # Should identify at least one domain

def test_conformational_substates(flexibility_analyzer):
    """Test conformational substate analysis."""
    results = flexibility_analyzer.analyze_conformational_substates(n_clusters=3)

    # Check result structure
    assert 'labels' in results
    assert 'centers' in results
    assert 'transitions' in results
    assert 'populations' in results

    # Check dimensions
    n_frames = len(flexibility_analyzer.trajectory)
    assert len(results['labels']) == n_frames
    assert len(results['populations']) == 3  # We requested 3 clusters
    assert results['transitions'].shape == (3, 3)  # Transition matrix should be square

    # Check probabilities
    assert np.allclose(np.sum(results['populations']), 1.0)  # Populations should sum to 1
    assert np.allclose(np.sum(results['transitions'], axis=1), 1.0)  # Transition probabilities should sum to 1

def test_entropy_profile(flexibility_analyzer):
    """Test conformational entropy calculation."""
    entropy = flexibility_analyzer.calculate_entropy_profile(window_size=5)

    # Check dimensions
    n_residues = len(flexibility_analyzer.topology.select('name CA'))
    assert len(entropy) == n_residues

    # Test different window sizes
    entropy_large = flexibility_analyzer.calculate_entropy_profile(window_size=10)
    assert len(entropy_large) == len(entropy)
    assert not np.allclose(entropy_large, entropy)  # Different window sizes should give different results

def test_flexibility_profile(flexibility_analyzer):
    """Test comprehensive flexibility profile calculation."""
    profile = flexibility_analyzer.calculate_flexibility_profile()

    # Check result structure
    assert 'rmsf' in profile
    assert 'ss_flexibility' in profile
    assert 'correlations' in profile
    assert 'flexible_regions' in profile
    assert 'domain_analysis' in profile

    # Check RMSF
    assert isinstance(profile['rmsf'], np.ndarray)
    assert len(profile['rmsf']) > 0

    # Check secondary structure flexibility
    assert all(ss_type in profile['ss_flexibility'] for ss_type in ['H', 'E', 'C'])

    # Check correlations
    assert isinstance(profile['correlations'], np.ndarray)
    assert profile['correlations'].shape[0] == profile['correlations'].shape[1]

@pytest.mark.parametrize("window_size", [3, 5, 10])
def test_entropy_profile_window_sizes(flexibility_analyzer, window_size):
    """Test entropy profile calculation with different window sizes."""
    entropy = flexibility_analyzer.calculate_entropy_profile(window_size=window_size)
    assert len(entropy) == len(flexibility_analyzer.topology.select('name CA'))
    assert np.all(np.isfinite(entropy))  # All values should be finite

def test_error_handling(sample_trajectory):
    """Test error handling in flexibility analysis."""
    # Test with invalid trajectory
    with pytest.raises(ValueError):
        FlexibilityAnalysis(None)

    # Test with empty trajectory
    empty_traj = md.Trajectory(
        xyz=np.empty((0, sample_trajectory.n_atoms, 3)),
        topology=sample_trajectory.topology
    )
    with pytest.raises(ValueError):
        FlexibilityAnalysis(empty_traj)

    # Test invalid method for correlation calculation
    analyzer = FlexibilityAnalysis(sample_trajectory)
    with pytest.raises(ValueError):
        analyzer.calculate_residue_correlations(method='invalid_method')
