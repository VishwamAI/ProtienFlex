"""
Unit tests for molecular dynamics components.
"""

import pytest
import numpy as np
import mdtraj as md
from pathlib import Path
import os
import tempfile
import shutil

from models.dynamics import EnhancedSampling, SimulationValidator
from models.dynamics.simulation import ReplicaExchange, Metadynamics

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / 'data'
ALANINE_PDB = TEST_DATA_DIR / 'alanine-dipeptide.pdb'

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_system():
    """Create a sample molecular system for testing."""
    import openmm as mm
    import openmm.app as app

    # Load alanine dipeptide
    pdb = app.PDBFile(str(ALANINE_PDB))

    # Create system with implicit solvent
    forcefield = app.ForceField('amber99sb.xml', 'implicit/gbn2.xml')
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds
    )

    return {
        'system': system,
        'topology': pdb.topology,
        'positions': pdb.positions
    }

@pytest.fixture
def enhanced_sampling(sample_system):
    """Create EnhancedSampling instance."""
    return EnhancedSampling(sample_system)

def test_replica_exchange_setup(enhanced_sampling):
    """Test replica exchange setup."""
    n_replicas = 4
    replicas = enhanced_sampling.setup_replica_exchange(
        n_replicas=n_replicas,
        temp_min=300.0,
        temp_max=400.0
    )

    # Check number of replicas
    assert len(replicas) == n_replicas

    # Check temperature ladder
    temps = [r.temperature for r in replicas]
    assert temps[0] == 300.0
    assert temps[-1] == 400.0
    assert all(t2 > t1 for t1, t2 in zip(temps[:-1], temps[1:]))

def test_metadynamics_setup(enhanced_sampling):
    """Test metadynamics setup."""
    # Define collective variables
    cv_definitions = [
        {'type': 'distance', 'atoms': [0, 5]},
        {'type': 'angle', 'atoms': [0, 5, 10]}
    ]

    meta = enhanced_sampling.setup_metadynamics(
        cv_definitions=cv_definitions,
        height=1.0,
        sigma=0.05,
        deposition_frequency=100
    )

    # Check metadynamics parameters
    assert meta.height == 1.0
    assert meta.sigma == 0.05
    assert meta.deposition_frequency == 100
    assert len(meta.cv_definitions) == len(cv_definitions)

def test_simulation_run(enhanced_sampling, temp_dir):
    """Test basic simulation run."""
    # Setup and run short simulation
    n_steps = 1000
    trajectory = enhanced_sampling.run_simulation(
        n_steps=n_steps,
        output_dir=temp_dir
    )

    # Check trajectory
    assert isinstance(trajectory, md.Trajectory)
    assert len(trajectory) > 0
    assert os.path.exists(os.path.join(temp_dir, 'traj.h5'))

def test_replica_exchange_simulation(enhanced_sampling, temp_dir):
    """Test replica exchange simulation."""
    # Setup replicas
    n_replicas = 2
    replicas = enhanced_sampling.setup_replica_exchange(
        n_replicas=n_replicas,
        temp_min=300.0,
        temp_max=350.0
    )

    # Run short simulation
    n_steps = 1000
    results = enhanced_sampling.run_replica_exchange(
        n_steps=n_steps,
        exchange_frequency=100,
        output_dir=temp_dir
    )

    # Check results
    assert 'exchange_stats' in results
    assert 'trajectories' in results
    assert len(results['trajectories']) == n_replicas
    assert os.path.exists(os.path.join(temp_dir, 'replica_0/traj.h5'))

def test_metadynamics_simulation(enhanced_sampling, temp_dir):
    """Test metadynamics simulation."""
    # Setup metadynamics
    cv_definitions = [
        {'type': 'distance', 'atoms': [0, 5]}
    ]
    meta = enhanced_sampling.setup_metadynamics(
        cv_definitions=cv_definitions,
        height=1.0,
        sigma=0.05
    )

    # Run short simulation
    n_steps = 1000
    results = enhanced_sampling.run_metadynamics(
        n_steps=n_steps,
        output_dir=temp_dir
    )


    # Check results
    assert 'trajectory' in results
    assert 'cv_values' in results
    assert 'bias_potential' in results
    assert len(results['cv_values']) > 0
    assert os.path.exists(os.path.join(temp_dir, 'meta_traj.h5'))

def test_temperature_exchange(enhanced_sampling):
    """Test temperature exchange calculations."""
    # Create two replicas
    replica1 = ReplicaExchange(temperature=300.0)
    replica2 = ReplicaExchange(temperature=350.0)

    # Set energies
    energy1, energy2 = -1000.0, -900.0

    # Calculate exchange probability
    prob = enhanced_sampling._calculate_exchange_probability(
        energy1, energy2,
        replica1.temperature, replica2.temperature
    )

    # Check probability
    assert 0.0 <= prob <= 1.0

    # Test extreme cases
    prob_same = enhanced_sampling._calculate_exchange_probability(
        energy1, energy1,
        replica1.temperature, replica1.temperature
    )
    assert np.isclose(prob_same, 1.0)

def test_bias_potential_calculation(enhanced_sampling):
    """Test bias potential calculations."""
    # Setup metadynamics
    cv_definitions = [{'type': 'distance', 'atoms': [0, 5]}]
    meta = enhanced_sampling.setup_metadynamics(
        cv_definitions=cv_definitions,
        height=1.0,
        sigma=0.05
    )

    # Generate some CV values and Gaussians
    cv_values = np.array([0.0, 0.1, 0.2])
    gaussian_centers = np.array([0.05, 0.15])

    # Calculate bias potential
    bias = meta._calculate_bias_potential(cv_values, gaussian_centers)

    # Check bias potential
    assert len(bias) == len(cv_values)
    assert np.all(bias >= 0)  # Bias should be non-negative

def test_simulation_validator(enhanced_sampling, temp_dir):
    """Test simulation validation."""
    # Run short simulation
    trajectory = enhanced_sampling.run_simulation(
        n_steps=1000,
        output_dir=temp_dir
    )

    # Create validator
    validator = SimulationValidator(trajectory)

    # Test stability validation
    stability = validator.validate_simulation_stability()
    assert 'rmsd_mean' in stability
    assert 'rg_mean' in stability

    # Test sampling validation
    sampling = validator.validate_sampling_quality()
    assert 'population_entropy' in sampling
    assert 'transition_density' in sampling

def test_error_handling(enhanced_sampling, temp_dir):
    """Test error handling in dynamics components."""
    # Test invalid replica exchange setup
    with pytest.raises(ValueError):
        enhanced_sampling.setup_replica_exchange(n_replicas=1)  # Need at least 2

    # Test invalid metadynamics setup
    with pytest.raises(ValueError):
        enhanced_sampling.setup_metadynamics(
            cv_definitions=[],  # Empty CV definitions
            height=1.0,
            sigma=0.05
        )

    # Test invalid simulation parameters
    with pytest.raises(ValueError):
        enhanced_sampling.run_simulation(n_steps=-1)  # Negative steps

@pytest.mark.parametrize("n_replicas,temp_min,temp_max", [
    (2, 300.0, 350.0),
    (4, 300.0, 400.0),
    (6, 290.0, 400.0)
])
def test_temperature_ladder(enhanced_sampling, n_replicas, temp_min, temp_max):
    """Test temperature ladder generation with different parameters."""
    replicas = enhanced_sampling.setup_replica_exchange(
        n_replicas=n_replicas,
        temp_min=temp_min,
        temp_max=temp_max
    )

    temps = [r.temperature for r in replicas]

    # Check temperature bounds
    assert np.isclose(temps[0], temp_min)
    assert np.isclose(temps[-1], temp_max)

    # Check geometric spacing
    ratios = np.diff(temps) / temps[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-5)
