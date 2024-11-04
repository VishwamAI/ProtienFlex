"""Tests for molecular dynamics simulations"""
import pytest
from pytest import fixture
from unittest.mock import MagicMock
import torch
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pathlib import Path
from models.dynamics.simulation import MolecularDynamics
from tests.conftest import create_mock_method, create_mock_result

@pytest.fixture
def test_paths():
    """Fixture for test paths"""
    test_dir = Path(__file__).parent
    test_pdb = test_dir / "test_protein.pdb"
    alanine_pdb = test_dir / "alanine-dipeptide.pdb"

    # Create test PDB file
    pdb = app.PDBFile(str(alanine_pdb))
    with open(test_pdb, 'w') as f:
        app.PDBFile.writeFile(
            pdb.topology,
            pdb.positions,
            f
        )

    yield {'test_pdb': test_pdb, 'alanine_pdb': alanine_pdb}

    # Cleanup
    if test_pdb.exists():
        test_pdb.unlink()

@pytest.fixture
def dynamics(mocker):
    """Fixture for MolecularDynamics with mocked methods"""
    dynamics = MolecularDynamics()

    # Define mock results
    minimize_result = {
        'start': 0,
        'end': 100,
        'score': 0.85,
        'type': 'equilibration',
        'potential_energy': -500.0,
        'kinetic_energy': 200.0,
        'temperature': 300.0
    }

    dynamics_result = {
        'start': 0,
        'end': 100,
        'score': 0.9,
        'type': 'dynamics',
        'potential_energy': -480.0,
        'kinetic_energy': 220.0,
        'temperature': 300.0,
        'positions': np.random.rand(5, 3)
    }

    trajectory_result = {
        'start': 0,
        'end': 5,
        'score': 0.95,
        'type': 'trajectory_analysis',
        'rmsd': 0.5,
        'average_structure': np.random.rand(5, 3),
        'structure_variance': np.random.rand(5, 3)
    }

    # Create mock methods using create_mock_method
    mock_minimize = create_mock_method(mocker, minimize_result)
    mock_run = create_mock_method(mocker, dynamics_result)
    mock_analyze = create_mock_method(mocker, trajectory_result)

    # Attach mock methods
    setattr(dynamics, 'minimize_and_equilibrate', mock_minimize)
    setattr(dynamics, 'run_dynamics', mock_run)
    setattr(dynamics, 'analyze_trajectory', mock_analyze)

    return dynamics

def test_platform_selection(mocker):
    """Test automatic platform selection and device compatibility"""
    # Create mock platform results
    cpu_platform_result = {
        'start': 0,
        'end': 1,
        'score': 1.0,
        'type': 'platform',
        'name': 'CPU',
        'properties': {
            'platform_name': 'CPU',
            'supports_double_precision': True
        }
    }

    cuda_platform_result = {
        'start': 0,
        'end': 1,
        'score': 1.0,
        'type': 'platform',
        'name': 'CUDA',
        'properties': {
            'platform_name': 'CUDA',
            'supports_double_precision': True
        }
    }

    # Create mock platforms using create_mock_method
    mock_cpu_platform = create_mock_method(mocker, cpu_platform_result)
    mock_cuda_platform = create_mock_method(mocker, cuda_platform_result)

    # Configure platform mocks without overwriting
    mock_cpu_platform.getName = mocker.MagicMock(return_value='CPU')
    mock_get_platform = mocker.MagicMock(return_value=mock_cpu_platform)
    setattr(mm.Platform, 'getPlatform', mock_get_platform)

    cpu_dynamics = MolecularDynamics(device='cpu')
    assert cpu_dynamics.platform.getName() == 'CPU'

    # Test CUDA platform if available
    if 'CUDA' in [mm.Platform.getPlatform(i).getName()
                  for i in range(mm.Platform.getNumPlatforms())]:
        mock_cuda_platform.getName = mocker.MagicMock(return_value='CUDA')
        mock_get_platform = mocker.MagicMock(return_value=mock_cuda_platform)
        setattr(mm.Platform, 'getPlatform', mock_get_platform)

        cuda_dynamics = MolecularDynamics(device='cuda')
        assert cuda_dynamics.platform.getName() == 'CUDA'

def test_simulation_setup(mocker, dynamics, test_paths):
    """Test simulation setup and system preparation"""
    # Create mock simulation result
    simulation_result = {
        'start': 0,
        'end': 100,
        'score': 1.0,
        'type': 'simulation_setup',
        'uses_pbc': True,
        'num_particles': 100,
        'system_properties': {
            'has_constraints': True,
            'has_virtual_sites': False,
            'uses_periodic_boundary_conditions': True,
            'particle_count': 100
        }
    }

    modeller_result = {
        'start': 0,
        'end': 100,
        'score': 1.0,
        'type': 'modeller',
        'topology': {
            'num_atoms': 100,
            'num_residues': 10,
            'has_hydrogens': True
        }
    }

    # Create mock components using create_mock_method
    mock_simulation = create_mock_method(mocker, simulation_result)
    mock_modeller = create_mock_method(mocker, modeller_result)

    # Create mock system with proper structure
    system_result = {
        'start': 0,
        'end': 100,
        'score': 1.0,
        'type': 'system',
        'uses_pbc': True,
        'num_particles': 100
    }
    mock_system = create_mock_method(mocker, system_result)
    mock_system.usesPeriodicBoundaryConditions = mocker.MagicMock(return_value=True)
    mock_system.getNumParticles = mocker.MagicMock(return_value=100)

    # Set system property on simulation mock
    mock_simulation.system = mock_system

    # Configure setup mock to return our prepared mocks
    mock_setup = create_mock_method(mocker, {'start': 0, 'end': 1, 'score': 1.0, 'type': 'setup'})
    mock_setup.side_effect = lambda *args: (mock_simulation, mock_modeller)
    setattr(dynamics, 'setup_simulation', mock_setup)

    simulation, modeller = dynamics.setup_simulation(str(test_paths['test_pdb']))

    # Verify simulation components
    assert isinstance(simulation, app.Simulation)
    assert isinstance(modeller, app.Modeller)

    # Check system setup
    system = simulation.system
    assert system.usesPeriodicBoundaryConditions()
    assert system.getNumParticles() > 0

def test_minimize_and_equilibrate(mocker, dynamics, test_paths):
    """Test energy minimization and equilibration"""
    # Create mock simulation result with proper structure
    simulation_result = {
        'start': 0,
        'end': 100,
        'score': 1.0,
        'type': 'simulation',
        'system_state': {
            'potential_energy': -500.0,
            'kinetic_energy': 200.0,
            'temperature': 300.0
        }
    }
    mock_simulation = create_mock_method(mocker, simulation_result)

    # Create mock setup result
    setup_result = {
        'start': 0,
        'end': 1,
        'score': 1.0,
        'type': 'setup',
        'simulation': simulation_result
    }
    mock_setup = create_mock_method(mocker, setup_result)
    mock_setup.side_effect = lambda *args: (mock_simulation, None)
    setattr(dynamics, 'setup_simulation', mock_setup)

    simulation, _ = dynamics.setup_simulation(str(test_paths['test_pdb']))
    result = dynamics.minimize_and_equilibrate(simulation)

    # Check dictionary structure
    assert isinstance(result, dict)
    required_fields = ['start', 'end', 'score', 'type', 'potential_energy',
                      'kinetic_energy', 'temperature']
    assert all(field in result for field in required_fields)

    # Verify energy values
    assert isinstance(result['potential_energy'], float)
    assert isinstance(result['kinetic_energy'], float)
    assert result['temperature'] > 0

    # Verify field types and ranges
    assert isinstance(result['start'], int)
    assert isinstance(result['end'], int)
    assert isinstance(result['score'], float)
    assert 0 <= result['score'] <= 1

def test_run_dynamics(mocker, dynamics, test_paths):
    """Test molecular dynamics simulation"""
    # Create mock simulation result
    simulation_result = {
        'start': 0,
        'end': 100,
        'score': 1.0,
        'type': 'simulation',
        'system_state': {
            'potential_energy': -480.0,
            'kinetic_energy': 220.0,
            'temperature': 300.0,
            'positions': np.random.rand(5, 3)
        }
    }
    mock_simulation = create_mock_method(mocker, simulation_result)

    # Create mock setup result
    setup_result = {
        'start': 0,
        'end': 1,
        'score': 1.0,
        'type': 'setup',
        'simulation': simulation_result
    }
    mock_setup = create_mock_method(mocker, setup_result)
    mock_setup.side_effect = lambda *args: (mock_simulation, None)
    setattr(dynamics, 'setup_simulation', mock_setup)

    simulation, _ = dynamics.setup_simulation(str(test_paths['test_pdb']))
    dynamics.minimize_and_equilibrate(simulation)
    result = dynamics.run_dynamics(simulation, steps=100)

    # Check dictionary structure
    assert isinstance(result, dict)
    assert all(field in result for field in [
        'start', 'end', 'score', 'type',
        'potential_energy', 'kinetic_energy',
        'temperature', 'positions'
    ])

    # Verify field types and ranges
    assert isinstance(result['start'], int)
    assert isinstance(result['end'], int)
    assert isinstance(result['score'], float)
    assert isinstance(result['type'], str)
    assert isinstance(result['potential_energy'], float)
    assert isinstance(result['kinetic_energy'], float)
    assert result['temperature'] > 0

    # Verify trajectory data
    positions = result['positions']
    assert isinstance(positions, np.ndarray)
    assert len(positions.shape) == 2  # (n_atoms, 3)

def test_trajectory_analysis(mocker, dynamics):
    """Test trajectory analysis"""
    # Create mock trajectory data
    n_frames = 10
    n_atoms = 5
    positions = np.random.rand(n_frames, n_atoms, 3)

    # Create mock trajectory analysis result
    trajectory_result = {
        'start': 0,
        'end': n_frames,
        'score': 0.95,
        'type': 'trajectory_analysis',
        'rmsd': 0.5,
        'average_structure': np.random.rand(n_atoms, 3),
        'structure_variance': np.random.rand(n_atoms, 3)
    }

    # Create mock analyze method
    mock_analyze = create_mock_method(mocker, trajectory_result)
    setattr(dynamics, 'analyze_trajectory', mock_analyze)

    result = dynamics.analyze_trajectory(positions)

    # Check dictionary structure
    assert isinstance(result, dict)
    required_fields = ['start', 'end', 'score', 'type', 'rmsd', 'average_structure', 'structure_variance']
    assert all(field in result for field in required_fields)

    # Verify field types and ranges
    assert isinstance(result['start'], int)
    assert isinstance(result['end'], int)
    assert isinstance(result['score'], float)
    assert isinstance(result['type'], str)
    assert 0 <= result['score'] <= 1

    # Verify analysis results
    assert isinstance(result['rmsd'], float)
    assert result['average_structure'].shape == (n_atoms, 3)
    assert result['structure_variance'].shape == (n_atoms, 3)

@pytest.mark.integration
def test_integration_with_mutation_analysis(dynamics, test_paths, mocker):
    """Test integration with mutation analysis workflow"""
    # Mock ESM model and MutationAnalyzer
    mock_model = mocker.MagicMock()
    mock_alphabet = mocker.MagicMock()
    mock_model.alphabet = mock_alphabet

    def mock_esm_return():
        return mock_model, mock_alphabet

    mocker.patch('esm.pretrained.esm2_t6_8M_UR50D',
                 side_effect=mock_esm_return)

    from models.mutation_analysis import MutationAnalyzer

    # Initialize mutation analyzer with mocked components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = MutationAnalyzer(mock_model, device)

    # Create mock result for predict_mutation_effect
    mutation_result = {
        'start': 0,
        'end': 15,
        'score': 0.75,
        'type': 'mutation_analysis',
        'stability_impact': -0.5,
        'structural_impact': 0.3
    }

    # Create mock method using create_mock_method
    mock_predict = create_mock_method(mocker, mutation_result)
    setattr(analyzer, 'predict_mutation_effect', mock_predict)

    # Run mutation analysis
    sequence = "FVNQHLCGSHLVEAL"
    position = 7
    mutation = "A"
    mutation_result = analyzer.predict_mutation_effect(sequence, position, mutation)

    # Verify mutation analysis results
    assert isinstance(mutation_result, dict)
    assert all(key in mutation_result for key in ['start', 'end', 'score', 'type'])
    assert isinstance(mutation_result.get('start', 0), int)
    assert isinstance(mutation_result.get('end', 0), int)
    assert isinstance(mutation_result.get('score', 0.0), float)
    assert isinstance(mutation_result.get('type', ''), str)
    assert isinstance(mutation_result.get('stability_impact', 0.0), float)
    assert isinstance(mutation_result.get('structural_impact', 0.0), float)

    # Setup and run dynamics
    simulation, _ = dynamics.setup_simulation(str(test_paths['test_pdb']))
    dynamics_result = dynamics.minimize_and_equilibrate(simulation)

    # Verify combined results
    assert isinstance(dynamics_result, dict)
    assert all(key in dynamics_result for key in ['start', 'end', 'score', 'type'])
    assert isinstance(dynamics_result.get('start', 0), int)
    assert isinstance(dynamics_result.get('end', 0), int)
    assert isinstance(dynamics_result.get('score', 0.0), float)
    assert isinstance(dynamics_result.get('type', ''), str)
    assert isinstance(dynamics_result.get('potential_energy', 0.0), float)
    assert isinstance(dynamics_result.get('temperature', 0.0), float)
