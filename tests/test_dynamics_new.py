import pytest
import numpy as np
from unittest.mock import Mock, patch
from proteinflex.models.dynamics.simulation import MolecularDynamics
from tests.utils.mock_objects import MockQuantity, create_mock_openmm

@pytest.fixture
def mock_dynamics(mock_openmm):
    with patch('proteinflex.models.dynamics.simulation.Path') as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.unlink = Mock()
        return MolecularDynamics(device='cpu')

# Existing test functions from current test_dynamics.py
def test_platform_selection(mock_dynamics, mock_openmm):
    """Test automatic platform selection and device compatibility"""
    assert mock_dynamics.platform.getName() == 'CPU'

def test_simulation_setup(mock_dynamics, mock_openmm):
    """Test simulation setup and system preparation"""
    simulation, modeller = mock_dynamics.setup_simulation("test.pdb")
    assert simulation is not None
    assert modeller is not None
    system = simulation.system
    assert system.usesPeriodicBoundaryConditions()
    assert system.getNumParticles() > 0

def test_minimize_and_equilibrate(mock_dynamics, mock_openmm):
    """Test energy minimization and equilibration"""
    simulation, _ = mock_dynamics.setup_simulation("test.pdb")
    result = mock_dynamics.minimize_and_equilibrate(simulation)
    assert 'potential_energy' in result
    assert 'kinetic_energy' in result
    assert 'temperature' in result
    assert isinstance(result['potential_energy'], float)
    assert isinstance(result['kinetic_energy'], float)
    assert result['temperature'] > 0

def test_run_dynamics(mock_dynamics, mock_openmm):
    """Test molecular dynamics simulation"""
    simulation, _ = mock_dynamics.setup_simulation("test.pdb")
    mock_dynamics.minimize_and_equilibrate(simulation)
    result = mock_dynamics.run_dynamics(simulation, steps=100)
    assert 'potential_energy' in result
    assert 'kinetic_energy' in result
    assert 'temperature' in result
    assert 'positions' in result
    assert isinstance(result['positions'], np.ndarray)
    assert len(result['positions'].shape) == 2

def test_trajectory_analysis(mock_dynamics):
    """Test trajectory analysis"""
    n_frames, n_atoms = 10, 5
    positions = np.random.rand(n_frames, n_atoms, 3)
    result = mock_dynamics.analyze_trajectory(positions)
    assert 'rmsd' in result
    assert 'average_structure' in result
    assert 'structure_variance' in result
    assert isinstance(result['rmsd'], float)
    assert result['average_structure'].shape == (n_atoms, 3)
    assert result['structure_variance'].shape == (n_atoms, 3)

def test_error_handling(mock_dynamics, mock_openmm):
    """Test error handling"""
    with pytest.raises(Exception):
        mock_dynamics.setup_simulation("invalid.pdb")
    with pytest.raises(Exception):
        mock_dynamics.minimize_and_equilibrate(None)
    with pytest.raises(Exception):
        mock_dynamics.analyze_trajectory(None)
