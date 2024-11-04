import pytest
from unittest.mock import Mock, patch
import numpy as np
from models.openmm_utils import OpenMMSimulator

@pytest.fixture
def openmm_simulator():
    """Fixture for creating an OpenMMSimulator instance with mocked dependencies."""
    with patch('models.openmm_utils.openmm') as mock_openmm, \
         patch('models.openmm_utils.app') as mock_app:
        # Mock OpenMM components
        mock_system = Mock()
        mock_integrator = Mock()
        mock_simulation = Mock()

        mock_openmm.System.return_value = mock_system
        mock_openmm.LangevinMiddleIntegrator.return_value = mock_integrator
        mock_app.Simulation.return_value = mock_simulation

        simulator = OpenMMSimulator()
        simulator.system = mock_system
        simulator.simulation = mock_simulation
        return simulator

@pytest.mark.parametrize("pdb_file,force_field", [
    ("protein.pdb", "amber14-all.xml"),
    ("complex.pdb", "charmm36.xml"),
])
def test_setup_system(openmm_simulator, pdb_file, force_field):
    """Test system setup with different force fields."""
    with patch('models.openmm_utils.PDBFile') as mock_pdb:
        mock_pdb.read.return_value = Mock()

        system = openmm_simulator.setup_system(pdb_file, force_field)

        assert system is not None
        assert hasattr(system, 'getNumParticles')

@pytest.mark.parametrize("temperature,pressure,friction", [
    (300.0, 1.0, 1.0),
    (310.0, 1.1, 2.0),
])
def test_setup_integrator(openmm_simulator, temperature, pressure, friction):
    """Test integrator setup with different parameters."""
    integrator = openmm_simulator.setup_integrator(
        temperature=temperature,
        pressure=pressure,
        friction=friction
    )

    assert integrator is not None
    assert hasattr(integrator, 'getStepSize')

@pytest.mark.parametrize("steps,report_interval", [
    (1000, 100),
    (2000, 200),
])
def test_run_simulation(openmm_simulator, steps, report_interval):
    """Test running simulation with different parameters."""
    with patch.object(openmm_simulator.simulation, 'step') as mock_step:
        results = openmm_simulator.run_simulation(steps, report_interval)

        assert isinstance(results, dict)
        assert "trajectory" in results
        assert "energies" in results
        assert "final_state" in results
        mock_step.assert_called_with(steps)

@pytest.mark.parametrize("constraint_tolerance,time_step", [
    (1e-4, 0.002),
    (1e-5, 0.001),
])
def test_configure_simulation(openmm_simulator, constraint_tolerance, time_step):
    """Test simulation configuration with different parameters."""
    config = openmm_simulator.configure_simulation(
        constraint_tolerance=constraint_tolerance,
        time_step=time_step
    )

    assert isinstance(config, dict)
    assert "constraint_tolerance" in config
    assert "time_step" in config
    assert config["constraint_tolerance"] == constraint_tolerance

def test_error_handling(openmm_simulator):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        openmm_simulator.setup_system("", "invalid.xml")

    with pytest.raises(ValueError):
        openmm_simulator.setup_integrator(temperature=-1)

    with pytest.raises(ValueError):
        openmm_simulator.run_simulation(-100, 10)

    with pytest.raises(ValueError):
        openmm_simulator.configure_simulation(constraint_tolerance=-1)

@pytest.mark.parametrize("trajectory_file,frame_indices", [
    ("traj.dcd", [0, 10, 20]),
    ("sim.dcd", [5, 15, 25]),
])
def test_analyze_trajectory(openmm_simulator, trajectory_file, frame_indices):
    """Test trajectory analysis with different parameters."""
    with patch('models.openmm_utils.md') as mock_md:
        mock_traj = Mock()
        mock_md.load.return_value = mock_traj

        analysis = openmm_simulator.analyze_trajectory(trajectory_file, frame_indices)

        assert isinstance(analysis, dict)
        assert "rmsd" in analysis
        assert "rmsf" in analysis
        assert "radius_of_gyration" in analysis
        assert isinstance(analysis["rmsd"], np.ndarray)

@pytest.mark.parametrize("energy_components", [
    ["kinetic", "potential"],
    ["bonds", "angles", "dihedrals"],
])
def test_calculate_energy_components(openmm_simulator, energy_components):
    """Test energy component calculation."""
    energies = openmm_simulator.calculate_energy_components(energy_components)

    assert isinstance(energies, dict)
    for component in energy_components:
        assert component in energies
        assert isinstance(energies[component], float)
