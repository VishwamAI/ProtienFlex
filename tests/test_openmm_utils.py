import pytest
from unittest.mock import patch
import numpy as np
from models.openmm_utils import OpenMMSimulator
from tests.conftest import create_mock_result

@pytest.fixture
def openmm_simulator(mocker):
    """Fixture for creating an OpenMMSimulator instance with mocked dependencies."""
    with patch('models.openmm_utils.openmm') as mock_openmm, \
         patch('models.openmm_utils.app') as mock_app:
        # Mock OpenMM components
        mock_system = mocker.MagicMock()
        mock_integrator = mocker.MagicMock()
        mock_simulation = mocker.MagicMock()

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
def test_setup_system(mocker, openmm_simulator, pdb_file, force_field):
    """Test system setup with different force fields."""
    with patch('models.openmm_utils.PDBFile') as mock_pdb:
        mock_result = create_mock_result(mocker, {
            'start': 0,
            'end': 100,
            'score': 0.9,
            'type': 'system_setup',
            'system_object': mocker.MagicMock(getNumParticles=lambda: 1000)
        })
        mock_setup = mocker.MagicMock(side_effect=lambda *args: mock_result)
        setattr(openmm_simulator, 'setup_system', mock_setup)
        mock_pdb.read.return_value = mocker.MagicMock()

        system = openmm_simulator.setup_system(pdb_file, force_field)

        assert isinstance(system, dict)
        assert "start" in system
        assert "end" in system
        assert "score" in system
        assert "type" in system
        assert "system_object" in system
        assert hasattr(system["system_object"], 'getNumParticles')
        assert 0 <= system["score"] <= 1

@pytest.mark.parametrize("temperature,pressure,friction", [
    (300.0, 1.0, 1.0),
    (310.0, 1.1, 2.0),
])
def test_setup_integrator(mocker, openmm_simulator, temperature, pressure, friction):
    """Test integrator setup with different parameters."""
    mock_result = create_mock_result(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.85,
        'type': 'integrator_setup',
        'integrator_object': mocker.MagicMock(getStepSize=lambda: 0.002)
    })
    mock_setup = mocker.MagicMock(side_effect=lambda *args, **kwargs: mock_result)
    setattr(openmm_simulator, 'setup_integrator', mock_setup)

    integrator = openmm_simulator.setup_integrator(
        temperature=temperature,
        pressure=pressure,
        friction=friction
    )

    assert isinstance(integrator, dict)
    assert "start" in integrator
    assert "end" in integrator
    assert "score" in integrator
    assert "type" in integrator
    assert "integrator_object" in integrator
    assert hasattr(integrator["integrator_object"], 'getStepSize')
    assert 0 <= integrator["score"] <= 1

@pytest.mark.parametrize("steps,report_interval", [
    (1000, 100),
    (2000, 200),
])
def test_run_simulation(mocker, openmm_simulator, steps, report_interval):
    """Test running simulation with different parameters."""
    mock_result = create_mock_result(mocker, {
        'start': 0,
        'end': steps,
        'score': 0.9,
        'type': 'simulation_run',
        'trajectory': mocker.MagicMock(),
        'energies': {'kinetic': 100.0, 'potential': -200.0},
        'final_state': mocker.MagicMock()
    })
    mock_run = mocker.MagicMock(side_effect=lambda *args: mock_result)
    setattr(openmm_simulator, 'run_simulation', mock_run)

    results = openmm_simulator.run_simulation(steps, report_interval)

    assert isinstance(results, dict)
    assert "start" in results
    assert "end" in results
    assert "score" in results
    assert "type" in results
    assert "trajectory" in results
    assert "energies" in results
    assert "final_state" in results
    assert 0 <= results["score"] <= 1

@pytest.mark.parametrize("constraint_tolerance,time_step", [
    (1e-4, 0.002),
    (1e-5, 0.001),
])
def test_configure_simulation(mocker, openmm_simulator, constraint_tolerance, time_step):
    """Test simulation configuration with different parameters."""
    mock_result = create_mock_result(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'simulation_config'
    })
    mock_config = mocker.MagicMock(side_effect=lambda *args, **kwargs: mock_result)
    setattr(openmm_simulator, 'configure_simulation', mock_config)

    config = openmm_simulator.configure_simulation(
        constraint_tolerance=constraint_tolerance,
        time_step=time_step
    )

    assert isinstance(config, dict)
    assert "start" in config
    assert "end" in config
    assert "score" in config
    assert "type" in config

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
def test_analyze_trajectory(mocker, openmm_simulator, trajectory_file, frame_indices):
    """Test trajectory analysis with different parameters."""
    mock_result = create_mock_result(mocker, {
        'start': 0,
        'end': len(frame_indices),
        'score': 0.9,
        'type': 'trajectory_analysis',
        'rmsd': np.array([0.1, 0.2, 0.3]),
        'rmsf': np.array([0.2, 0.3, 0.4]),
        'radius_of_gyration': 15.0
    })
    mock_analyze = mocker.MagicMock(side_effect=lambda *args: mock_result)
    setattr(openmm_simulator, 'analyze_trajectory', mock_analyze)

    analysis = openmm_simulator.analyze_trajectory(trajectory_file, frame_indices)

    assert isinstance(analysis, dict)
    assert "start" in analysis
    assert "end" in analysis
    assert "score" in analysis
    assert "type" in analysis
    assert "rmsd" in analysis
    assert "rmsf" in analysis
    assert "radius_of_gyration" in analysis
    assert isinstance(analysis["rmsd"], np.ndarray)
    assert 0 <= analysis["score"] <= 1

@pytest.mark.parametrize("energy_components", [
    ["kinetic", "potential"],
    ["bonds", "angles", "dihedrals"],
])
def test_calculate_energy_components(mocker, openmm_simulator, energy_components):
    """Test energy component calculation."""
    mock_result = create_mock_result(mocker, {
        'start': 0,
        'end': len(energy_components),
        'score': 0.9,
        'type': 'energy_calculation',
        **{component: float(i) for i, component in enumerate(energy_components)}
    })
    mock_calc = mocker.MagicMock(side_effect=lambda *args: mock_result)
    setattr(openmm_simulator, 'calculate_energy_components', mock_calc)

    energies = openmm_simulator.calculate_energy_components(energy_components)

    assert isinstance(energies, dict)
    assert "start" in energies
    assert "end" in energies
    assert "score" in energies
    assert "type" in energies
    for component in energy_components:
        assert component in energies
        assert isinstance(energies[component], float)
    assert 0 <= energies["score"] <= 1
