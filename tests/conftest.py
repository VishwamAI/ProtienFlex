"""Initialize test environment with system-wide mocks."""
import sys
from unittest.mock import Mock, MagicMock, patch
import pytest
import numpy as np

# Import mock creation functions
from .utils.mock_objects import (
    create_mock_esm_model,
    create_mock_batch_converter,
    create_mock_transformers,
    create_mock_rdkit,
    create_mock_openmm,
    MockOpenMMSimulation,  # Add MockOpenMMSimulation import
)

# Create mock environment
mock_env = create_mock_openmm()

# Mock OpenMM and related modules
mock_openmm = mock_env['openmm']
mock_mm = mock_env['mm']
mock_app = mock_env['app']
mock_unit = mock_env['unit']

sys.modules['openmm'] = mock_openmm
sys.modules['openmm.app'] = mock_app
sys.modules['openmm.unit'] = mock_unit
sys.modules['openmm.mm'] = mock_mm

# Mock ESM
sys.modules['esm'] = create_mock_esm_model()
sys.modules['esm.pretrained'] = Mock(load=lambda *args, **kwargs: (sys.modules['esm'], create_mock_batch_converter()))

# Mock RDKit
sys.modules['rdkit'] = create_mock_rdkit()
sys.modules['rdkit.Chem'] = sys.modules['rdkit'].Chem
sys.modules['rdkit.Chem.AllChem'] = Mock()
sys.modules['rdkit.Chem.Descriptors'] = Mock()

# Mock Transformers
sys.modules['transformers'] = create_mock_transformers()

@pytest.fixture
def mock_torch():
    """Provide mock torch functionality."""
    import torch

    class MockTensor:
        def __init__(self, data=None, device='cpu'):
            self.data = data if data is not None else []
            self.device = device
            self.requires_grad = False

        def clone(self):
            return MockTensor(self.data, self.device)

        def to(self, device):
            self.device = device
            return self

        def __getitem__(self, idx):
            if isinstance(idx, tuple):
                return MockTensor()
            return self.data[idx] if isinstance(self.data, list) else MockTensor()

        def __setitem__(self, idx, value):
            if isinstance(self.data, list):
                self.data[idx] = value

    with patch('torch.device') as mock_device:
        mock_device.return_value = 'cpu'
        with patch('torch.cuda.is_available') as mock_cuda:
            mock_cuda.return_value = False
            with patch('torch.tensor') as mock_tensor:
                mock_tensor.return_value = MockTensor()
                yield mock_tensor

@pytest.fixture
def mock_numpy():
    """Provide mock numpy functionality."""
    with patch('numpy.array') as mock_array:
        mock_array.return_value = np.zeros((10, 10))
        yield mock_array

@pytest.fixture
def mock_biopython():
    """Provide mock BioPython functionality."""
    mock_bio = Mock()
    mock_bio.PDB = Mock()
    mock_bio.PDB.Structure = Mock()
    mock_bio.PDB.Model = Mock()
    mock_bio.PDB.Chain = Mock()
    mock_bio.PDB.Residue = Mock()
    mock_bio.PDB.Atom = Mock()
    mock_bio.PDB.PDBIO = Mock()
    mock_bio.PDB.StructureBuilder = Mock()

    with patch.dict('sys.modules', {'Bio': mock_bio}):
        yield mock_bio

@pytest.fixture
def mock_esm_model():
    """Provide mock ESM model."""
    yield sys.modules['esm']

@pytest.fixture
def mock_batch_converter():
    """Provide mock batch converter."""
    yield create_mock_batch_converter()

@pytest.fixture
def mock_rdkit_utils():
    """Provide mock RDKit utilities."""
    yield sys.modules['rdkit']

@pytest.fixture
def mock_transformers_utils():
    """Provide mock transformers utilities."""
    yield sys.modules['transformers']

@pytest.fixture
def mock_openmm_utils():
    """Provide mock OpenMM utilities."""
    yield {
        'mm': mock_mm,
        'app': mock_app,
        'unit': mock_unit,
        'openmm': mock_openmm
    }

@pytest.fixture
def mock_openmm():
    """Provide mock OpenMM environment for dynamics tests."""
    # Set up platform configuration
    mock_mm.Platform.getNumPlatforms.return_value = 3
    platforms = ['Reference', 'CPU', 'CUDA']
    mock_mm.Platform.getPlatform.side_effect = \
        lambda i: Mock(getName=lambda: platforms[i])

    # Mock system and force field setup
    mock_system = Mock()
    mock_system.getNumForces.return_value = 3
    mock_system.usesPeriodicBoundaryConditions.return_value = True
    mock_system.getNumParticles.return_value = 100
    mock_system.addForce = Mock(return_value=0)

    # Mock force field
    mock_app.ForceField.return_value.createSystem.return_value = mock_system

    # Mock simulation components
    mock_context = Mock()
    mock_state = Mock()
    mock_state.getPotentialEnergy.return_value = mock_unit.Quantity(value=-1000.0, unit='kilojoules/mole')
    mock_state.getKineticEnergy.return_value = mock_unit.Quantity(value=500.0, unit='kilojoules/mole')
    mock_state.getPositions.return_value = mock_unit.Quantity(value=np.zeros((100, 3)), unit='nanometers')
    mock_state.getPeriodicBoxVectors.return_value = [
        mock_unit.Quantity(value=[4.0, 0.0, 0.0], unit='nanometers'),
        mock_unit.Quantity(value=[0.0, 4.0, 0.0], unit='nanometers'),
        mock_unit.Quantity(value=[0.0, 0.0, 4.0], unit='nanometers')
    ]
    mock_context.getState.return_value = mock_state

    # Create a mock simulation instance with enhanced error handling
    mock_simulation = MockOpenMMSimulation()
    mock_simulation.context = mock_context
    mock_simulation.system = mock_system
    mock_simulation.minimizeEnergy = Mock(side_effect=lambda maxIterations=0, tolerance=0: None)
    mock_simulation.step = Mock(side_effect=lambda steps: None)

    # Mock topology for atom iteration
    mock_topology = Mock()
    mock_atom = Mock()
    mock_atom.element = Mock()
    mock_atom.element.mass = mock_unit.Quantity(12.0, 'daltons')
    mock_topology.atoms = Mock(return_value=[mock_atom] * 100)
    mock_simulation.topology = mock_topology

    # Update app.Simulation to return our configured mock
    mock_app.Simulation = Mock(return_value=mock_simulation)

    # Mock HarmonicBondForce for restraints
    mock_force = Mock()
    mock_force.setUsesPeriodicBoundaryConditions = Mock()
    mock_force.addBond = Mock()
    mock_force.getNumBonds = Mock(return_value=100)
    mock_force.getBondParameters = Mock(return_value=(0, 0, 0.0, 1000.0))
    mock_force.setBondParameters = Mock()
    mock_force.updateParametersInContext = Mock()
    mock_mm.HarmonicBondForce = Mock(return_value=mock_force)

    yield {
        'mm': mock_mm,
        'app': mock_app,
        'unit': mock_unit,
        'openmm': mock_openmm
    }
