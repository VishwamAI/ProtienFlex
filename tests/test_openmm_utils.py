import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from Bio.PDB import Structure, Model, Chain, Residue, Atom
from proteinflex.models.utils.openmm_utils import (
    setup_openmm_system,
    minimize_structure,
    calculate_contact_map,
    calculate_structure_confidence,
    setup_simulation,
    minimize_and_equilibrate
)
from tests.utils import setup_mock_environment, MockQuantity

# Set up mock OpenMM environment
mock_openmm = setup_mock_environment()

@pytest.fixture
def mock_pdb():
    """Create mock PDB with proper topology"""
    mock_pdb = Mock()

    # Create mock topology with proper residue handling
    mock_topology = Mock()

    # Create mock atoms with proper indices
    atoms = []
    for i in range(3):
        mock_atom = Mock()
        mock_atom.index = i
        atoms.append(mock_atom)

    # Create mock residues with proper atom assignments
    residues = []
    # First residue with atoms 0 and 1
    mock_residue1 = Mock()
    mock_residue1.atoms.return_value = [atoms[0], atoms[1]]
    residues.append(mock_residue1)
    # Second residue with atom 2
    mock_residue2 = Mock()
    mock_residue2.atoms.return_value = [atoms[2]]
    residues.append(mock_residue2)

    # Set up topology with proper residue and atom access
    mock_topology.residues = Mock(return_value=residues)
    mock_topology.atoms = Mock(return_value=atoms)
    # Add method to get number of residues
    mock_topology.residues.return_value = residues
    mock_topology.atoms.return_value = atoms

    # Set up positions with proper units using MockQuantity
    positions = [
        MockQuantity(np.array([1.0, 1.0, 1.0]), MockUnit('angstrom')),
        MockQuantity(np.array([2.0, 2.0, 2.0]), MockUnit('angstrom')),
        MockQuantity(np.array([8.0, 8.0, 8.0]), MockUnit('angstrom'))
    ]

    mock_pdb.topology = mock_topology
    mock_pdb.positions = positions

    return mock_pdb

@pytest.fixture
def mock_system():
    system = Mock()
    system.getNumParticles.return_value = 3
    return system

@pytest.fixture
def mock_force_field():
    force_field = Mock()
    force_field.createSystem.return_value = Mock()
    return force_field

@pytest.fixture
def sample_pdb_string():
    return """
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   2.000   2.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   3.000   3.000  1.00  0.00           C
END
"""

def test_setup_openmm_system(sample_pdb_string, mock_force_field, mock_openmm_utils):
    """Test OpenMM system setup"""
    app = mock_openmm_utils['app']
    mm = mock_openmm_utils['mm']

    with patch('proteinflex.models.utils.openmm_utils.app.PDBFile') as mock_pdb_file, \
         patch('proteinflex.models.utils.openmm_utils.app.ForceField') as mock_ff_class, \
         patch('proteinflex.models.utils.openmm_utils.mm.CustomExternalForce') as mock_force:

        # Create mock PDB with proper position attributes
        mock_pdb = Mock()
        mock_pdb.topology = Mock()
        positions = []
        for i in range(3):
            pos = Mock()
            pos.x = float(i)
            pos.y = float(i)
            pos.z = float(i)
            positions.append(pos)
        mock_pdb.positions = positions
        mock_pdb_file.return_value = mock_pdb

        # Set up force field mock
        mock_ff_class.return_value = mock_force_field
        mock_force.return_value = Mock(
            addGlobalParameter=Mock(),
            addPerParticleParameter=Mock(),
            addParticle=Mock()
        )

        system, pdb = setup_openmm_system(sample_pdb_string)

        assert system is not None
        assert pdb is not None
        assert mock_ff_class.called
        assert mock_force.called

def test_setup_openmm_system_error():
    """Test error handling in system setup"""
    with patch('proteinflex.models.utils.openmm_utils.app.PDBFile') as mock_pdb_file:
        mock_pdb_file.side_effect = Exception("PDB file error")
        system, pdb = setup_openmm_system("invalid pdb")
        assert system is None
        assert pdb is None

def test_minimize_structure(mock_system, mock_pdb, mock_openmm_utils):
    """Test structure minimization"""
    mm = mock_openmm_utils['mm']
    unit = mock_openmm_utils['unit']

    # Set up mock positions with proper units
    positions = [unit.Quantity(np.array([1.0, 1.0, 1.0]), unit.angstrom) for _ in range(3)]
    mock_pdb.positions = positions

    # Create mock simulation
    mock_simulation = mm.Simulation.return_value
    mock_simulation.context.getState.return_value.getPositions.return_value = positions
    mock_simulation.context.getState.return_value.getPotentialEnergy.return_value = unit.Quantity(0.0, unit.kilocalories_per_mole)

    minimized_positions = minimize_structure(mock_system, positions)
    assert minimized_positions is not None
    assert len(minimized_positions) == len(positions)

def test_minimize_structure_error():
    """Test error handling in structure minimization"""
    positions = minimize_structure(None, None)
    assert positions is None

def test_calculate_contact_map(mock_pdb, mock_openmm_utils):
    """Test contact map calculation"""
    unit = mock_openmm_utils['unit']

    # Create positions with proper units
    positions = [
        unit.Quantity(np.array([1.0, 1.0, 1.0]), unit.angstrom),
        unit.Quantity(np.array([2.0, 2.0, 2.0]), unit.angstrom),
        unit.Quantity(np.array([8.0, 8.0, 8.0]), unit.angstrom)
    ]
    mock_pdb.positions = positions

    contact_map = calculate_contact_map(positions, mock_pdb.topology, cutoff=5.0)
    assert contact_map is not None
    assert isinstance(contact_map, np.ndarray)
    assert contact_map.shape == (3, 3)
    # First two atoms should be in contact (distance < 5.0 Å)
    assert contact_map[0, 1] == 1
    assert contact_map[1, 0] == 1
    # Third atom should not be in contact with others (distance > 5.0 Å)
    assert contact_map[0, 2] == 0
    assert contact_map[2, 0] == 0

def test_calculate_contact_map_error():
    """Test error handling in contact map calculation"""
    contact_map = calculate_contact_map(None, None)
    assert contact_map is None

def test_calculate_structure_confidence():
    """Test structure confidence calculation"""
    # Create a simple contact map with known density
    contact_map = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])

    confidence = calculate_structure_confidence(contact_map, sequence_length=3)
    assert confidence is not None
    assert isinstance(confidence, float)
    # Contact density is 5/9 ≈ 55.56%
    assert 50.0 <= confidence <= 60.0

def test_calculate_structure_confidence_error():
    """Test error handling in structure confidence calculation"""
    confidence = calculate_structure_confidence(None, 0)
    assert confidence == 50.0  # Default confidence for error case

@pytest.mark.parametrize("cutoff,expected_contacts", [
    (5.0, 1),  # Close atoms should form contacts
    (1.0, 0),  # Far atoms should not form contacts
])
def test_contact_map_cutoff(mock_pdb, mock_openmm_utils, cutoff, expected_contacts):
    """Test contact map calculation with different cutoffs"""
    unit = mock_openmm_utils['unit']
    positions = [
        unit.Quantity(np.array([1.0, 1.0, 1.0]), unit.angstrom),
        unit.Quantity(np.array([2.0, 2.0, 2.0]), unit.angstrom),
        unit.Quantity(np.array([8.0, 8.0, 8.0]), unit.angstrom)
    ]
    mock_pdb.positions = positions

    contact_map = calculate_contact_map(positions, mock_pdb.topology, cutoff=cutoff)
    assert contact_map is not None
    assert isinstance(contact_map, np.ndarray)
    total_contacts = np.sum(contact_map) / 2  # Divide by 2 to account for symmetry
    assert total_contacts == expected_contacts

def test_structure_confidence_scaling():
    """Test confidence score scaling"""
    # Test different contact densities
    test_cases = [
        (np.zeros((10, 10)), 50.0),     # No contacts -> baseline confidence
        (np.ones((10, 10)), 100.0),     # All contacts -> maximum confidence
        (np.eye(10), 55.56)             # Only diagonal contacts (~10% density)
    ]

    for contact_map, expected in test_cases:
        confidence = calculate_structure_confidence(contact_map, 10)
        # Allow small numerical differences
        assert abs(confidence - expected) < 0.1, f"Expected {expected}, got {confidence}"

def test_setup_simulation(mock_pdb, mock_openmm_utils):
    """Test simulation setup with protein structure"""
    # Get mock OpenMM components
    mm = mock_openmm_utils['mm']
    app = mock_openmm_utils['app']
    unit = mock_openmm_utils['unit']

    # Create a simple Bio.PDB structure
    structure = Structure.Structure('test')
    model = Model.Model(0)
    chain = Chain.Chain('A')
    residue = Residue.Residue((' ', 1, ' '), 'ALA', '')

    # Add atoms to create a simple alanine
    atom_n = Atom.Atom('N', np.array([1.0, 1.0, 1.0]), 20.0, 1.0, ' ', 'N', 1, 'N')
    atom_ca = Atom.Atom('CA', np.array([2.0, 1.0, 1.0]), 20.0, 1.0, ' ', 'CA', 2, 'C')
    atom_c = Atom.Atom('C', np.array([3.0, 1.0, 1.0]), 20.0, 1.0, ' ', 'C', 3, 'C')

    for atom in [atom_n, atom_ca, atom_c]:
        residue.add(atom)
    chain.add(residue)
    model.add(chain)
    structure.add(model)

    # Test simulation setup
    simulation = setup_simulation(structure)
    assert simulation is not None
    assert hasattr(simulation, 'context')
    assert hasattr(simulation, 'system')
    assert hasattr(simulation, 'integrator')

def test_setup_simulation_error(mock_openmm_utils):
    """Test error handling in simulation setup"""
    with pytest.raises(ValueError):
        setup_simulation(None)

def test_minimize_and_equilibrate(mock_pdb, mock_openmm_utils):
    """Test minimization and equilibration"""
    # Get mock OpenMM components
    mm = mock_openmm_utils['mm']
    unit = mock_openmm_utils['unit']

    # Create mock simulation
    mock_simulation = mm.Simulation.return_value
    mock_simulation.context.getState.return_value.getPositions.return_value = [
        unit.Quantity(np.array([1.0, 1.0, 1.0]), unit.angstrom) for _ in range(3)
    ]
    mock_simulation.context.getState.return_value.getPotentialEnergy.return_value = unit.Quantity(0.0, unit.kilocalories_per_mole)

    # Test minimization and equilibration
    state = minimize_and_equilibrate(mock_simulation)
    assert state is not None
    assert hasattr(state, 'getPositions')
    assert hasattr(state, 'getPotentialEnergy')

def test_minimize_and_equilibrate_error():
    """Test error handling in minimization and equilibration"""
    with pytest.raises(ValueError):
        minimize_and_equilibrate(None)
