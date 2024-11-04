import pytest
from unittest.mock import Mock, patch
import numpy as np
from models.drug_binding import DrugBindingSimulator

@pytest.fixture
def binding_simulator():
    """Fixture for creating a DrugBindingSimulator instance with mocked dependencies."""
    with patch('models.drug_binding.openmm') as mock_openmm, \
         patch('models.drug_binding.app') as mock_app:
        # Mock OpenMM system and force field
        mock_system = Mock()
        mock_forcefield = Mock()
        mock_app.ForceField.return_value = mock_forcefield
        mock_forcefield.createSystem.return_value = mock_system

        simulator = DrugBindingSimulator()
        simulator.system = mock_system
        return simulator

@pytest.mark.parametrize("protein_pdb,ligand_smiles", [
    ("test_protein.pdb", "CC1=CC=C(C=C1)CC(C(=O)O)N"),
    ("test_protein2.pdb", "CC(=O)NC1=CC=C(O)C=C1"),
])
def test_setup_binding_simulation(binding_simulator, protein_pdb, ligand_smiles):
    """Test binding simulation setup with different proteins and ligands."""
    with patch('models.drug_binding.PDBFile') as mock_pdb:
        mock_pdb.read.return_value = Mock()

        result = binding_simulator.setup_binding_simulation(protein_pdb, ligand_smiles)

        assert isinstance(result, dict)
        assert "system" in result
        assert "topology" in result
        assert "positions" in result
        assert "success" in result
        assert result["success"] is True

@pytest.mark.parametrize("protein_sequence,binding_site,ligand_smiles", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", (10, 20), "CC1=CC=C(C=C1)CC(C(=O)O)N"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", (5, 15), "CC(=O)NC1=CC=C(O)C=C1"),
])
def test_analyze_binding_interactions(binding_simulator, protein_sequence, binding_site, ligand_smiles):
    """Test analysis of binding interactions."""
    interactions = binding_simulator.analyze_binding_interactions(
        protein_sequence, binding_site, ligand_smiles
    )

    assert isinstance(interactions, dict)
    assert "hydrogen_bonds" in interactions
    assert "hydrophobic_contacts" in interactions
    assert "ionic_interactions" in interactions
    assert "binding_energy" in interactions
    assert isinstance(interactions["binding_energy"], float)

@pytest.mark.parametrize("protein_sequence,ligand_smiles", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "CC1=CC=C(C=C1)CC(C(=O)O)N"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "CC(=O)NC1=CC=C(O)C=C1"),
])
def test_calculate_binding_energy(binding_simulator, protein_sequence, ligand_smiles):
    """Test binding energy calculation."""
    with patch.object(binding_simulator, '_run_energy_calculation') as mock_calc:
        mock_calc.return_value = -50.0  # Mock energy value

        energy = binding_simulator.calculate_binding_energy(protein_sequence, ligand_smiles)

        assert isinstance(energy, float)
        assert energy < 0  # Binding energy should be negative for favorable interactions

@pytest.mark.parametrize("trajectory_file,frame_count", [
    ("test_trajectory.dcd", 100),
    ("test_trajectory2.dcd", 200),
])
def test_analyze_binding_trajectory(binding_simulator, trajectory_file, frame_count):
    """Test binding trajectory analysis."""
    with patch('models.drug_binding.md') as mock_md:
        mock_md.load.return_value = Mock(n_frames=frame_count)

        analysis = binding_simulator.analyze_binding_trajectory(trajectory_file)

        assert isinstance(analysis, dict)
        assert "rmsd" in analysis
        assert "contact_frequency" in analysis
        assert "residence_time" in analysis
        assert len(analysis["rmsd"]) == frame_count

def test_error_handling(binding_simulator):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        binding_simulator.setup_binding_simulation("", "")

    with pytest.raises(ValueError):
        binding_simulator.analyze_binding_interactions("", (0, 10), "INVALID")

    with pytest.raises(ValueError):
        binding_simulator.calculate_binding_energy("", "")

    with pytest.raises(FileNotFoundError):
        binding_simulator.analyze_binding_trajectory("nonexistent.dcd")


@pytest.mark.parametrize("protein_sequence,ligand_smiles,temperature", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "CC1=CC=C(C=C1)CC(C(=O)O)N", 300),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "CC(=O)NC1=CC=C(O)C=C1", 310),
])
def test_run_binding_simulation(binding_simulator, protein_sequence, ligand_smiles, temperature):
    """Test running binding simulation with different parameters."""
    with patch.object(binding_simulator, '_setup_simulation') as mock_setup:
        mock_setup.return_value = (Mock(), Mock(), Mock())

        results = binding_simulator.run_binding_simulation(
            protein_sequence, ligand_smiles, temperature=temperature
        )

        assert isinstance(results, dict)
        assert "trajectory" in results
        assert "energies" in results
        assert "final_state" in results
        assert isinstance(results["energies"], list)
