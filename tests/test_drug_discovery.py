import pytest
from unittest.mock import Mock, patch
import numpy as np
from models.drug_discovery import DrugDiscoveryEngine

@pytest.fixture
def drug_discovery_engine():
    """Fixture for creating a DrugDiscoveryEngine instance with mocked dependencies."""
    with patch('models.drug_discovery.esm') as mock_esm:
        # Mock ESM model and alphabet
        mock_model = Mock()
        mock_alphabet = Mock()
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_model, mock_alphabet)

        engine = DrugDiscoveryEngine()
        engine.model = mock_model
        engine.alphabet = mock_alphabet
        return engine

@pytest.mark.parametrize("sequence,site_start,site_end,ligand_smiles", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", 10, 20, "CC1=CC=C(C=C1)CC(C(=O)O)N"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", 5, 15, "CC(=O)NC1=CC=C(O)C=C1"),
])
def test_optimize_binding_site(drug_discovery_engine, sequence, site_start, site_end, ligand_smiles):
    """Test binding site optimization with different sequences and ligands."""
    result = drug_discovery_engine.optimize_binding_site(sequence, site_start, site_end, ligand_smiles)

    assert isinstance(result, dict)
    assert "site_analysis" in result
    assert "optimization_suggestions" in result
    assert "optimization_score" in result
    assert "predicted_improvement" in result

    # Validate site analysis
    site_analysis = result["site_analysis"]
    assert isinstance(site_analysis["hydrophobicity"], float)
    assert isinstance(site_analysis["length"], int)
    assert isinstance(site_analysis["residue_properties"], list)

    # Validate optimization suggestions
    suggestions = result["optimization_suggestions"]
    assert isinstance(suggestions, list)
    for suggestion in suggestions:
        assert "type" in suggestion
        assert "issue" in suggestion
        assert "suggestion" in suggestion
        assert "confidence" in suggestion
        assert 0 <= suggestion["confidence"] <= 1

    # Validate scores
    assert 0 <= result["optimization_score"] <= 1
    assert 0 <= result["predicted_improvement"] <= 1

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_analyze_binding_sites(drug_discovery_engine, sequence):
    """Test binding site analysis with different sequences."""
    result = drug_discovery_engine.analyze_binding_sites(sequence)

    assert isinstance(result, list)
    for site in result:
        assert "start" in site
        assert "end" in site
        assert "score" in site
        assert "properties" in site
        assert isinstance(site["score"], float)
        assert 0 <= site["score"] <= 1

@pytest.mark.parametrize("sequence,ligand_smiles", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "CC1=CC=C(C=C1)CC(C(=O)O)N"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "CC(=O)NC1=CC=C(O)C=C1"),
])
def test_predict_drug_interactions(drug_discovery_engine, sequence, ligand_smiles):
    """Test drug interaction prediction with different sequences and ligands."""
    result = drug_discovery_engine.predict_drug_interactions(sequence, ligand_smiles)

    assert isinstance(result, dict)
    assert "binding_affinity" in result
    assert "interaction_sites" in result
    assert "stability_score" in result

    assert isinstance(result["binding_affinity"], float)
    assert isinstance(result["stability_score"], float)
    assert isinstance(result["interaction_sites"], list)

    for site in result["interaction_sites"]:
        assert "position" in site
        assert "type" in site
        assert "score" in site
        assert 0 <= site["score"] <= 1

@pytest.mark.parametrize("sequence,ligand_smiles", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "CC1=CC=C(C=C1)CC(C(=O)O)N"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "CC(=O)NC1=CC=C(O)C=C1"),
])
def test_screen_off_targets(drug_discovery_engine, sequence, ligand_smiles):
    """Test off-target screening with different sequences and ligands."""
    result = drug_discovery_engine.screen_off_targets(sequence, ligand_smiles)

    assert isinstance(result, dict)
    assert "off_targets" in result
    assert "risk_assessment" in result

    for target in result["off_targets"]:
        assert "protein_family" in target
        assert "similarity_score" in target
        assert "risk_level" in target
        assert target["risk_level"] in ["low", "medium", "high"]
        assert 0 <= target["similarity_score"] <= 1

def test_error_handling(drug_discovery_engine):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        drug_discovery_engine.analyze_binding_sites("")

    with pytest.raises(ValueError):
        drug_discovery_engine.predict_drug_interactions("INVALID", "INVALID")

    with pytest.raises(ValueError):
        drug_discovery_engine.screen_off_targets("", "")

    with pytest.raises(ValueError):
        drug_discovery_engine.optimize_binding_site("SEQ", 10, 5, "SMILES")  # Invalid range
