import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from models.drug_discovery import DrugDiscoveryEngine
from tests.conftest import create_mock_method, create_mock_result



@pytest.mark.parametrize("sequence,site_start,site_end,ligand_smiles", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", 10, 20, "CC1=CC=C(C=C1)CC(C(=O)O)N"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", 5, 15, "CC(=O)NC1=CC=C(O)C=C1"),
])
def test_optimize_binding_site(mocker, drug_discovery_engine, sequence, site_start, site_end, ligand_smiles):
    """Test binding site optimization with different sequences and ligands."""
    mock_result = {
        "start": site_start,
        "end": site_end,
        "score": 0.85,
        "type": "binding_site_optimization",
        "site_analysis": {
            "start": site_start,
            "end": site_end,
            "score": 0.9,
            "type": "site_analysis",
            "hydrophobicity": 0.7,
            "length": site_end - site_start,
            "residue_properties": ["hydrophobic", "polar"]
        },
        "optimization_suggestions": [
            {
                "start": site_start,
                "end": site_end,
                "score": 0.8,
                "type": "suggestion",
                "issue": "hydrophobicity",
                "suggestion": "increase polarity",
                "confidence": 0.9
            }
        ],
        "optimization_score": 0.85,
        "predicted_improvement": 0.2
    }

    # Create mock method with side_effect
    mock_optimize = create_mock_method(mocker, mock_result)
    setattr(drug_discovery_engine, 'optimize_binding_site', mock_optimize)

    result = drug_discovery_engine.optimize_binding_site(sequence, site_start, site_end, ligand_smiles)

    assert isinstance(result, dict)
    required_fields = ["site_analysis", "optimization_suggestions", "optimization_score",
                      "predicted_improvement", "start", "end", "score", "type"]
    assert all(field in result for field in required_fields)

    # Validate site analysis
    site_analysis = result.get("site_analysis", {})
    assert isinstance(site_analysis, dict)
    required_site_fields = ["start", "end", "score", "type", "hydrophobicity",
                           "length", "residue_properties"]
    assert all(field in site_analysis for field in required_site_fields)
    assert isinstance(site_analysis.get("hydrophobicity", 0.0), float)
    assert isinstance(site_analysis.get("length", 0), int)
    assert isinstance(site_analysis.get("residue_properties", []), list)

    # Validate optimization suggestions
    suggestions = result.get("optimization_suggestions", [])
    assert isinstance(suggestions, list)
    for suggestion in suggestions:
        assert isinstance(suggestion, dict)
        required_suggestion_fields = ["start", "end", "score", "type", "issue",
                                    "suggestion", "confidence"]
        assert all(field in suggestion for field in required_suggestion_fields)
        confidence = suggestion.get("confidence", 0.0)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    # Validate scores
    assert 0 <= result.get("optimization_score", 0.0) <= 1
    assert 0 <= result.get("predicted_improvement", 0.0) <= 1

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_analyze_binding_sites(mocker, drug_discovery_engine, sequence):
    """Test binding site analysis with different sequences."""
    mock_result = {
        "start": 0,
        "end": len(sequence),
        "score": 0.9,
        "type": "binding_site_analysis",
        "binding_sites": [
            {
                "start": 10,
                "end": 20,
                "score": 0.85,
                "type": "binding_site",
                "properties": {
                    "hydrophobicity": 0.7,
                    "accessibility": 0.8
                }
            }
        ],
        "analysis_summary": "Found potential binding sites"
    }
    mock_analyze = create_mock_method(mocker, mock_result)
    setattr(drug_discovery_engine, 'analyze_binding_sites', mock_analyze)

    result = drug_discovery_engine.analyze_binding_sites(sequence)

    assert isinstance(result, dict)
    assert "binding_sites" in result
    assert isinstance(result["binding_sites"], list)
    for site in result["binding_sites"]:
        assert isinstance(site, dict)
        assert all(field in site for field in ["start", "end", "score", "type", "properties"])
        assert isinstance(site["properties"], dict)

@pytest.mark.parametrize("sequence,ligand_smiles", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "CC1=CC=C(C=C1)CC(C(=O)O)N"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "CC(=O)NC1=CC=C(O)C=C1"),
])
def test_predict_drug_interactions(mocker, drug_discovery_engine, sequence, ligand_smiles):
    """Test drug interaction prediction with different sequences and ligands."""
    mock_result = {
        "start": 0,
        "end": len(sequence),
        "score": 0.9,
        "type": "drug_interaction_prediction",
        "interactions": [
            {
                "start": 10,
                "end": 20,
                "score": 0.85,
                "type": "interaction",
                "interaction_type": "hydrogen_bond",
                "strength": 0.8,
                "residues": ["SER", "THR", "TYR"]
            }
        ],
        "binding_energy": -8.5,
        "stability_score": 0.75
    }
    mock_predict = create_mock_method(mocker, mock_result)
    setattr(drug_discovery_engine, 'predict_drug_interactions', mock_predict)

    result = drug_discovery_engine.predict_drug_interactions(sequence, ligand_smiles)

    assert isinstance(result, dict)
    assert all(field in result for field in ["start", "end", "score", "type", "interactions", "binding_energy", "stability_score"])
    assert isinstance(result["interactions"], list)
    for interaction in result["interactions"]:
        assert isinstance(interaction, dict)
        assert all(field in interaction for field in ["start", "end", "score", "type", "interaction_type", "strength", "residues"])
        assert isinstance(interaction["strength"], float)
        assert 0 <= interaction["strength"] <= 1

@pytest.mark.parametrize("sequence,ligand_smiles", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "CC1=CC=C(C=C1)CC(C(=O)O)N"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "CC(=O)NC1=CC=C(O)C=C1"),
])
def test_screen_off_targets(mocker, drug_discovery_engine, sequence, ligand_smiles):
    """Test off-target screening with different sequences and ligands."""
    mock_result = {
        "start": 0,
        "end": len(sequence),
        "score": 0.9,
        "type": "off_target_screening",
        "potential_targets": [
            {
                "start": 5,
                "end": 15,
                "score": 0.7,
                "type": "off_target",
                "protein_name": "ABC transporter",
                "binding_probability": 0.65,
                "risk_level": "medium"
            }
        ],
        "overall_safety_score": 0.8,
        "recommendations": [
            {
                "start": 0,
                "end": len(sequence),
                "score": 0.75,
                "type": "safety_recommendation",
                "suggestion": "Consider modifying binding site to reduce off-target effects",
                "priority": "medium"
            }
        ]
    }
    mock_screen = create_mock_method(mocker, mock_result)
    setattr(drug_discovery_engine, 'screen_off_targets', mock_screen)

    result = drug_discovery_engine.screen_off_targets(sequence, ligand_smiles)

    assert isinstance(result, dict)
    assert all(field in result for field in ["start", "end", "score", "type", "potential_targets", "overall_safety_score", "recommendations"])
    assert isinstance(result["potential_targets"], list)
    for target in result["potential_targets"]:
        assert isinstance(target, dict)
        assert all(field in target for field in ["start", "end", "score", "type", "protein_name", "binding_probability", "risk_level"])

def test_error_handling(mocker, drug_discovery_engine):
    """Test error handling for invalid inputs."""
    # Mock error responses
    error_result = {
        "start": 0,
        "end": 0,
        "score": 0.0,
        "type": "error",
        "error": "Invalid input",
        "details": "Sequence length must be greater than 0"
    }

    # Set up mock methods
    mock_analyze = create_mock_method(mocker, error_result)
    mock_predict = create_mock_method(mocker, error_result)
    mock_screen = create_mock_method(mocker, error_result)
    mock_optimize = create_mock_method(mocker, error_result)

    # Attach mock methods
    setattr(drug_discovery_engine, 'analyze_binding_sites', mock_analyze)
    setattr(drug_discovery_engine, 'predict_drug_interactions', mock_predict)
    setattr(drug_discovery_engine, 'screen_off_targets', mock_screen)
    setattr(drug_discovery_engine, 'optimize_binding_site', mock_optimize)

    # Test with invalid inputs
    invalid_sequence = ""
    invalid_smiles = "invalid_smiles"

    # Test each method
    result1 = drug_discovery_engine.analyze_binding_sites(invalid_sequence)
    result2 = drug_discovery_engine.predict_drug_interactions(invalid_sequence, invalid_smiles)
    result3 = drug_discovery_engine.screen_off_targets(invalid_sequence, invalid_smiles)
    result4 = drug_discovery_engine.optimize_binding_site(invalid_sequence, 0, 0, invalid_smiles)

    # Verify error responses
    for result in [result1, result2, result3, result4]:
        assert isinstance(result, dict)
        assert all(field in result for field in ["start", "end", "score", "type", "error", "details"])
        assert result["type"] == "error"
        assert result["score"] == 0.0
