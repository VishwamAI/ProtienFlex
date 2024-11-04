import pytest
from unittest.mock import Mock, patch
import numpy as np
from models.domain_analysis import DomainAnalyzer

@pytest.fixture
def domain_analyzer():
    """Fixture for creating a DomainAnalyzer instance with mocked dependencies."""
    with patch('models.domain_analysis.esm') as mock_esm:
        # Mock ESM model and alphabet
        mock_model = Mock()
        mock_alphabet = Mock()
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_model, mock_alphabet)

        analyzer = DomainAnalyzer()
        analyzer.model = mock_model
        analyzer.alphabet = mock_alphabet
        return analyzer

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_identify_domains(domain_analyzer, sequence):
    """Test domain identification in protein sequences."""
    with patch.object(domain_analyzer.model, 'forward') as mock_forward:
        # Mock model output
        mock_forward.return_value = {
            "representations": {33: np.random.randn(1, len(sequence), 1280)},
            "attentions": np.random.randn(1, 1, len(sequence), len(sequence))
        }

        domains = domain_analyzer.identify_domains(sequence)

        assert isinstance(domains, list)
        for domain in domains:
            assert "start" in domain
            assert "end" in domain
            assert "confidence" in domain
            assert "type" in domain
            assert 0 <= domain["confidence"] <= 1
            assert domain["start"] < domain["end"]

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_analyze_domain_interactions(domain_analyzer, sequence):
    """Test analysis of domain interactions."""
    with patch.object(domain_analyzer.model, 'forward') as mock_forward:
        mock_forward.return_value = {
            "representations": {33: np.random.randn(1, len(sequence), 1280)},
            "attentions": np.random.randn(1, 1, len(sequence), len(sequence))
        }

        interactions = domain_analyzer.analyze_domain_interactions(sequence)

        assert isinstance(interactions, list)
        for interaction in interactions:
            assert "domain1" in interaction
            assert "domain2" in interaction
            assert "interaction_type" in interaction
            assert "strength" in interaction
            assert 0 <= interaction["strength"] <= 1

@pytest.mark.parametrize("sequence,domain_type", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "binding"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "catalytic"),
])
def test_predict_domain_function(domain_analyzer, sequence, domain_type):
    """Test domain function prediction."""
    predictions = domain_analyzer.predict_domain_function(sequence, domain_type)

    assert isinstance(predictions, dict)
    assert "function" in predictions
    assert "confidence" in predictions
    assert "supporting_features" in predictions
    assert 0 <= predictions["confidence"] <= 1

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_calculate_domain_stability(domain_analyzer, sequence):
    """Test domain stability calculation."""
    stability_scores = domain_analyzer.calculate_domain_stability(sequence)

    assert isinstance(stability_scores, dict)
    for domain_id, score in stability_scores.items():
        assert isinstance(score, dict)
        assert "stability_score" in score
        assert "confidence" in score
        assert 0 <= score["stability_score"] <= 1
        assert 0 <= score["confidence"] <= 1

def test_error_handling(domain_analyzer):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        domain_analyzer.identify_domains("")

    with pytest.raises(ValueError):
        domain_analyzer.analyze_domain_interactions("X")  # Invalid sequence

    with pytest.raises(ValueError):
        domain_analyzer.predict_domain_function("", "binding")

    with pytest.raises(ValueError):
        domain_analyzer.calculate_domain_stability("INVALID")

@pytest.mark.parametrize("sequence,window_size", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", 5),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", 7),
])
def test_scan_domain_boundaries(domain_analyzer, sequence, window_size):
    """Test scanning for domain boundaries."""
    boundaries = domain_analyzer.scan_domain_boundaries(sequence, window_size)

    assert isinstance(boundaries, list)
    for boundary in boundaries:
        assert "position" in boundary
        assert "confidence" in boundary
        assert "type" in boundary
        assert 0 <= boundary["confidence"] <= 1
        assert 0 <= boundary["position"] < len(sequence)
