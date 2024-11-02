import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from proteinflex.models.analysis.mutation_analysis import MutationAnalyzer

@pytest.fixture
def mock_esm_model():
    """Create a mock ESM model for testing."""
    mock_model = Mock()
    mock_model.alphabet = Mock()
    mock_model.alphabet.get_idx = lambda x: ord(x) - ord('A')

    def mock_forward(seq):
        batch_size, seq_len = seq.shape
        # Mock representations
        representations = {33: torch.randn(batch_size, seq_len, 768)}
        # Mock attention maps
        attentions = torch.randn(batch_size, 12, seq_len, seq_len)
        return {'representations': representations, 'attentions': attentions}

    mock_model.side_effect = mock_forward
    return mock_model

@pytest.fixture
def analyzer(mock_esm_model):
    """Create a MutationAnalyzer instance with mock model."""
    device = torch.device('cpu')
    return MutationAnalyzer(mock_esm_model, device)

def test_invalid_sequence(analyzer):
    """Test handling of invalid sequence input."""
    result = analyzer.predict_mutation_effect("X123", 0, "A")
    assert "error" in result
    assert "Invalid inputs" in result["error"]

def test_invalid_position(analyzer):
    """Test handling of invalid mutation position."""
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    result = analyzer.predict_mutation_effect(sequence, len(sequence), "A")
    assert "error" in result
    assert "Invalid inputs" in result["error"]

def test_invalid_mutation(analyzer):
    """Test handling of invalid mutation amino acid."""
    result = analyzer.predict_mutation_effect("ACDEF", 0, "X")
    assert "error" in result
    assert "Invalid inputs" in result["error"]

def test_memory_optimization(analyzer):
    """Test memory optimization with torch.no_grad."""
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    with patch('torch.no_grad') as mock_no_grad:
        analyzer.predict_mutation_effect(sequence, 0, "A")
        assert mock_no_grad.called
        # Should be called 3 times: stability, structural, conservation
        assert mock_no_grad.call_count == 3

def test_error_handling_during_calculation(analyzer, mock_esm_model):
    """Test error handling during calculation process."""
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    # Make model raise an exception
    mock_esm_model.side_effect = RuntimeError("GPU memory error")

    result = analyzer.predict_mutation_effect(sequence, 0, "A")
    assert "error" in result
    assert "GPU memory error" in result["error"]

def test_stability_impact_edge_case(analyzer):
    """Test stability impact calculation with edge case sequence."""
    sequence = "A" * 100  # Long homopolymer sequence
    result = analyzer.predict_mutation_effect(sequence, 50, "D")
    assert "stability_impact" in result
    assert 0 <= result["stability_impact"] <= 1

def test_structural_impact_boundary(analyzer):
    """Test structural impact calculation at sequence boundaries."""
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    # Test first position
    result1 = analyzer.predict_mutation_effect(sequence, 0, "A")
    # Test last position
    result2 = analyzer.predict_mutation_effect(sequence, len(sequence)-1, "A")

    assert 0 <= result1["structural_impact"] <= 1
    assert 0 <= result2["structural_impact"] <= 1

def test_conservation_score_special_case(analyzer):
    """Test conservation score calculation with special case."""
    # Sequence with repeated motif
    sequence = "ACDEF" * 4
    result = analyzer.predict_mutation_effect(sequence, 5, "A")
    assert 0 <= result["conservation_score"] <= 1

def test_confidence_score_extremes(analyzer):
    """Test confidence score calculation with extreme values."""
    sequence = "ACDEFGHIKLMNPQRSTVWY"

    # Mock stability and structural calculations to return extreme values
    with patch.object(analyzer, '_calculate_stability_impact', return_value=1.0), \
         patch.object(analyzer, '_calculate_structural_impact', return_value=1.0):
        result = analyzer.predict_mutation_effect(sequence, 0, "A")
        assert result["confidence"] == 100.0

    with patch.object(analyzer, '_calculate_stability_impact', return_value=0.0), \
         patch.object(analyzer, '_calculate_structural_impact', return_value=0.0):
        result = analyzer.predict_mutation_effect(sequence, 0, "A")
        assert result["confidence"] == 0.0

def test_batch_memory_usage(analyzer):
    """Test memory usage during batch processing."""
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 5  # Longer sequence
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Process multiple mutations
    mutations = [("A", i) for i in range(0, len(sequence), 10)]
    for mut, pos in mutations:
        analyzer.predict_mutation_effect(sequence, pos, mut)

    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    # Memory should be released after each prediction
    assert final_memory == initial_memory
