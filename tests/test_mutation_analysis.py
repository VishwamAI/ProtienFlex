import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from proteinflex.models.analysis.mutation_analysis import MutationAnalyzer

@pytest.fixture
def mock_mutation_analyzer(mock_esm_model):
    """Provide mock mutation analyzer."""
    model = mock_esm_model
    model.eval.return_value = model
    model.alphabet.get_batch_converter.return_value = Mock()
    device = torch.device('cpu')
    analyzer = MutationAnalyzer(model, device)
    return analyzer

@pytest.fixture
def test_data():
    """Provide test data."""
    return {
        'sequence': "FVNQHLCGSHLVEAL",  # Shorter segment of insulin sequence
        'position': 7,
        'mutation': "A"
    }

def test_predict_mutation_effect(mock_mutation_analyzer, test_data):
    """Test the complete mutation effect prediction pipeline."""
    result = mock_mutation_analyzer.predict_mutation_effect(
        test_data['sequence'],
        test_data['position'],
        test_data['mutation']
    )

    # Check result structure
    assert isinstance(result, dict)
    assert 'stability_impact' in result
    assert 'structural_impact' in result
    assert 'conservation_score' in result
    assert 'overall_impact' in result
    assert 'confidence' in result

    # Check value ranges
    assert 0 <= result['stability_impact'] <= 1
    assert 0 <= result['structural_impact'] <= 1
    assert 0 <= result['conservation_score'] <= 1
    assert 0 <= result['overall_impact'] <= 1
    assert 0 <= result['confidence'] <= 100

def test_stability_impact(mock_mutation_analyzer, test_data):
    """Test stability impact calculation"""
    stability_score = mock_mutation_analyzer._calculate_stability_impact(
        test_data['sequence'],
        test_data['position'],
        test_data['mutation']
    )

    assert isinstance(stability_score, float)
    assert 0 <= stability_score <= 1

    # Test with different mutations
    mutations = ['G', 'P', 'D']
    for mut in mutations:
        score = mock_mutation_analyzer._calculate_stability_impact(
            test_data['sequence'],
            test_data['position'],
            mut
        )
        assert isinstance(score, float)
        assert 0 <= score <= 1

def test_structural_impact(mock_mutation_analyzer, test_data):
    """Test structural impact calculation"""
    structural_score = mock_mutation_analyzer._calculate_structural_impact(
        test_data['sequence'],
        test_data['position']
    )

    assert isinstance(structural_score, float)
    assert 0 <= structural_score <= 1

    # Test at different positions
    positions = [3, 7, 11]  # Positions within the shorter sequence length
    for pos in positions:
        score = mock_mutation_analyzer._calculate_structural_impact(
            test_data['sequence'],
            pos
        )
        assert isinstance(score, float)
        assert 0 <= score <= 1

def test_conservation(mock_mutation_analyzer, test_data):
    """Test conservation score calculation"""
    conservation_score = mock_mutation_analyzer._calculate_conservation(
        test_data['sequence'],
        test_data['position']
    )

    assert isinstance(conservation_score, float)
    assert 0 <= conservation_score <= 1

    # Test at different positions
    positions = [2, 5, 8]  # Positions within the shorter sequence length
    for pos in positions:
        score = mock_mutation_analyzer._calculate_conservation(
            test_data['sequence'],
            pos
        )
        assert isinstance(score, float)
        assert 0 <= score <= 1

def test_device_compatibility(mock_mutation_analyzer, test_data):
    """Test that the analyzer works on both CPU and GPU"""
    # Test on CPU
    cpu_device = torch.device("cpu")
    cpu_model = mock_mutation_analyzer.model.to(cpu_device)
    cpu_analyzer = MutationAnalyzer(cpu_model, cpu_device)
    cpu_result = cpu_analyzer.predict_mutation_effect(
        test_data['sequence'],
        test_data['position'],
        test_data['mutation']
    )
    assert isinstance(cpu_result, dict)
    assert 'error' not in cpu_result

    # Test on GPU if available
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_model = mock_mutation_analyzer.model.to(gpu_device)
        gpu_analyzer = MutationAnalyzer(gpu_model, gpu_device)
        gpu_result = gpu_analyzer.predict_mutation_effect(
            test_data['sequence'],
            test_data['position'],
            test_data['mutation']
        )
        assert isinstance(gpu_result, dict)
        assert 'error' not in gpu_result

def test_error_handling(mock_mutation_analyzer, test_data):
    """Test error handling with invalid inputs"""
    # Test with invalid position
    result = mock_mutation_analyzer.predict_mutation_effect(
        test_data['sequence'],
        len(test_data['sequence']) + 1,  # Invalid position
        test_data['mutation']
    )
    assert 'error' in result

    # Test with invalid mutation
    result = mock_mutation_analyzer.predict_mutation_effect(
        test_data['sequence'],
        test_data['position'],
        'X'  # Invalid amino acid
    )
    assert 'error' in result

    # Test with invalid sequence
    result = mock_mutation_analyzer.predict_mutation_effect(
        "INVALID123",  # Invalid sequence
        test_data['position'],
        test_data['mutation']
    )
    assert 'error' in result
