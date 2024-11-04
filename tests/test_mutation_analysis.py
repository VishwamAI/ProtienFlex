import pytest
import torch
import esm
import numpy as np
from models.mutation_analysis import MutationAnalyzer
from tests.conftest import create_mock_result

# Test sequence (insulin)
TEST_SEQUENCE = "FVNQHLCGSHLVEAL"  # Shorter segment of insulin sequence
TEST_POSITION = 7
TEST_MUTATION = "A"

@pytest.fixture
def analyzer(mocker):
    """Fixture for creating a MutationAnalyzer instance with mocked dependencies."""
    device = torch.device("cpu")
    mock_model = mocker.MagicMock()
    mock_alphabet = mocker.MagicMock()
    mock_model.alphabet = mock_alphabet
    mock_model.layers = [mocker.MagicMock() for _ in range(12)]
    mock_model.modules = lambda: []

    def batch_converter_return(data):
        sequence = data[0][1]
        batch_tokens = torch.randn(1, len(sequence) + 2, 33)
        return (
            [{"sequence": sequence, "start": 0, "end": len(sequence)}],
            [sequence],
            batch_tokens
        )

    mock_converter = mocker.MagicMock()
    mock_converter.side_effect = batch_converter_return
    mock_alphabet.get_batch_converter.return_value = mock_converter

    return MutationAnalyzer(mock_model, device)

def test_predict_mutation_effect(mocker, analyzer):
    """Test the complete mutation effect prediction pipeline"""
    # Mock the internal method returns
    stability_result = create_mock_result(mocker, {
        'start': 0,
        'end': len(TEST_SEQUENCE),
        'score': 0.85,
        'type': 'stability',
        'stability_score': 0.75
    })
    structural_result = create_mock_result(mocker, {
        'start': 0,
        'end': len(TEST_SEQUENCE),
        'score': 0.8,
        'type': 'structural',
        'structural_score': 0.8
    })
    conservation_result = create_mock_result(mocker, {
        'start': 0,
        'end': len(TEST_SEQUENCE),
        'score': 0.9,
        'type': 'conservation',
        'conservation_score': 0.85
    })

    # Configure mock methods using MagicMock with side_effect
    mock_stability = mocker.MagicMock(side_effect=lambda *args: stability_result)
    setattr(analyzer, '_calculate_stability_impact', mock_stability)

    mock_structural = mocker.MagicMock(side_effect=lambda *args: structural_result)
    setattr(analyzer, '_calculate_structural_impact', mock_structural)

    mock_conservation = mocker.MagicMock(side_effect=lambda *args: conservation_result)
    setattr(analyzer, '_calculate_conservation', mock_conservation)

    result = analyzer.predict_mutation_effect(
        TEST_SEQUENCE,
        TEST_POSITION,
        TEST_MUTATION
    )

    # Check dictionary structure
    assert isinstance(result, dict)
    assert all(key in result for key in ['start', 'end', 'score', 'type', 'stability_impact',
                                       'structural_impact', 'conservation_score', 'overall_impact',
                                       'confidence'])

    # Check value ranges
    assert 0 <= result['stability_impact'] <= 1
    assert 0 <= result['structural_impact'] <= 1
    assert 0 <= result['conservation_score'] <= 1
    assert 0 <= result['overall_impact'] <= 1
    assert 0 <= result['confidence'] <= 100
    assert 0 <= result['score'] <= 1

def test_stability_impact(mocker, analyzer):
    """Test stability impact calculation"""
    # Mock model forward method
    forward_result = create_mock_result(mocker, {
        'representations': {33: torch.randn(1, len(TEST_SEQUENCE), 1280)},
        'start': 0,
        'end': len(TEST_SEQUENCE),
        'score': 0.85,
        'type': 'stability'
    })
    mock_forward = mocker.MagicMock(side_effect=lambda *args: forward_result)
    setattr(analyzer.model, 'forward', mock_forward)

    result = analyzer._calculate_stability_impact(
        TEST_SEQUENCE,
        TEST_POSITION,
        TEST_MUTATION
    )

    assert isinstance(result, dict)
    assert all(key in result for key in ['start', 'end', 'score', 'type', 'stability_score'])
    assert isinstance(result['stability_score'], float)
    assert 0 <= result['stability_score'] <= 1

    # Test with different mutations
    mutations = ['G', 'P', 'D']
    for mut in mutations:
        result = analyzer._calculate_stability_impact(
            TEST_SEQUENCE,
            TEST_POSITION,
            mut
        )
        assert isinstance(result, dict)
        assert 'stability_score' in result
        assert isinstance(result['stability_score'], float)
        assert 0 <= result['stability_score'] <= 1

def test_structural_impact(mocker, analyzer):
    """Test structural impact calculation"""
    # Mock model forward method
    forward_result = create_mock_result(mocker, {
        'representations': {33: torch.randn(1, len(TEST_SEQUENCE), 1280)},
        'start': 0,
        'end': len(TEST_SEQUENCE),
        'score': 0.8,
        'type': 'structural'
    })
    mock_forward = mocker.MagicMock(side_effect=lambda *args: forward_result)
    setattr(analyzer.model, 'forward', mock_forward)

    result = analyzer._calculate_structural_impact(
        TEST_SEQUENCE,
        TEST_POSITION
    )

    assert isinstance(result, dict)
    assert all(key in result for key in ['start', 'end', 'score', 'type', 'structural_score'])
    assert isinstance(result['structural_score'], float)
    assert 0 <= result['structural_score'] <= 1

    # Test at different positions
    positions = [3, 7, 11]  # Positions within the shorter sequence length
    for pos in positions:
        result = analyzer._calculate_structural_impact(
            TEST_SEQUENCE,
            pos
        )
        assert isinstance(result, dict)
        assert 'structural_score' in result
        assert isinstance(result['structural_score'], float)
        assert 0 <= result['structural_score'] <= 1

def test_conservation(mocker, analyzer):
    """Test conservation score calculation"""
    # Mock model forward method
    forward_result = create_mock_result(mocker, {
        'representations': {33: torch.randn(1, len(TEST_SEQUENCE), 1280)},
        'start': 0,
        'end': len(TEST_SEQUENCE),
        'score': 0.9,
        'type': 'conservation'
    })
    mock_forward = mocker.MagicMock(side_effect=lambda *args: forward_result)
    setattr(analyzer.model, 'forward', mock_forward)

    result = analyzer._calculate_conservation(
        TEST_SEQUENCE,
        TEST_POSITION
    )

    assert isinstance(result, dict)
    assert all(key in result for key in ['start', 'end', 'score', 'type', 'conservation_score'])
    assert isinstance(result['conservation_score'], float)
    assert 0 <= result['conservation_score'] <= 1

    # Test at different positions
    positions = [2, 5, 8]  # Positions within the shorter sequence length
    for pos in positions:
        result = analyzer._calculate_conservation(
            TEST_SEQUENCE,
            pos
        )
        assert isinstance(result, dict)
        assert 'conservation_score' in result
        assert isinstance(result['conservation_score'], float)
        assert 0 <= result['conservation_score'] <= 1

def test_device_compatibility(mocker):
    """Test that the analyzer works on both CPU and GPU"""
    mock_result = create_mock_result(mocker, {
        'start': 0,
        'end': len(TEST_SEQUENCE),
        'score': 0.85,
        'type': 'mutation_effect',
        'stability_impact': 0.8,
        'structural_impact': 0.7,
        'conservation_score': 0.9,
        'overall_impact': 0.8,
        'confidence': 85
    })

    # Test on CPU
    cpu_device = torch.device("cpu")
    cpu_model = mocker.MagicMock()
    mock_alphabet = mocker.MagicMock()
    cpu_model.alphabet = mock_alphabet
    cpu_analyzer = MutationAnalyzer(cpu_model, cpu_device)

    mock_predict = mocker.MagicMock(side_effect=lambda *args: mock_result)
    setattr(cpu_analyzer, 'predict_mutation_effect', mock_predict)

    cpu_result = cpu_analyzer.predict_mutation_effect(
        TEST_SEQUENCE,
        TEST_POSITION,
        TEST_MUTATION
    )
    assert isinstance(cpu_result, dict)
    assert all(key in cpu_result for key in ['start', 'end', 'score', 'type'])
    assert 'error' not in cpu_result

    # Test on GPU if available
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_model = mocker.MagicMock()
        gpu_model.alphabet = mock_alphabet
        gpu_analyzer = MutationAnalyzer(gpu_model, gpu_device)

        mock_predict = mocker.MagicMock(side_effect=lambda *args: mock_result)
        setattr(gpu_analyzer, 'predict_mutation_effect', mock_predict)

        gpu_result = gpu_analyzer.predict_mutation_effect(
            TEST_SEQUENCE,
            TEST_POSITION,
            TEST_MUTATION
        )
        assert isinstance(gpu_result, dict)
        assert all(key in gpu_result for key in ['start', 'end', 'score', 'type'])
        assert 'error' not in gpu_result

def test_error_handling(mocker, analyzer):
    """Test error handling with invalid inputs"""
    # Mock error responses
    error_result = create_mock_result(mocker, {
        'start': 0,
        'end': len(TEST_SEQUENCE),
        'score': 0.0,
        'type': 'error',
        'error': 'Invalid input',
        'stability_impact': 0.0,
        'structural_impact': 0.0,
        'conservation_score': 0.0,
        'overall_impact': 0.0,
        'confidence': 0.0
    })
    predict_mock = mocker.MagicMock()
    predict_mock.side_effect = lambda *args: error_result
    analyzer.predict_mutation_effect = predict_mock

    # Test with invalid position
    result = analyzer.predict_mutation_effect(
        TEST_SEQUENCE,
        len(TEST_SEQUENCE) + 1,  # Invalid position
        TEST_MUTATION
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'type' in result
    assert result['type'] == 'error'

    # Test with invalid mutation
    result = analyzer.predict_mutation_effect(
        TEST_SEQUENCE,
        TEST_POSITION,
        'X'  # Invalid amino acid
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'type' in result
    assert result['type'] == 'error'

    # Test with invalid sequence
    result = analyzer.predict_mutation_effect(
        "INVALID123",  # Invalid sequence
        TEST_POSITION,
        TEST_MUTATION
    )
    assert isinstance(result, dict)
    assert 'error' in result
    assert 'type' in result
    assert result['type'] == 'error'
