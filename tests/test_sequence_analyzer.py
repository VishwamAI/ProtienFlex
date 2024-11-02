import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from proteinflex.models.analysis.sequence_analyzer import SequenceAnalyzer

@pytest.fixture
def sequence_analyzer():
    return SequenceAnalyzer()

@pytest.fixture
def mock_sequence():
    return "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"

def test_init(sequence_analyzer):
    """Test initialization of SequenceAnalyzer"""
    assert hasattr(sequence_analyzer, 'esm_model')
    assert hasattr(sequence_analyzer, 'device')

def test_validate_sequence(sequence_analyzer, mock_sequence):
    """Test sequence validation"""
    assert sequence_analyzer._validate_sequence(mock_sequence)
    assert not sequence_analyzer._validate_sequence("")
    assert not sequence_analyzer._validate_sequence(None)
    assert not sequence_analyzer._validate_sequence("XYZ123")  # Invalid amino acids

def test_calculate_sequence_properties(sequence_analyzer, mock_sequence):
    """Test calculation of sequence properties"""
    properties = sequence_analyzer.calculate_sequence_properties(mock_sequence)

    assert isinstance(properties, dict)
    assert 'length' in properties
    assert 'molecular_weight' in properties
    assert 'isoelectric_point' in properties
    assert 'hydrophobicity' in properties

    assert properties['length'] == len(mock_sequence)
    assert isinstance(properties['molecular_weight'], float)
    assert isinstance(properties['isoelectric_point'], float)
    assert isinstance(properties['hydrophobicity'], float)

def test_predict_secondary_structure(sequence_analyzer, mock_sequence):
    """Test secondary structure prediction"""
    with patch('proteinflex.models.analysis.sequence_analyzer.torch') as mock_torch:
        mock_torch.device.return_value = 'cpu'
        mock_torch.tensor.return_value = Mock()

        prediction = sequence_analyzer.predict_secondary_structure(mock_sequence)

        assert isinstance(prediction, dict)
        assert 'helix' in prediction
        assert 'sheet' in prediction
        assert 'coil' in prediction

        assert all(isinstance(v, float) for v in prediction.values())
        assert abs(sum(prediction.values()) - 1.0) < 1e-6  # Probabilities sum to 1

def test_analyze_conservation(sequence_analyzer, mock_sequence):
    """Test sequence conservation analysis"""
    with patch('proteinflex.models.analysis.sequence_analyzer.esm') as mock_esm:
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.return_value = {
            'representations': {
                33: torch.randn(1, len(mock_sequence), 768)
            }
        }
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_model, Mock())

        conservation = sequence_analyzer.analyze_conservation(mock_sequence)

        assert isinstance(conservation, np.ndarray)
        assert len(conservation) == len(mock_sequence)
        assert np.all(conservation >= 0) and np.all(conservation <= 1)

def test_get_sequence_embeddings(sequence_analyzer, mock_sequence):
    """Test sequence embedding generation"""
    with patch('proteinflex.models.analysis.sequence_analyzer.esm') as mock_esm:
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.return_value = {
            'representations': {
                33: torch.randn(1, len(mock_sequence), 768)
            }
        }
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_model, Mock())

        embeddings = sequence_analyzer._get_sequence_embeddings(mock_sequence)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[1] == 768  # ESM embedding dimension
        assert embeddings.shape[0] == len(mock_sequence)

def test_error_handling(sequence_analyzer):
    """Test error handling for invalid inputs"""
    # Test with None sequence
    with pytest.raises(ValueError):
        sequence_analyzer.calculate_sequence_properties(None)

    # Test with empty sequence
    with pytest.raises(ValueError):
        sequence_analyzer.predict_secondary_structure("")

    # Test with invalid amino acids
    with pytest.raises(ValueError):
        sequence_analyzer.analyze_conservation("XYZ123")

def test_batch_processing(sequence_analyzer):
    """Test batch processing of sequences"""
    sequences = [
        "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
        "ARNDCQEGHILKMFPSTWYV"
    ]

    results = sequence_analyzer.process_sequences(sequences)

    assert isinstance(results, list)
    assert len(results) == len(sequences)
    assert all(isinstance(r, dict) for r in results)
    assert all('properties' in r for r in results)
    assert all('secondary_structure' in r for r in results)
    assert all('conservation' in r for r in results)
