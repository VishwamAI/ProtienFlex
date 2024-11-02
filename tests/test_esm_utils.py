import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from proteinflex.models.utils.esm_utils import ESMModel

@pytest.fixture
def mock_esm_model():
    """Create mock ESM model with proper tensor outputs"""
    class MockESMModel(Mock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_mode = False
            self.is_cuda = False

        def eval(self):
            self.eval_mode = True
            return self

        def cuda(self):
            self.is_cuda = True
            return self

        def __call__(self, batch_tokens, repr_layers=None, **kwargs):
            # Ensure batch_tokens is a tensor
            if not isinstance(batch_tokens, torch.Tensor):
                raise ValueError("batch_tokens must be a tensor")

            # Create mock output with proper dimensions
            seq_length = batch_tokens.shape[1]
            return {
                "representations": {
                    33: torch.randn(1, seq_length, 768)  # Batch size 1, seq_length tokens, 768 features
                }
            }

    return MockESMModel()

@pytest.fixture
def mock_alphabet():
    alphabet = Mock()
    batch_converter = Mock()
    batch_converter.return_value = (
        ["protein"],  # batch_labels
        ["SEQUENCE"],  # batch_strs
        torch.randint(0, 100, (1, 10))  # batch_tokens
    )
    alphabet.get_batch_converter.return_value = batch_converter
    return alphabet

@pytest.fixture
def esm_predictor(mock_esm_model, mock_alphabet):
    with patch('proteinflex.models.utils.esm_utils.esm') as mock_esm:
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_esm_model, mock_alphabet)
        return ESMModel()

@pytest.fixture
def sample_sequence():
    return "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"

def test_init_success():
    """Test successful initialization of ESMPredictor"""
    with patch('proteinflex.models.utils.esm_utils.esm') as mock_esm, \
         patch('torch.cuda.is_available', return_value=False):

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_model, Mock())

        predictor = ESMModel()
        assert predictor.model == mock_model
        assert not predictor.model.cuda.called

def test_init_with_cuda():
    """Test initialization with CUDA available"""
    with patch('proteinflex.models.utils.esm_utils.esm') as mock_esm, \
         patch('torch.cuda.is_available', return_value=True):

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model.cuda.return_value = mock_model
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_model, Mock())

        model = ESMModel()
        assert model.model == mock_model
        assert model.model.cuda.called

def test_init_error():
    """Test error handling during initialization"""
    with patch('proteinflex.models.utils.esm_utils.esm') as mock_esm:
        mock_esm.pretrained.esm2_t33_650M_UR50D.side_effect = Exception("Model loading error")
        with pytest.raises(Exception):
            ESMModel()

def test_predict_structure(esm_predictor, sample_sequence):
    """Test structure prediction functionality"""
    representations, confidence = esm_predictor.predict_structure(sample_sequence)

    assert isinstance(representations, torch.Tensor)
    assert isinstance(confidence, list)
    assert len(confidence) > 0
    assert all(isinstance(score, float) for score in confidence)
    assert all(0 <= score <= 1 for score in confidence)

def test_predict_structure_with_cuda(mock_esm_model, mock_alphabet):
    """Test structure prediction with CUDA"""
    with patch('proteinflex.models.utils.esm_utils.esm') as mock_esm, \
         patch('torch.cuda.is_available', return_value=True), \
         patch('torch.Tensor.cuda', return_value=torch.randn(1, 10)), \
         patch('torch.nn.Module.cuda', return_value=mock_esm_model):

        # Configure mock model to return properly structured data
        mock_model = Mock()
        mock_model.eval = Mock(return_value=mock_model)
        mock_model.cuda = Mock(return_value=mock_model)
        mock_model.return_value = {
            "representations": {
                33: torch.randn(1, 10, 768)  # Batch size 1, 10 tokens, 768 features
            }
        }
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_model, mock_alphabet)
        predictor = ESMModel()

        representations, confidence = predictor.predict_structure("SEQUENCE")
        assert isinstance(representations, torch.Tensor)
        assert isinstance(confidence, list)

def test_predict_structure_error(esm_predictor):
    """Test error handling in structure prediction"""
    with pytest.raises(ValueError, match="Input sequence cannot be None"):
        esm_predictor.predict_structure(None)

def test_calculate_confidence(esm_predictor):
    """Test confidence score calculation"""
    # Create mock representations tensor
    representations = torch.randn(1, 10, 768)  # Batch size 1, 10 tokens, 768 features
    confidence = esm_predictor._calculate_confidence(representations)

    assert isinstance(confidence, list)
    assert len(confidence) == 10  # One score per residue
    assert all(isinstance(score, float) for score in confidence)
    assert all(0 <= score <= 1 for score in confidence)

def test_calculate_confidence_error(esm_predictor):
    """Test error handling in confidence calculation"""
    with pytest.raises(Exception):
        esm_predictor._calculate_confidence(None)

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFT",          # Short sequence
    "A" * 100,             # Long sequence
    "ACDEFGHIKLMNPQRSTVWY" # All amino acids
])
def test_predict_structure_different_sequences(esm_predictor, sequence):
    """Test structure prediction with different sequence types"""
    representations, confidence = esm_predictor.predict_structure(sequence)
    assert isinstance(representations, torch.Tensor)
    assert isinstance(confidence, list)
    assert len(confidence) > 0

def test_batch_conversion(esm_predictor, sample_sequence):
    """Test batch conversion functionality"""
    with patch.object(esm_predictor.alphabet, 'get_batch_converter') as mock_converter:
        mock_converter.return_value = Mock(return_value=(
            ["protein"],
            ["SEQUENCE"],
            torch.randint(0, 100, (1, 10))
        ))

        representations, confidence = esm_predictor.predict_structure(sample_sequence)
        assert isinstance(representations, torch.Tensor)
        assert isinstance(confidence, list)
        assert mock_converter.called

def test_gpu_memory_handling(esm_predictor, sample_sequence):
    """Test GPU memory handling"""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.Tensor.cuda', return_value=torch.randn(1, len(sample_sequence))), \
         patch('torch.nn.Module.cuda', return_value=esm_predictor.model):

        representations, confidence = esm_predictor.predict_structure(sample_sequence)
        assert isinstance(representations, torch.Tensor)
        assert isinstance(confidence, list)
        # Should not raise any GPU memory errors

def test_representation_shape(esm_predictor, sample_sequence):
    """Test output representation shape"""
    representations, _ = esm_predictor.predict_structure(sample_sequence)
    assert len(representations.shape) == 3  # (batch_size, sequence_length, hidden_dim)
    assert representations.shape[0] == 1    # Batch size
    assert representations.shape[2] == 768  # ESM-2 hidden dimension
