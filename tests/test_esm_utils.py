import pytest
from unittest.mock import Mock, patch
import torch
import numpy as np
from models.esm_utils import ESMModelHandler

@pytest.fixture
def esm_handler():
    """Fixture for creating an ESMModelHandler instance with mocked ESM model."""
    with patch('models.esm_utils.esm') as mock_esm:
        # Mock ESM model and alphabet
        mock_model = Mock()
        mock_alphabet = Mock()
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_model, mock_alphabet)

        # Mock batch converter
        mock_alphabet.get_batch_converter.return_value = Mock(return_value=(
            Mock(),  # batch_labels
            Mock(),  # batch_strs
            torch.randn(1, 100, 33)  # batch_tokens
        ))

        handler = ESMModelHandler()
        handler.model = mock_model
        handler.alphabet = mock_alphabet
        return handler

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_get_sequence_embeddings(esm_handler, sequence):
    """Test sequence embedding generation."""
    with patch.object(esm_handler.model, 'forward') as mock_forward:
        # Mock model output
        mock_forward.return_value = {
            "representations": {33: torch.randn(1, len(sequence), 1280)},
            "logits": torch.randn(1, len(sequence), 33),
            "attentions": torch.randn(1, 1, len(sequence), len(sequence))
        }

        embeddings = esm_handler.get_sequence_embeddings(sequence)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[1] == 1280  # ESM-2 embedding dimension
        assert embeddings.shape[0] == len(sequence)

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_get_attention_maps(esm_handler, sequence):
    """Test attention map generation."""
    with patch.object(esm_handler.model, 'forward') as mock_forward:
        # Mock model output with attention maps
        mock_forward.return_value = {
            "attentions": torch.randn(1, 1, len(sequence), len(sequence))
        }

        attention_maps = esm_handler.get_attention_maps(sequence)

        assert isinstance(attention_maps, torch.Tensor)
        assert attention_maps.shape[-1] == len(sequence)
        assert attention_maps.shape[-2] == len(sequence)

@pytest.mark.parametrize("sequence,window_size", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", 5),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", 7),
])
def test_analyze_sequence_windows(esm_handler, sequence, window_size):
    """Test sequence window analysis."""
    with patch.object(esm_handler.model, 'forward') as mock_forward:
        # Mock model output
        mock_forward.return_value = {
            "representations": {33: torch.randn(1, len(sequence), 1280)},
            "logits": torch.randn(1, len(sequence), 33)
        }

        window_analysis = esm_handler.analyze_sequence_windows(sequence, window_size)

        assert isinstance(window_analysis, list)
        assert len(window_analysis) == max(1, len(sequence) - window_size + 1)

        for window in window_analysis:
            assert "start" in window
            assert "end" in window
            assert "score" in window
            assert isinstance(window["score"], float)
            assert 0 <= window["score"] <= 1

@pytest.mark.parametrize("sequence1,sequence2", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"),
])
def test_compare_sequences(esm_handler, sequence1, sequence2):
    """Test sequence comparison functionality."""
    with patch.object(esm_handler.model, 'forward') as mock_forward:
        # Mock model output for both sequences
        mock_forward.side_effect = [
            {"representations": {33: torch.randn(1, len(sequence1), 1280)}},
            {"representations": {33: torch.randn(1, len(sequence2), 1280)}}
        ]

        similarity_score = esm_handler.compare_sequences(sequence1, sequence2)

        assert isinstance(similarity_score, float)
        assert 0 <= similarity_score <= 1

def test_error_handling(esm_handler):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        esm_handler.get_sequence_embeddings("")

    with pytest.raises(ValueError):
        esm_handler.get_attention_maps("X")  # Invalid amino acid

    with pytest.raises(ValueError):
        esm_handler.analyze_sequence_windows("SEQUENCE", 0)  # Invalid window size

    with pytest.raises(ValueError):
        esm_handler.compare_sequences("", "SEQUENCE")

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_confidence_scores(esm_handler, sequence):
    """Test confidence score calculation."""
    with patch.object(esm_handler.model, 'forward') as mock_forward:
        # Mock model output with logits
        mock_forward.return_value = {
            "logits": torch.randn(1, len(sequence), 33)
        }

        confidence_scores = esm_handler.calculate_confidence_scores(sequence)

        assert isinstance(confidence_scores, np.ndarray)
        assert len(confidence_scores) == len(sequence)
        assert all(0 <= score <= 1 for score in confidence_scores)
