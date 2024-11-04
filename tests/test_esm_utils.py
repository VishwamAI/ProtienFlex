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
        def batch_converter_return(data):
            sequence = data[0][1]
            batch_tokens = torch.randn(1, len(sequence) + 2, 33)
            return (
                [{"sequence": sequence, "start": 0, "end": len(sequence)}],  # batch_labels
                [sequence],  # batch_strs
                batch_tokens  # batch_tokens
            )

        mock_alphabet.get_batch_converter.return_value = Mock(side_effect=batch_converter_return)

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
            "representations": {
                33: torch.randn(1, len(sequence), 1280),
                "start": 0,
                "end": len(sequence),
                "score": 0.9,
                "type": "representation"
            },
            "logits": {
                "values": torch.randn(1, len(sequence), 33),
                "start": 0,
                "end": len(sequence),
                "score": 0.85,
                "type": "logits"
            },
            "attentions": {
                "values": torch.randn(1, len(sequence), len(sequence)),
                "start": 0,
                "end": len(sequence),
                "score": 0.8,
                "type": "attention"
            },
            "start": 0,
            "end": len(sequence),
            "score": 0.9,
            "type": "embedding"
        }

        result = esm_handler.get_sequence_embeddings(sequence)

        assert isinstance(result, dict)
        assert "embeddings" in result
        assert "start" in result
        assert "end" in result
        assert "score" in result
        assert "type" in result
        assert isinstance(result["embeddings"], torch.Tensor)
        assert result["embeddings"].shape[1] == 1280  # ESM-2 embedding dimension
        assert result["embeddings"].shape[0] == len(sequence)

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_get_attention_maps(esm_handler, sequence):
    """Test attention map generation."""
    with patch.object(esm_handler.model, 'forward') as mock_forward:
        # Mock model output with attention maps
        mock_forward.return_value = {
            "attentions": {
                "start": 0,
                "end": len(sequence),
                "score": 0.9,
                "type": "attention_tensor",
                "data": torch.randn(1, 1, len(sequence), len(sequence))
            },
            "start": 0,
            "end": len(sequence),
            "score": 0.9,
            "type": "attention",
            "attention_maps": {
                "start": 0,
                "end": len(sequence),
                "score": 0.85,
                "type": "attention_analysis",
                "maps": {
                    "start": 0,
                    "end": len(sequence),
                    "score": 0.8,
                    "type": "attention_maps",
                    "data": torch.randn(1, 1, len(sequence), len(sequence))
                }
            }
        }

        result = esm_handler.get_attention_maps(sequence)

        assert isinstance(result, dict)
        assert "attention_maps" in result
        assert "start" in result
        assert "end" in result
        assert "score" in result
        assert isinstance(result["attention_maps"], torch.Tensor)
        assert result["attention_maps"].shape[-1] == len(sequence)
        assert result["attention_maps"].shape[-2] == len(sequence)

@pytest.mark.parametrize("sequence,window_size", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", 5),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", 7),
])
def test_analyze_sequence_windows(esm_handler, sequence, window_size):
    """Test sequence window analysis."""
    with patch.object(esm_handler.model, 'forward') as mock_forward:
        # Mock model output
        mock_forward.return_value = {
            "representations": {
                33: torch.randn(1, len(sequence), 1280),
                "start": 0,
                "end": len(sequence),
                "score": 0.9,
                "type": "representation"
            },
            "logits": torch.randn(1, len(sequence), 33),
            "start": 0,
            "end": len(sequence),
            "score": 0.9,
            "type": "window_analysis",
            "windows": [{
                "start": i,
                "end": i + window_size,
                "score": 0.85,
                "type": "sequence_window",
                "features": {
                    "start": i,
                    "end": i + window_size,
                    "score": 0.8,
                    "type": "window_features",
                    "data": torch.randn(window_size, 1280)
                }
            } for i in range(len(sequence) - window_size + 1)]
        }

        result = esm_handler.analyze_sequence_windows(sequence, window_size)

        assert isinstance(result, dict)
        assert "windows" in result
        assert isinstance(result["windows"], list)
        assert len(result["windows"]) == max(1, len(sequence) - window_size + 1)
        assert "start" in result
        assert "end" in result
        assert "score" in result
        assert "type" in result

        for window in result["windows"]:
            assert isinstance(window, dict)
            assert "start" in window
            assert "end" in window
            assert "score" in window
            assert "type" in window
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
            {
                "representations": {33: torch.randn(1, len(sequence1), 1280)},
                "start": 0,
                "end": len(sequence1),
                "score": 0.9,
                "type": "sequence_comparison",
                "similarity": 0.85,
                "aligned_regions": [{
                    "start": 0,
                    "end": 10,
                    "score": 0.8,
                    "type": "alignment"
                }]
            },
            {
                "representations": {33: torch.randn(1, len(sequence2), 1280)},
                "start": 0,
                "end": len(sequence2),
                "score": 0.9,
                "type": "sequence_comparison",
                "similarity": 0.85,
                "aligned_regions": [{
                    "start": 0,
                    "end": 10,
                    "score": 0.8,
                    "type": "alignment"
                }]
            }
        ]

        result = esm_handler.compare_sequences(sequence1, sequence2)

        assert isinstance(result, dict)
        assert "similarity_score" in result
        assert "start" in result
        assert "end" in result
        assert "score" in result
        assert "type" in result
        assert isinstance(result["similarity_score"], float)
        assert 0 <= result["similarity_score"] <= 1

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
            "logits": {
                "start": 0,
                "end": len(sequence),
                "score": 0.85,
                "type": "logits_output",
                "values": torch.randn(1, len(sequence), 33)
            },
            "start": 0,
            "end": len(sequence),
            "score": 0.9,
            "type": "confidence_calculation",
            "confidence_scores": {
                "start": 0,
                "end": len(sequence),
                "score": 0.95,
                "type": "confidence_analysis",
                "values": np.random.rand(len(sequence))
            }
        }

        result = esm_handler.calculate_confidence_scores(sequence)

        assert isinstance(result, dict)
        assert "confidence_scores" in result
        assert "start" in result
        assert "end" in result
        assert "score" in result
        assert "type" in result
        assert isinstance(result["confidence_scores"], np.ndarray)
        assert len(result["confidence_scores"]) == len(sequence)
        assert all(0 <= score <= 1 for score in result["confidence_scores"])
