import pytest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from models.esm_utils import ESMModelHandler
from tests.conftest import create_mock_method, create_mock_result

@pytest.fixture
def esm_handler(mocker):
    """Fixture for creating an ESMModelHandler instance with mocked ESM model."""
    with patch('models.esm_utils.esm') as mock_esm:
        # Mock ESM model and alphabet with proper structure
        mock_model_result = {
            'start': 0,
            'end': 100,
            'score': 0.95,
            'type': 'esm_model',
            'model': create_mock_result(mocker, {
                'start': 0,
                'end': 100,
                'score': 0.95,
                'type': 'model_config'
            })
        }
        mock_model = create_mock_method(mocker, mock_model_result)
        mock_alphabet = create_mock_method(mocker, {
            'start': 0,
            'end': 100,
            'score': 0.95,
            'type': 'alphabet',
            'tokens': ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V']
        })

        # Mock batch converter
        def batch_converter_return(data):
            sequence = data[0][1]
            batch_tokens = torch.randn(1, len(sequence) + 2, 33)
            return (
                create_mock_result(mocker, {
                    "sequence": sequence,
                    "start": 0,
                    "end": len(sequence),
                    "score": 0.95,
                    "type": "batch_labels"
                }),
                [sequence],
                batch_tokens
            )

        batch_converter = create_mock_method(mocker, {
            'start': 0,
            'end': 100,
            'score': 0.95,
            'type': 'batch_converter',
            'converter': batch_converter_return
        })
        mock_alphabet.get_batch_converter = create_mock_method(mocker, {
            'start': 0,
            'end': 100,
            'score': 0.95,
            'type': 'batch_converter_getter',
            'converter': batch_converter
        })

        # Configure ESM mock
        mock_esm.pretrained.esm2_t33_650M_UR50D = create_mock_method(mocker, {
            'start': 0,
            'end': 100,
            'score': 0.95,
            'type': 'pretrained_model',
            'model': mock_model,
            'alphabet': mock_alphabet
        })

        # Create handler instance
        handler = ESMModelHandler()
        setattr(handler, 'model', mock_model)
        setattr(handler, 'alphabet', mock_alphabet)
        return handler

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_get_sequence_embeddings(mocker, esm_handler, sequence):
    """Test sequence embedding generation."""
    mock_result = create_mock_result(mocker, {
        "representations": create_mock_result(mocker, {
            33: torch.randn(1, len(sequence), 1280),
            "start": 0,
            "end": len(sequence),
            "score": 0.9,
            "type": "representation"
        }),
        "logits": create_mock_result(mocker, {
            "values": torch.randn(1, len(sequence), 33),
            "start": 0,
            "end": len(sequence),
            "score": 0.85,
            "type": "logits"
        }),
        "attentions": create_mock_result(mocker, {
            "values": torch.randn(1, len(sequence), len(sequence)),
            "start": 0,
            "end": len(sequence),
            "score": 0.8,
            "type": "attention"
        }),
        "start": 0,
        "end": len(sequence),
        "score": 0.9,
        "type": "embedding"
    })
    mock_forward = create_mock_method(mocker, mock_result)
    setattr(esm_handler.model, 'forward', mock_forward)

    result = esm_handler.get_sequence_embeddings(sequence)

    # Verify dictionary structure and required fields
    assert isinstance(result, dict)
    required_fields = ["embeddings", "start", "end", "score", "type"]
    assert all(key in result for key in required_fields), f"Missing required fields. Expected {required_fields}, got {list(result.keys())}"

    # Verify field types and values
    embeddings = result.get("embeddings")
    assert isinstance(embeddings, torch.Tensor), "Embeddings must be a torch.Tensor"
    assert embeddings.shape[1] == 1280, f"Expected embedding dimension 1280, got {embeddings.shape[1]}"  # ESM-2 embedding dimension
    assert embeddings.shape[0] == len(sequence), f"Expected sequence length {len(sequence)}, got {embeddings.shape[0]}"
    assert isinstance(result.get("start"), int), "Start must be an integer"
    assert isinstance(result.get("end"), int), "End must be an integer"
    assert isinstance(result.get("score"), float), "Score must be a float"
    assert isinstance(result.get("type"), str), "Type must be a string"

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_get_attention_maps(mocker, esm_handler, sequence):
    """Test attention map generation."""
    mock_result = create_mock_result(mocker, {
        "attentions": create_mock_result(mocker, {
            "start": 0,
            "end": len(sequence),
            "score": 0.9,
            "type": "attention_tensor",
            "data": torch.randn(1, 1, len(sequence), len(sequence))
        }),
        "start": 0,
        "end": len(sequence),
        "score": 0.9,
        "type": "attention",
        "attention_maps": create_mock_result(mocker, {
            "start": 0,
            "end": len(sequence),
            "score": 0.85,
            "type": "attention_analysis",
            "maps": create_mock_result(mocker, {
                "start": 0,
                "end": len(sequence),
                "score": 0.8,
                "type": "attention_maps",
                "data": torch.randn(1, 1, len(sequence), len(sequence))
            })
        })
    })
    mock_forward = create_mock_method(mocker, mock_result)
    setattr(esm_handler.model, 'forward', mock_forward)

    result = esm_handler.get_attention_maps(sequence)

    assert isinstance(result, dict)
    assert all(key in result for key in ['attention_maps', 'start', 'end', 'score', 'type'])
    assert isinstance(result["start"], int)
    assert isinstance(result["end"], int)
    assert isinstance(result["score"], float)
    assert isinstance(result["type"], str)
    attention_maps = result["attention_maps"]
    assert isinstance(attention_maps, torch.Tensor)
    assert attention_maps.shape[-1] == len(sequence)
    assert attention_maps.shape[-2] == len(sequence)

@pytest.mark.parametrize("sequence,window_size", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", 5),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", 7),
])
def test_analyze_sequence_windows(mocker, esm_handler, sequence, window_size):
    """Test sequence window analysis."""
    mock_result = create_mock_result(mocker, {
        "representations": create_mock_result(mocker, {
            33: torch.randn(1, len(sequence), 1280),
            "start": 0,
            "end": len(sequence),
            "score": 0.9,
            "type": "representation"
        }),
        "logits": torch.randn(1, len(sequence), 33),
        "start": 0,
        "end": len(sequence),
        "score": 0.9,
        "type": "window_analysis",
        "windows": [create_mock_result(mocker, {
            "start": i,
            "end": i + window_size,
            "score": 0.85,
            "type": "sequence_window",
            "features": create_mock_result(mocker, {
                "start": i,
                "end": i + window_size,
                "score": 0.8,
                "type": "window_features",
                "data": torch.randn(window_size, 1280)
            })
        }) for i in range(len(sequence) - window_size + 1)]
    })
    mock_forward = create_mock_method(mocker, mock_result)
    setattr(esm_handler.model, 'forward', mock_forward)

    result = esm_handler.analyze_sequence_windows(sequence, window_size)

    # Verify base dictionary structure
    assert isinstance(result, dict)
    assert all(key in result for key in ["windows", "start", "end", "score", "type"])

    # Verify windows list
    windows = result["windows"]
    assert isinstance(windows, list)
    assert len(windows) == max(1, len(sequence) - window_size + 1)

    # Verify field types
    assert isinstance(result["start"], int)
    assert isinstance(result["end"], int)
    assert isinstance(result["score"], float)
    assert isinstance(result["type"], str)

    # Verify window structures
    for window in windows:
        assert isinstance(window, dict)
        assert all(key in window for key in ["start", "end", "score", "type"])
        assert isinstance(window["start"], int)
        assert isinstance(window["end"], int)
        assert isinstance(window["score"], float)
        assert 0 <= window["score"] <= 1

@pytest.mark.parametrize("sequence1,sequence2", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"),
])
def test_compare_sequences(mocker, esm_handler, sequence1, sequence2):
    """Test sequence comparison functionality."""
    mock_result_1 = create_mock_result(mocker, {
        "representations": {33: torch.randn(1, len(sequence1), 1280)},
        "start": 0,
        "end": len(sequence1),
        "score": 0.9,
        "type": "sequence_comparison",
        "similarity": 0.85,
        "aligned_regions": [create_mock_result(mocker, {
            "start": 0,
            "end": 10,
            "score": 0.8,
            "type": "alignment"
        })]
    })
    mock_result_2 = create_mock_result(mocker, {
        "representations": {33: torch.randn(1, len(sequence2), 1280)},
        "start": 0,
        "end": len(sequence2),
        "score": 0.9,
        "type": "sequence_comparison",
        "similarity": 0.85,
        "aligned_regions": [create_mock_result(mocker, {
            "start": 0,
            "end": 10,
            "score": 0.8,
            "type": "alignment"
        })]
    })
    mock_forward = create_mock_method(mocker, mock_result_1)
    setattr(esm_handler.model, 'forward', mock_forward)

    result = esm_handler.compare_sequences(sequence1, sequence2)

    assert isinstance(result, dict)
    assert all(key in result for key in ["similarity_score", "start", "end", "score", "type"])
    assert isinstance(result["similarity_score"], float)
    assert isinstance(result["start"], int)
    assert isinstance(result["end"], int)
    assert isinstance(result["score"], float)
    assert isinstance(result["type"], str)
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
def test_confidence_scores(mocker, esm_handler, sequence):
    """Test confidence score calculation."""
    mock_result = create_mock_result(mocker, {
        "logits": create_mock_result(mocker, {
            "start": 0,
            "end": len(sequence),
            "score": 0.85,
            "type": "logits_output",
            "values": torch.randn(1, len(sequence), 33)
        }),
        "start": 0,
        "end": len(sequence),
        "score": 0.9,
        "type": "confidence_calculation",
        "confidence_scores": create_mock_result(mocker, {
            "start": 0,
            "end": len(sequence),
            "score": 0.95,
            "type": "confidence_analysis",
            "values": np.random.rand(len(sequence))
        })
    })
    mock_forward = mocker.MagicMock(side_effect=lambda *args, **kwargs: mock_result)
    setattr(esm_handler.model, 'forward', mock_forward)

    result = esm_handler.calculate_confidence_scores(sequence)

    # Verify dictionary structure and required fields
    assert isinstance(result, dict)
    assert all(key in result for key in ['confidence_scores', 'start', 'end', 'score', 'type'])

    # Verify field types
    assert isinstance(result["start"], int)
    assert isinstance(result["end"], int)
    assert isinstance(result["score"], float)
    assert isinstance(result["type"], str)

    # Verify confidence scores
    confidence_scores = result["confidence_scores"]
    assert isinstance(confidence_scores, np.ndarray)
    assert len(confidence_scores) == len(sequence)
    assert all(0 <= score <= 1 for score in confidence_scores)
