"""
Tests for UnifiedProteinModel integration.
"""
import os
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from models.generative.unified_model import UnifiedProteinModel
from models.generative.api_integration import APIManager

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'OPENAI_APIKEY': 'test_openai_key',
        'CLAUDE_API_KEY': 'test_claude_key',
        'GEMINI_API': 'test_gemini_key'
    }):
        yield

@pytest.fixture
def mock_api_manager():
    """Mock APIManager for testing."""
    with patch('models.generative.unified_model.APIManager') as mock:
        manager = MagicMock()
        manager.apis = {
            'openai': MagicMock(),
            'claude': MagicMock(),
            'gemini': MagicMock()
        }
        mock.return_value = manager
        yield mock

@pytest.fixture
def unified_model(mock_env_vars, mock_api_manager):
    """Create UnifiedProteinModel instance for testing."""
    return UnifiedProteinModel(use_gpu=False)

def test_initialization(unified_model):
    """Test UnifiedProteinModel initialization."""
    assert unified_model.device == torch.device('cpu')
    assert hasattr(unified_model, 'local_generator')
    assert hasattr(unified_model, 'structure_predictor')
    assert hasattr(unified_model, 'screening_model')
    assert hasattr(unified_model, 'api_manager')

def test_generate_sequence(unified_model):
    """Test sequence generation with both local and API models."""
    result = unified_model.generate_sequence(
        "Generate a stable protein sequence",
        use_apis=True
    )
    assert 'local' in result
    assert all(api in result for api in ['openai', 'claude', 'gemini'])

def test_predict_structure(unified_model):
    """Test structure prediction with both local and API models."""
    sequence = "MVKVGVNG"
    result = unified_model.predict_structure(sequence, use_apis=True)
    assert 'local' in result
    assert all(api in result for api in ['openai', 'claude', 'gemini'])

def test_analyze_stability(unified_model):
    """Test stability analysis with both local and API models."""
    sequence = "MVKVGVNG"
    result = unified_model.analyze_stability(sequence, use_apis=True)
    assert 'local' in result
    assert 'stability_score' in result['local']
    assert all(api in result for api in ['openai', 'claude', 'gemini'])

def test_screen_compounds(unified_model):
    """Test compound screening with both local and API models."""
    # Mock protein structure tensor
    structure = torch.randn(100, 3)
    compounds = ["CC(=O)O", "CCO"]

    result = unified_model.screen_compounds(structure, compounds, use_apis=True)
    assert 'local' in result
    assert all(api in result for api in ['openai', 'claude', 'gemini'])

def test_ensemble_predict(unified_model):
    """Test ensemble predictions combining local and API results."""
    sequence = "MVKVGVNG"
    weights = {
        'local': 0.4,
        'openai': 0.2,
        'claude': 0.2,
        'gemini': 0.2
    }

    result = unified_model.ensemble_predict(sequence, weights=weights)
    assert 'structure' in result
    assert 'stability' in result
    assert 'weights' in result
    assert 'confidence_score' in result
    assert abs(sum(result['weights'].values()) - 1.0) < 1e-6

@pytest.mark.parametrize("use_apis", [True, False])
def test_api_integration_toggle(unified_model, use_apis):
    """Test toggling API integration on/off."""
    sequence = "MVKVGVNG"
    result = unified_model.predict_structure(sequence, use_apis=use_apis)
    assert 'local' in result
    if use_apis:
        assert len(result) > 1
    else:
        assert len(result) == 1

def test_error_handling(unified_model, mock_api_manager):
    """Test error handling for API failures."""
    # Make one API raise an exception
    mock_api_manager.return_value.apis['openai'].generate.side_effect = Exception("API Error")

    result = unified_model.generate_sequence("Test sequence", use_apis=True)
    assert 'local' in result
    assert 'error' in result['openai']
    assert isinstance(result['openai']['error'], str)

def test_gpu_support():
    """Test GPU support when available."""
    with patch('torch.cuda.is_available', return_value=True):
        model = UnifiedProteinModel(use_gpu=True)
        assert model.device == torch.device('cuda')


    with patch('torch.cuda.is_available', return_value=False):
        model = UnifiedProteinModel(use_gpu=True)
        assert model.device == torch.device('cpu')
