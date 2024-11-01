"""
Tests for API integration module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from models.generative.api_integration import (
    OpenAIAPI, ClaudeAPI, GeminiAPI, APIManager
)

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'OPENAI_APIKEY': 'test_openai_key',
        'CLAUDE_API_KEY': 'test_claude_key',
        'GEMINI_API': 'test_gemini_key'
    }):
        yield

def test_openai_api_initialization(mock_env_vars):
    """Test OpenAI API initialization."""
    api = OpenAIAPI()
    assert api.api_key == 'test_openai_key'

def test_claude_api_initialization(mock_env_vars):
    """Test Claude API initialization."""
    api = ClaudeAPI()
    assert api.api_key == 'test_claude_key'

def test_gemini_api_initialization(mock_env_vars):
    """Test Gemini API initialization."""
    api = GeminiAPI()
    assert api.api_key == 'test_gemini_key'

def test_api_manager_initialization(mock_env_vars):
    """Test APIManager initialization with all APIs."""
    manager = APIManager()
    assert 'openai' in manager.apis
    assert 'claude' in manager.apis
    assert 'gemini' in manager.apis

@pytest.mark.parametrize("api_name", ["openai", "claude", "gemini"])
def test_api_generation(mock_env_vars, api_name):
    """Test generation method for each API."""
    manager = APIManager()
    api = manager.get_api(api_name)
    response = api.generate("Test prompt")
    assert isinstance(response, str)

@pytest.mark.parametrize("api_name", ["openai", "claude", "gemini"])
def test_protein_analysis(mock_env_vars, api_name):
    """Test protein analysis for each API."""
    manager = APIManager()
    api = manager.get_api(api_name)
    result = api.analyze_protein("MVKVGVNG")
    assert isinstance(result, dict)

def test_analyze_with_all(mock_env_vars):
    """Test analyzing protein with all available APIs."""
    manager = APIManager()
    results = manager.analyze_with_all("MVKVGVNG")
    assert isinstance(results, dict)
    assert all(api in results for api in ['openai', 'claude', 'gemini'])
