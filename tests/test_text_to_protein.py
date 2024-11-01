"""
Tests for Text to Protein Generator integration.
"""
import os
import pytest
import torch
import asyncio
from unittest.mock import patch, MagicMock
import google.generativeai as genai

from models.generative.text_to_protein import TextToProteinGenerator

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'Gemini_api': 'test_gemini_key',
        'OPENAI_APIKEY': 'test_openai_key',
        'CLAUDE_API_KEY': 'test_claude_key'
    }):
        yield

@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response."""
    response = MagicMock()
    response.text = """
    Generated Protein Sequence: MKVGVNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGTVKAENGKLVINGNPITIFQERDPSKIKWGDAGAEYVVESTGVFTTMEKAGAHLQGGAKRVIISAPSADAPMFVMGVNHEKYDNSLKIISNASCTTNCLAPLAKVIHDNFGIVEGLMTTVHAITATQKTVDGPSGKLWRDGRGALQNIIPASTGAAKAVGKVIPELDGKLTGMAFRVPTANVSVVDLTCRLEKPAKYDDIKKVVKQASEGPLKGILGYTEHQVVSSDFNSDTHSSTFDAGAGIALNDHFVKLISWYDNEFGYSNRVVDLMAHMASKE

    Key Features:
    1. Length: 335 amino acids
    2. Contains glycolytic enzyme motifs
    3. Predicted alpha-helical content: ~40%
    4. Stable fold prediction

    Stability Assessment:
    - Predicted to be highly stable
    - Contains multiple stabilizing salt bridges
    - Hydrophobic core well-packed
    """
    return response

@pytest.fixture
def mock_genai():
    """Mock Gemini API."""
    with patch('google.generativeai') as mock:
        model = MagicMock()
        mock.GenerativeModel.return_value = model
        yield mock

@pytest.fixture
def generator(mock_env_vars, mock_genai):
    """Create TextToProteinGenerator instance for testing."""
    return TextToProteinGenerator(use_gpu=False)

def test_initialization(generator):
    """Test TextToProteinGenerator initialization."""
    assert generator.device == torch.device('cpu')
    assert hasattr(generator, 'unified_model')
    assert hasattr(generator, 'gemini_model')

@pytest.mark.asyncio
async def test_generate_protein(generator, mock_gemini_response):
    """Test protein generation from text description."""
    generator.gemini_model.generate_content = MagicMock(
        return_value=mock_gemini_response
    )

    result = await generator.generate_protein(
        "Design a stable glycolytic enzyme"
    )

    assert 'sequence' in result
    assert 'stability_analysis' in result
    assert 'predicted_structure' in result
    assert 'source' in result
    assert result['source'] == 'gemini+local'
    assert len(result['sequence']) > 0
    assert all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in result['sequence'])

@pytest.mark.asyncio
async def test_generate_protein_with_invalid_response(generator):
    """Test protein generation with invalid Gemini response."""
    # Mock Gemini response without valid sequence
    invalid_response = MagicMock()
    invalid_response.text = "No valid protein sequence here"
    generator.gemini_model.generate_content = MagicMock(
        return_value=invalid_response
    )

    result = await generator.generate_protein(
        "Design a stable enzyme"
    )

    assert 'sequence' in result
    assert 'error' in result
    assert result['source'] == 'local_fallback'

def test_validate_sequence(generator):
    """Test sequence validation."""
    # Test valid sequence
    valid_result = generator.validate_sequence("MVKVGVNG")
    assert valid_result['is_valid']
    assert valid_result['length'] == 8
    assert 'stability' in valid_result
    assert 'structure' in valid_result

    # Test invalid sequence
    invalid_result = generator.validate_sequence("MVK1GVNG")
    assert not invalid_result['is_valid']
    assert 'error' in invalid_result
    assert '1' in invalid_result['invalid_residues']

@pytest.mark.asyncio
async def test_batch_generate(generator, mock_gemini_response):
    """Test batch protein generation."""
    generator.gemini_model.generate_content = MagicMock(
        return_value=mock_gemini_response
    )

    descriptions = [
        "Design a stable enzyme",
        "Create a membrane protein"
    ]

    results = generator.batch_generate(descriptions)
    assert len(results) == len(descriptions)
    assert all('sequence' in result for result in results)
    assert all('stability_analysis' in result for result in results)

def test_format_protein_prompt(generator):
    """Test prompt formatting."""
    description = "Design a stable enzyme"
    prompt = generator._format_protein_prompt(description)

    assert description in prompt
    assert "protein sequence" in prompt.lower()
    assert "stability" in prompt.lower()
    assert "amino acid" in prompt.lower()

@pytest.mark.asyncio
async def test_error_handling(generator):
    """Test error handling during generation."""
    # Mock Gemini API to raise an exception
    generator.gemini_model.generate_content = MagicMock(
        side_effect=Exception("API Error")
    )

    result = await generator.generate_protein(
        "Design a stable enzyme"
    )

    assert 'error' in result
    assert result['source'] == 'local_fallback'
    assert 'sequence' in result  # Should still have sequence from local fallback
    assert 'stability_analysis' in result

def test_gpu_support():
    """Test GPU support when available."""
    with patch('torch.cuda.is_available', return_value=True):
        generator = TextToProteinGenerator(use_gpu=True)
        assert generator.device == torch.device('cuda')

    with patch('torch.cuda.is_available', return_value=False):
        generator = TextToProteinGenerator(use_gpu=True)
        assert generator.device == torch.device('cpu')
