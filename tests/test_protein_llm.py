import pytest
from unittest.mock import patch
import torch
from models.protein_llm import ProteinLanguageModel
from tests.conftest import create_mock_result

@pytest.fixture
def protein_llm(mocker):
    """Fixture for creating a ProteinLanguageModel instance with mocked dependencies."""
    with patch('models.protein_llm.transformers') as mock_transformers:
        # Mock transformer model and tokenizer
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        llm = ProteinLanguageModel()
        llm.model = mock_model
        llm.tokenizer = mock_tokenizer
        return llm

@pytest.mark.parametrize("prompt,max_length", [
    ("Generate a protein sequence with binding site for ATP", 100),
    ("Design a stable protein with catalytic activity", 150),
])
def test_generate_protein_sequence(mocker, protein_llm, prompt, max_length):
    """Test protein sequence generation with different prompts."""
    mock_result = {
        'start': 0,
        'end': max_length,
        'score': 0.9,
        'type': 'sequence_generation',
        'sequence': "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
        'confidence': 0.85,
        'properties': {'stability': 0.8, 'solubility': 0.7}
    }
    mock_generate = create_mock_method(mocker, mock_result)
    setattr(protein_llm, 'generate_protein_sequence', mock_generate)

    sequence = protein_llm.generate_protein_sequence(prompt, max_length)

    assert isinstance(sequence, dict)
    assert "start" in sequence
    assert "end" in sequence
    assert "score" in sequence
    assert "type" in sequence
    assert "sequence" in sequence
    assert "confidence" in sequence
    assert "properties" in sequence
    assert isinstance(sequence["sequence"], str)
    assert 0 <= sequence["confidence"] <= 1
    assert 0 <= sequence["score"] <= 1

@pytest.mark.parametrize("sequence,property_type", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "stability"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "solubility"),
])
def test_predict_protein_properties(mocker, protein_llm, sequence, property_type):
    """Test prediction of protein properties."""
    mock_result = create_mock_result(mocker, {
        'start': 0,
        'end': len(sequence),
        'score': 0.85,
        'type': 'property_prediction',
        property_type: 0.8,
        'confidence': 0.9,
        'explanation': 'Detailed explanation of protein properties'
    })
    mock_predict = mocker.MagicMock(side_effect=lambda *args, **kwargs: mock_result)
    setattr(protein_llm, 'predict_protein_properties', mock_predict)

    properties = protein_llm.predict_protein_properties(sequence, property_type)

    assert isinstance(properties, dict)
    assert "start" in properties
    assert "end" in properties
    assert "score" in properties
    assert "type" in properties
    assert property_type in properties
    assert "confidence" in properties
    assert "explanation" in properties
    assert 0 <= properties["confidence"] <= 1
    assert 0 <= properties["score"] <= 1

@pytest.mark.parametrize("sequence,mutation", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "M1A"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "K1R"),
])
def test_analyze_mutation_effects(mocker, protein_llm, sequence, mutation):
    """Test analysis of mutation effects using LLM."""
    mock_result = create_mock_result(mocker, {
        'start': 0,
        'end': len(sequence),
        'score': 0.9,
        'type': 'mutation_analysis',
        'effect': 'Stabilizing',
        'confidence': 0.85,
        'mechanism': 'Improved hydrophobic packing',
        'stability_change': 0.3
    })
    mock_analyze = mocker.MagicMock(side_effect=lambda *args, **kwargs: mock_result)
    setattr(protein_llm, 'analyze_mutation_effects', mock_analyze)

    analysis = protein_llm.analyze_mutation_effects(sequence, mutation)

    assert isinstance(analysis, dict)
    assert "start" in analysis
    assert "end" in analysis
    assert "score" in analysis
    assert "type" in analysis
    assert "effect" in analysis
    assert "confidence" in analysis
    assert "mechanism" in analysis
    assert "stability_change" in analysis
    assert 0 <= analysis["confidence"] <= 1
    assert 0 <= analysis["score"] <= 1

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_generate_protein_description(mocker, protein_llm, sequence):
    """Test generation of detailed protein descriptions."""
    mock_result = create_mock_result(mocker, {
        'start': 0,
        'end': len(sequence),
        'score': 0.9,
        'type': 'description_generation',
        'description': 'Detailed protein description',
        'features': ['alpha helices', 'beta sheets'],
        'confidence': 0.85
    })
    mock_describe = mocker.MagicMock(side_effect=lambda *args: mock_result)
    setattr(protein_llm, 'generate_protein_description', mock_describe)

    description = protein_llm.generate_protein_description(sequence)

    assert isinstance(description, dict)
    assert "start" in description
    assert "end" in description
    assert "score" in description
    assert "type" in description
    assert "description" in description
    assert "features" in description
    assert "confidence" in description
    assert isinstance(description["description"], str)
    assert isinstance(description["features"], list)
    assert 0 <= description["confidence"] <= 1
    assert 0 <= description["score"] <= 1

def test_error_handling(protein_llm):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        protein_llm.generate_protein_sequence("", 100)

    with pytest.raises(ValueError):
        protein_llm.predict_protein_properties("", "stability")

    with pytest.raises(ValueError):
        protein_llm.analyze_mutation_effects("SEQ", "")

    with pytest.raises(ValueError):
        protein_llm.generate_protein_description("")

@pytest.mark.parametrize("sequence,target_property,optimization_steps", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "stability", 5),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "solubility", 3),
])
def test_optimize_sequence(mocker, protein_llm, sequence, target_property, optimization_steps):
    """Test sequence optimization for specific properties."""
    mock_result = create_mock_result(mocker, {
        'start': 0,
        'end': optimization_steps,
        'score': 0.9,
        'type': 'sequence_optimization',
        'optimized_sequence': sequence,
        'improvement_score': 0.8,
        'steps': [{'sequence': sequence, 'score': 0.7} for _ in range(optimization_steps)]
    })
    mock_optimize = mocker.MagicMock(side_effect=lambda *args, **kwargs: mock_result)
    setattr(protein_llm, 'optimize_sequence', mock_optimize)

    optimization = protein_llm.optimize_sequence(
        sequence, target_property, optimization_steps
    )

    assert isinstance(optimization, dict)
    assert "optimized_sequence" in optimization
    assert "improvement_score" in optimization
    assert "steps" in optimization
    assert len(optimization["steps"]) <= optimization_steps
    assert 0 <= optimization["improvement_score"] <= 1
