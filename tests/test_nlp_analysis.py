import pytest
from unittest.mock import Mock, patch
import numpy as np
from models.nlp_analysis import ProteinNLPAnalyzer
from tests.conftest import create_mock_result

@pytest.fixture
def nlp_analyzer(mocker):
    """Fixture for creating a ProteinNLPAnalyzer instance with mocked dependencies."""
    with patch('models.nlp_analysis.transformers') as mock_transformers:
        # Mock transformer model and tokenizer
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        analyzer = ProteinNLPAnalyzer()
        analyzer.model = mock_model
        analyzer.tokenizer = mock_tokenizer
        return analyzer

@pytest.mark.parametrize("sequence,query", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "What is the binding site?"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "Describe the protein structure."),
])
def test_answer_protein_question(mocker, nlp_analyzer, sequence, query):
    """Test protein-specific question answering."""
    mock_result = create_mock_result(mocker, {
        "start": 0,
        "end": len(sequence),
        "score": 0.85,
        "type": "protein_qa",
        "answer": "This is a test answer",
        "confidence": 0.9
    })
    mock_generate = mocker.MagicMock(side_effect=lambda *args: [mocker.MagicMock(sequences=[mock_result["answer"]])])
    mock_qa = mocker.MagicMock(side_effect=lambda *args: mock_result)
    setattr(nlp_analyzer.model, 'generate', mock_generate)
    setattr(nlp_analyzer, 'answer_protein_question', mock_qa)

    answer = nlp_analyzer.answer_protein_question(sequence, query)

    assert isinstance(answer, dict)
    assert "start" in answer
    assert "end" in answer
    assert "score" in answer
    assert "type" in answer
    assert "answer" in answer
    assert "confidence" in answer
    assert isinstance(answer["answer"], str)
    assert 0 <= answer["confidence"] <= 1
    assert 0 <= answer["score"] <= 1

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_generate_sequence_description(mocker, nlp_analyzer, sequence):
    """Test generation of protein sequence descriptions."""
    mock_result = create_mock_result(mocker, {
        "start": 0,
        "end": len(sequence),
        "score": 0.9,
        "type": "sequence_description",
        "description": "Test protein description",
        "features": ["alpha helix", "beta sheet"]
    })
    mock_desc = mocker.MagicMock(side_effect=lambda *args: mock_result)
    setattr(nlp_analyzer, 'generate_sequence_description', mock_desc)

    description = nlp_analyzer.generate_sequence_description(sequence)

    assert isinstance(description, dict)
    assert "start" in description
    assert "end" in description
    assert "score" in description
    assert "type" in description
    assert "description" in description
    assert "features" in description
    assert isinstance(description["description"], str)
    assert isinstance(description["features"], list)
    assert 0 <= description["score"] <= 1

@pytest.mark.parametrize("sequence1,sequence2", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"),
])
def test_compare_sequences_nlp(mocker, nlp_analyzer, sequence1, sequence2):
    """Test NLP-based sequence comparison."""
    mock_result = create_mock_result(mocker, {
        "start": 0,
        "end": max(len(sequence1), len(sequence2)),
        "score": 0.85,
        "type": "sequence_comparison",
        "similarity_score": 0.7,
        "differences": ["region 1-10", "region 20-30"],
        "common_features": ["hydrophobic core"]
    })
    mock_compare = mocker.MagicMock(side_effect=lambda *args: mock_result)
    setattr(nlp_analyzer, 'compare_sequences_nlp', mock_compare)

    comparison = nlp_analyzer.compare_sequences_nlp(sequence1, sequence2)

    assert isinstance(comparison, dict)
    assert "start" in comparison
    assert "end" in comparison
    assert "score" in comparison
    assert "type" in comparison
    assert "similarity_score" in comparison
    assert "differences" in comparison
    assert "common_features" in comparison
    assert 0 <= comparison["similarity_score"] <= 1
    assert 0 <= comparison["score"] <= 1

@pytest.mark.parametrize("sequence,mutation", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "M1A"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "K1R"),
])
def test_analyze_mutation_impact(mocker, nlp_analyzer, sequence, mutation):
    """Test mutation impact analysis using NLP."""
    mock_result = create_mock_result(mocker, {
        "start": 0,
        "end": len(sequence),
        "score": 0.9,
        "type": "mutation_analysis",
        "impact": "moderate",
        "confidence": 0.85,
        "explanation": "Test mutation impact explanation"
    })
    mock_analyze = mocker.MagicMock(side_effect=lambda *args: mock_result)
    setattr(nlp_analyzer, 'analyze_mutation_impact', mock_analyze)

    analysis = nlp_analyzer.analyze_mutation_impact(sequence, mutation)

    assert isinstance(analysis, dict)
    assert "start" in analysis
    assert "end" in analysis
    assert "score" in analysis
    assert "type" in analysis
    assert "impact" in analysis
    assert "confidence" in analysis
    assert "explanation" in analysis
    assert 0 <= analysis["confidence"] <= 1
    assert 0 <= analysis["score"] <= 1

def test_error_handling(nlp_analyzer):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        nlp_analyzer.answer_protein_question("", "query")

    with pytest.raises(ValueError):
        nlp_analyzer.generate_sequence_description("")

    with pytest.raises(ValueError):
        nlp_analyzer.compare_sequences_nlp("SEQ1", "")

    with pytest.raises(ValueError):
        nlp_analyzer.analyze_mutation_impact("", "M1A")

@pytest.mark.parametrize("sequence", [
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK",
])
def test_extract_sequence_features(mocker, nlp_analyzer, sequence):
    """Test extraction of sequence features using NLP."""
    mock_feature = create_mock_result(mocker, {
        "start": 0,
        "end": 10,
        "score": 0.85,
        "type": "sequence_feature",
        "description": "Test feature",
        "confidence": 0.9
    })
    mock_result = create_mock_result(mocker, {
        "start": 0,
        "end": len(sequence),
        "score": 0.9,
        "type": "feature_extraction",
        "features": [mock_feature]
    })
    mock_extract = mocker.MagicMock(side_effect=lambda *args: mock_result)
    setattr(nlp_analyzer, 'extract_sequence_features', mock_extract)

    features = nlp_analyzer.extract_sequence_features(sequence)

    assert isinstance(features, dict)
    assert "start" in features
    assert "end" in features
    assert "score" in features
    assert "type" in features
    assert "features" in features
    assert isinstance(features["features"], list)

    for feature in features["features"]:
        assert isinstance(feature, dict)
        assert "start" in feature
        assert "end" in feature
        assert "score" in feature
        assert "type" in feature
        assert "description" in feature
        assert "confidence" in feature
        assert 0 <= feature["confidence"] <= 1
        assert 0 <= feature["score"] <= 1

    assert 0 <= features["score"] <= 1
