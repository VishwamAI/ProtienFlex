import pytest
from unittest.mock import Mock, patch
from models.qa_system import ProteinQASystem

@pytest.fixture
def qa_system():
    """Fixture for creating a ProteinQASystem instance with mocked dependencies."""
    with patch('models.qa_system.transformers') as mock_transformers:
        # Mock transformer model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_transformers.AutoModelForQuestionAnswering.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        system = ProteinQASystem()
        system.model = mock_model
        system.tokenizer = mock_tokenizer
        return system

@pytest.mark.parametrize("question,context", [
    ("What is the binding site?", "The protein contains an ATP binding site at residues 10-20."),
    ("Is this protein stable?", "The protein shows high stability with a melting temperature of 80°C."),
])
def test_answer_question(qa_system, question, context):
    """Test question answering with different inputs."""
    with patch.object(qa_system.model, 'generate') as mock_generate:
        mock_generate.return_value = [Mock(sequences=["This is the answer"])]

        answer = qa_system.answer_question(question, context)

        assert isinstance(answer, dict)
        assert "start" in answer
        assert "end" in answer
        assert "score" in answer
        assert "type" in answer
        assert "answer" in answer
        assert "confidence" in answer
        assert "context_used" in answer
        assert isinstance(answer["answer"], str)
        assert 0 <= answer["confidence"] <= 1
        assert 0 <= answer["score"] <= 1

@pytest.mark.parametrize("sequence,property_query", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "binding sites"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "secondary structure"),
])
def test_analyze_protein_property(qa_system, sequence, property_query):
    """Test protein property analysis through QA system."""
    analysis = qa_system.analyze_protein_property(sequence, property_query)

    assert isinstance(analysis, dict)
    assert "start" in analysis
    assert "end" in analysis
    assert "score" in analysis
    assert "type" in analysis
    assert "analysis" in analysis
    assert "confidence" in analysis
    assert "supporting_evidence" in analysis
    assert isinstance(analysis["analysis"], str)
    assert 0 <= analysis["confidence"] <= 1
    assert 0 <= analysis["score"] <= 1

@pytest.mark.parametrize("sequence1,sequence2", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"),
])
def test_compare_proteins(qa_system, sequence1, sequence2):
    """Test protein comparison through QA system."""
    comparison = qa_system.compare_proteins(sequence1, sequence2)

    assert isinstance(comparison, dict)
    assert "start" in comparison
    assert "end" in comparison
    assert "score" in comparison
    assert "type" in comparison
    assert "comparison" in comparison
    assert "similarities" in comparison
    assert "differences" in comparison
    assert "confidence" in comparison
    assert isinstance(comparison["similarities"], list)
    assert isinstance(comparison["differences"], list)

def test_error_handling(qa_system):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        qa_system.answer_question("", "context")

    with pytest.raises(ValueError):
        qa_system.analyze_protein_property("", "property")

    with pytest.raises(ValueError):
        qa_system.compare_proteins("SEQ1", "")

@pytest.mark.parametrize("question,sequence,mutation", [
    ("What is the effect of this mutation?", "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "M1A"),
    ("Is this mutation stabilizing?", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "K1R"),
])
def test_analyze_mutation_impact(qa_system, question, sequence, mutation):
    """Test mutation impact analysis through QA system."""
    analysis = qa_system.analyze_mutation_impact(question, sequence, mutation)

    assert isinstance(analysis, dict)
    assert "start" in analysis
    assert "end" in analysis
    assert "score" in analysis
    assert "type" in analysis
    assert "impact" in analysis
    assert "explanation" in analysis
    assert "confidence" in analysis
    assert "evidence" in analysis
    assert 0 <= analysis["confidence"] <= 1
    assert 0 <= analysis["score"] <= 1

@pytest.mark.parametrize("sequence,query_type", [
    ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "structure"),
    ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "function"),
])
def test_generate_protein_report(qa_system, sequence, query_type):
    """Test generation of comprehensive protein reports."""
    report = qa_system.generate_protein_report(sequence, query_type)

    assert isinstance(report, dict)
    assert "summary" in report
    assert "details" in report
    assert "confidence" in report
    assert isinstance(report["summary"], str)
    assert isinstance(report["details"], dict)
    assert 0 <= report["confidence"] <= 1
