import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from proteinflex.models.analysis.qa_system import ProteinQASystem

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    # Mock tokenizer behavior
    def mock_decode(token_ids, **kwargs):
        # Return different answers based on input length
        if isinstance(token_ids, torch.Tensor) and token_ids.numel() == 1:
            return "Unable to process question"
        return "predicted answer"

    def mock_call(*args, **kwargs):
        # Handle error cases
        question = kwargs.get('question', args[0] if args else None)
        context = kwargs.get('context', args[1] if len(args) > 1 else None)

        if question is None or context is None or not question or not context:
            # Return empty tensors for error cases
            return {
                'input_ids': torch.tensor([[0]]),
                'attention_mask': torch.tensor([[0]])
            }

        # Return normal tensors for valid inputs
        return {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]]),
            'token_type_ids': torch.tensor([[0, 0, 0, 1, 1]])
        }

    tokenizer.decode = Mock(side_effect=mock_decode)
    tokenizer.__call__ = mock_call
    return tokenizer

@pytest.fixture
def mock_model():
    model = Mock()
    # Configure model.eval() and training attribute
    model.eval = Mock()
    model.training = False

    # Mock model outputs with different responses for valid and invalid inputs
    def mock_call(*args, **kwargs):
        inputs = kwargs.get('input_ids', None)
        if inputs is None or inputs.shape[1] == 1:  # Error case
            outputs = Mock()
            outputs.start_logits = torch.tensor([[0.0]])
            outputs.end_logits = torch.tensor([[0.0]])
            return outputs

        # Normal case - return logits that will produce a valid answer
        outputs = Mock()
        outputs.start_logits = torch.tensor([[0.1, 0.2, 0.8, 0.1, 0.1]])  # Position 2 has highest score
        outputs.end_logits = torch.tensor([[0.1, 0.1, 0.1, 0.9, 0.1]])    # Position 3 has highest score
        outputs.keys = Mock(return_value=['start_logits', 'end_logits'])
        outputs.__getitem__ = lambda self, key: getattr(self, key)
        return outputs

    model.__call__ = mock_call
    return model

@pytest.fixture
def qa_system(mock_tokenizer, mock_model):
    with patch('proteinflex.models.analysis.qa_system.AutoTokenizer') as mock_auto_tokenizer, \
         patch('proteinflex.models.analysis.qa_system.AutoModelForQuestionAnswering') as mock_auto_model:

        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        return ProteinQASystem()

def test_init_success():
    """Test successful initialization of ProteinQASystem"""
    with patch('proteinflex.models.analysis.qa_system.AutoTokenizer') as mock_tokenizer, \
         patch('proteinflex.models.analysis.qa_system.AutoModelForQuestionAnswering') as mock_model:

        qa = ProteinQASystem()
        assert qa.tokenizer == mock_tokenizer.from_pretrained.return_value
        assert qa.model == mock_model.from_pretrained.return_value

def test_init_error():
    """Test error handling during initialization"""
    with patch('proteinflex.models.analysis.qa_system.AutoTokenizer') as mock_tokenizer:
        mock_tokenizer.from_pretrained.side_effect = Exception("Model loading error")
        with pytest.raises(Exception):
            ProteinQASystem()

def test_answer_question_success(qa_system):
    """Test successful question answering"""
    context = "The protein sequence contains hydrophobic residues."
    question = "What does the sequence contain?"

    result = qa_system.answer_question(context, question)

    assert isinstance(result, dict)
    assert 'answer' in result
    assert 'confidence' in result
    assert isinstance(result['answer'], str)
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1

def test_answer_question_long_input(qa_system):
    """Test handling of long input sequences"""
    context = "A" * 1000  # Long context
    question = "What is this?"

    result = qa_system.answer_question(context, question)

    assert isinstance(result, dict)
    assert 'answer' in result
    assert 'confidence' in result

def test_answer_question_empty_input(qa_system):
    """Test handling of empty inputs"""
    # Test empty context
    result = qa_system.answer_question("", "What is this?")
    assert result['answer'] == "Unable to process question"
    assert result['confidence'] == 0.0

    # Test empty question
    result = qa_system.answer_question("Some context", "")
    assert result['answer'] == "Unable to process question"
    assert result['confidence'] == 0.0

def test_answer_question_invalid_input(qa_system):
    """Test handling of invalid inputs"""
    # Test None inputs
    result = qa_system.answer_question(None, "What is this?")
    assert result['answer'] == "Unable to process question"
    assert result['confidence'] == 0.0

    result = qa_system.answer_question("Some context", None)
    assert result['answer'] == "Unable to process question"
    assert result['confidence'] == 0.0

def test_confidence_calculation(qa_system):
    """Test confidence score calculation"""
    context = "The protein has high stability."
    question = "What property does the protein have?"

    result = qa_system.answer_question(context, question)
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1

@pytest.mark.parametrize("context,question,expected_error", [
    (None, "Question?", True),
    ("Context", None, True),
    ("", "", True),
    ("Valid context", "Valid question", False)
])
def test_error_handling(qa_system, context, question, expected_error):
    """Test error handling with various input combinations"""
    result = qa_system.answer_question(context, question)

    if expected_error:
        assert result['answer'] == "Unable to process question"
        assert result['confidence'] == 0.0
    else:
        assert result['answer'] != "Unable to process question"
        assert result['confidence'] > 0.0

def test_model_evaluation_mode(qa_system):
    """Test that model is in evaluation mode"""
    assert not qa_system.model.training

def test_tokenizer_max_length(qa_system):
    """Test tokenizer max length handling"""
    context = "A" * 1000
    question = "B" * 100

    result = qa_system.answer_question(context, question)
    assert isinstance(result, dict)
    assert 'answer' in result
    # Should not raise any errors due to length

def test_answer_span_extraction(qa_system):
    """Test answer span extraction from logits"""
    context = "The protein sequence contains hydrophobic residues."
    question = "What does the sequence contain?"

    result = qa_system.answer_question(context, question)

    # Verify that we get a valid answer for valid input
    assert isinstance(result['answer'], str)
    assert result['answer'] == "predicted answer"
    assert result['confidence'] > 0.0
