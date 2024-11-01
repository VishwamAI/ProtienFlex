import unittest
from unittest.mock import patch, MagicMock
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from models.analysis.qa_system import ProteinQASystem

class TestProteinQASystem(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.qa_system = ProteinQASystem()

        # Test inputs
        self.test_sequence = "MKLLVLGLCALIISASCKS"
        self.test_question = "What are the binding sites in this protein?"
        self.test_context = f"The protein sequence {self.test_sequence} contains calcium binding sites at positions 10-15."

        # Mock transformer model and tokenizer
        self.mock_model = MagicMock(spec=PreTrainedModel)
        self.mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)

        # Mock model outputs
        self.mock_output = {
            'start_logits': torch.randn(1, 100),
            'end_logits': torch.randn(1, 100)
        }

        # Mock tokenizer outputs
        self.mock_tokens = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.ones(1, 5),
            'token_type_ids': torch.zeros(1, 5)
        }
        self.mock_tokenizer.return_value = self.mock_tokens

    @patch('models.analysis.qa_system.AutoModelForQuestionAnswering')
    @patch('models.analysis.qa_system.AutoTokenizer')
    def test_initialize_model(self, mock_tokenizer_class, mock_model_class):
        """Test model initialization."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model

        model, tokenizer = self.qa_system.initialize_model()

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch('models.analysis.qa_system.AutoModelForQuestionAnswering')
    @patch('models.analysis.qa_system.AutoTokenizer')
    def test_answer_question(self, mock_tokenizer_class, mock_model_class):
        """Test question answering functionality."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        self.mock_model.return_value = self.mock_output

        answer = self.qa_system.answer_question(
            self.test_question,
            self.test_context
        )

        self.assertIsInstance(answer, dict)
        self.assertIn('answer', answer)
        self.assertIn('confidence', answer)
        self.assertIn('context_used', answer)

    def test_preprocess_question(self):
        """Test question preprocessing."""
        processed = self.qa_system.preprocess_question(self.test_question)

        self.assertIsInstance(processed, str)
        self.assertTrue(len(processed) > 0)
        self.assertNotEqual(processed, "")

    def test_validate_inputs(self):
        """Test input validation."""
        # Valid inputs
        valid = self.qa_system.validate_inputs(
            self.test_question,
            self.test_context
        )
        self.assertTrue(valid)

        # Invalid inputs
        with self.assertRaises(ValueError):
            self.qa_system.validate_inputs("", self.test_context)
        with self.assertRaises(ValueError):
            self.qa_system.validate_inputs(self.test_question, "")

    @patch('models.analysis.qa_system.AutoModelForQuestionAnswering')
    @patch('models.analysis.qa_system.AutoTokenizer')
    def test_batch_qa(self, mock_tokenizer_class, mock_model_class):
        """Test batch question answering."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        self.mock_model.return_value = self.mock_output

        questions = [
            "What are the binding sites?",
            "What is the secondary structure?"
        ]

        answers = self.qa_system.batch_answer_questions(
            questions,
            self.test_context
        )

        self.assertIsInstance(answers, list)
        self.assertEqual(len(answers), len(questions))
        for answer in answers:
            self.assertIn('answer', answer)
            self.assertIn('confidence', answer)

    def test_context_preparation(self):
        """Test context preparation."""
        prepared_context = self.qa_system.prepare_context(
            self.test_sequence,
            include_structure=True
        )

        self.assertIsInstance(prepared_context, str)
        self.assertIn(self.test_sequence, prepared_context)

    def test_answer_scoring(self):
        """Test answer confidence scoring."""
        mock_start_logits = torch.tensor([[1.0, 2.0, 3.0]])
        mock_end_logits = torch.tensor([[0.5, 1.5, 2.5]])

        score = self.qa_system.calculate_answer_confidence(
            mock_start_logits,
            mock_end_logits
        )

        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)

    def test_answer_extraction(self):
        """Test answer extraction from context."""
        start_pos = 10
        end_pos = 15
        extracted = self.qa_system.extract_answer(
            self.test_context,
            start_pos,
            end_pos
        )

        self.assertIsInstance(extracted, str)
        self.assertTrue(len(extracted) > 0)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        with self.assertRaises(ValueError):
            self.qa_system.answer_question("", "")

        with self.assertRaises(ValueError):
            self.qa_system.batch_answer_questions([], "")

        with self.assertRaises(ValueError):
            self.qa_system.prepare_context("")

    @patch('models.analysis.qa_system.AutoModelForQuestionAnswering')
    @patch('models.analysis.qa_system.AutoTokenizer')
    def test_model_configuration(self, mock_tokenizer_class, mock_model_class):
        """Test model configuration settings."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model

        config = self.qa_system.get_model_configuration()

        self.assertIsInstance(config, dict)
        self.assertIn('model_type', config)
        self.assertIn('max_length', config)
        self.assertIn('device', config)

if __name__ == '__main__':
    unittest.main()
