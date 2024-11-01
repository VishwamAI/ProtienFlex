import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from models.generative.protein_llm import ProteinLanguageModel

class TestProteinLanguageModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.protein_llm = ProteinLanguageModel()

        # Test inputs
        self.test_sequence = "MKLLVLGLCALIISASCKS"
        self.test_prompt = "Generate a protein sequence with calcium binding sites"

        # Mock transformer model and tokenizer
        self.mock_model = MagicMock(spec=PreTrainedModel)
        self.mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)

        # Mock model outputs
        self.mock_logits = torch.randn(1, 20, 25)  # batch_size=1, seq_len=20, vocab_size=25
        self.mock_attention = torch.randn(1, 12, 20, 20)  # batch_size=1, num_heads=12, seq_len=20, seq_len=20

        # Mock tokenizer outputs
        self.mock_tokens = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.ones(1, 5)
        }
        self.mock_tokenizer.return_value = self.mock_tokens

    @patch('models.generative.protein_llm.AutoModelForCausalLM')
    @patch('models.generative.protein_llm.AutoTokenizer')
    def test_load_model(self, mock_tokenizer_class, mock_model_class):
        """Test model loading functionality."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model

        model, tokenizer = self.protein_llm.load_model()

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch('models.generative.protein_llm.AutoModelForCausalLM')
    @patch('models.generative.protein_llm.AutoTokenizer')
    def test_generate_sequence(self, mock_tokenizer_class, mock_model_class):
        """Test sequence generation."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        sequence = self.protein_llm.generate_sequence(
            self.test_prompt,
            max_length=100,
            temperature=0.7
        )

        self.assertIsInstance(sequence, str)
        self.mock_model.generate.assert_called_once()

    def test_validate_sequence(self):
        """Test sequence validation."""
        # Valid sequence
        valid_result = self.protein_llm.validate_sequence(self.test_sequence)
        self.assertTrue(valid_result['valid'])

        # Invalid sequence
        invalid_sequence = "MKLLVLGLCALIISASCKS123"
        invalid_result = self.protein_llm.validate_sequence(invalid_sequence)
        self.assertFalse(invalid_result['valid'])

    @patch('models.generative.protein_llm.AutoModelForCausalLM')
    @patch('models.generative.protein_llm.AutoTokenizer')
    def test_get_attention_maps(self, mock_tokenizer_class, mock_model_class):
        """Test attention map generation."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        self.mock_model.return_value = {'attentions': self.mock_attention}

        attention_maps = self.protein_llm.get_attention_maps(self.test_sequence)

        self.assertIsInstance(attention_maps, dict)
        self.assertIn('attention_weights', attention_maps)
        self.assertIn('head_importance', attention_maps)

    def test_process_attention(self):
        """Test attention processing."""
        attention_weights = torch.randn(1, 12, 20, 20)  # batch_size=1, num_heads=12, seq_len=20, seq_len=20

        processed = self.protein_llm.process_attention(
            attention_weights,
            self.test_sequence
        )

        self.assertIsInstance(processed, dict)
        self.assertIn('averaged_attention', processed)
        self.assertIn('important_positions', processed)

    @patch('models.generative.protein_llm.AutoModelForCausalLM')
    @patch('models.generative.protein_llm.AutoTokenizer')
    def test_batch_generation(self, mock_tokenizer_class, mock_model_class):
        """Test batch sequence generation."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        prompts = [
            "Generate a protein sequence with calcium binding sites",
            "Generate a protein sequence with beta sheets"
        ]

        sequences = self.protein_llm.batch_generate_sequences(
            prompts,
            max_length=100,
            temperature=0.7
        )

        self.assertIsInstance(sequences, list)
        self.assertEqual(len(sequences), len(prompts))
        self.mock_model.generate.assert_called_once()

    def test_sequence_metrics(self):
        """Test sequence metric calculations."""
        metrics = self.protein_llm.calculate_sequence_metrics(self.test_sequence)

        self.assertIsInstance(metrics, dict)
        self.assertIn('length', metrics)
        self.assertIn('amino_acid_composition', metrics)
        self.assertIn('molecular_weight', metrics)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        with self.assertRaises(ValueError):
            self.protein_llm.generate_sequence("")

        with self.assertRaises(ValueError):
            self.protein_llm.validate_sequence("")

        with self.assertRaises(ValueError):
            self.protein_llm.get_attention_maps("")

    @patch('models.generative.protein_llm.AutoModelForCausalLM')
    @patch('models.generative.protein_llm.AutoTokenizer')
    def test_model_configuration(self, mock_tokenizer_class, mock_model_class):
        """Test model configuration settings."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model

        config = self.protein_llm.get_model_configuration()

        self.assertIsInstance(config, dict)
        self.assertIn('model_type', config)
        self.assertIn('vocab_size', config)
        self.assertIn('hidden_size', config)

if __name__ == '__main__':
    unittest.main()
