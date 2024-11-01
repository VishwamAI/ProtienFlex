import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from models.analysis.nlp_analysis import ProteinNLPAnalyzer

class TestProteinNLPAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.nlp_analyzer = ProteinNLPAnalyzer()

        # Test inputs
        self.test_sequence = "MKLLVLGLCALIISASCKS"
        self.test_description = "A protein complex containing calcium binding sites and beta-helix core regions"

        # Mock transformer model and tokenizer
        self.mock_model = MagicMock(spec=PreTrainedModel)
        self.mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)

        # Mock model outputs
        self.mock_embeddings = torch.randn(1, 768)
        self.mock_logits = torch.randn(1, 10)  # 10 possible domain classes

        # Mock tokenizer output
        self.mock_tokens = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.ones(1, 5)
        }
        self.mock_tokenizer.return_value = self.mock_tokens

    @patch('models.analysis.nlp_analysis.AutoModel')
    @patch('models.analysis.nlp_analysis.AutoTokenizer')
    def test_analyze_sequence_description(self, mock_tokenizer_class, mock_model_class):
        """Test sequence description analysis."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        self.mock_model.return_value = {'last_hidden_state': self.mock_embeddings}

        analysis = self.nlp_analyzer.analyze_sequence_description(self.test_description)

        self.assertIsInstance(analysis, dict)
        self.assertIn('domain_predictions', analysis)
        self.assertIn('confidence_scores', analysis)
        self.assertIn('key_features', analysis)

    @patch('models.analysis.nlp_analysis.AutoModelForSequenceClassification')
    @patch('models.analysis.nlp_analysis.AutoTokenizer')
    def test_predict_protein_function(self, mock_tokenizer_class, mock_model_class):
        """Test protein function prediction from sequence."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        self.mock_model.return_value = {'logits': self.mock_logits}

        predictions = self.nlp_analyzer.predict_protein_function(self.test_sequence)

        self.assertIsInstance(predictions, dict)
        self.assertIn('predicted_functions', predictions)
        self.assertIn('confidence_scores', predictions)

    def test_extract_key_features(self):
        """Test key feature extraction from text description."""
        features = self.nlp_analyzer.extract_key_features(self.test_description)

        self.assertIsInstance(features, dict)
        self.assertIn('structural_features', features)
        self.assertIn('binding_sites', features)
        self.assertIn('domains', features)

    @patch('models.analysis.nlp_analysis.AutoModel')
    @patch('models.analysis.nlp_analysis.AutoTokenizer')
    def test_compare_sequences(self, mock_tokenizer_class, mock_model_class):
        """Test sequence comparison functionality."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        self.mock_model.return_value = {'last_hidden_state': self.mock_embeddings}

        sequence1 = self.test_sequence
        sequence2 = "MKLLVLGLCALIISASCKT"  # One residue different

        comparison = self.nlp_analyzer.compare_sequences(sequence1, sequence2)

        self.assertIsInstance(comparison, dict)
        self.assertIn('similarity_score', comparison)
        self.assertIn('differences', comparison)
        self.assertIn('alignment', comparison)

    def test_validate_description(self):
        """Test description validation."""
        valid_description = "A protein complex with alpha helices"
        invalid_description = ""

        # Test valid description
        self.assertTrue(self.nlp_analyzer.validate_description(valid_description))

        # Test invalid description
        with self.assertRaises(ValueError):
            self.nlp_analyzer.validate_description(invalid_description)

    @patch('models.analysis.nlp_analysis.AutoModelForTokenClassification')
    @patch('models.analysis.nlp_analysis.AutoTokenizer')
    def test_extract_mutations(self, mock_tokenizer_class, mock_model_class):
        """Test mutation extraction from text."""
        # Configure mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model

        test_text = "The mutation A123T affects protein stability"

        mutations = self.nlp_analyzer.extract_mutations(test_text)

        self.assertIsInstance(mutations, list)
        self.assertTrue(all(isinstance(m, dict) for m in mutations))
        for mutation in mutations:
            self.assertIn('original', mutation)
            self.assertIn('position', mutation)
            self.assertIn('mutated', mutation)

    def test_batch_processing(self):
        """Test batch processing of multiple sequences/descriptions."""
        descriptions = [
            "A protein with alpha helices",
            "A protein with beta sheets"
        ]

        results = self.nlp_analyzer.batch_process_descriptions(descriptions)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(descriptions))
        for result in results:
            self.assertIn('analysis', result)
            self.assertIn('processing_time', result)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        with self.assertRaises(ValueError):
            self.nlp_analyzer.analyze_sequence_description("")

        with self.assertRaises(ValueError):
            self.nlp_analyzer.predict_protein_function("")

        with self.assertRaises(ValueError):
            self.nlp_analyzer.compare_sequences("", self.test_sequence)

if __name__ == '__main__':
    unittest.main()
