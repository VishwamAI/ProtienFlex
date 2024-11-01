import unittest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from models.utils.esm_utils import ESMWrapper

class TestESMWrapper(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.esm_wrapper = ESMWrapper()
        self.test_sequence = "MKLLVLGLCALIISASCKS"

        # Mock ESM model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()

        # Mock embeddings and attention
        self.mock_embeddings = torch.randn(1, len(self.test_sequence) + 2, 1280)  # +2 for special tokens
        self.mock_attention = torch.randn(1, 33, len(self.test_sequence) + 2, len(self.test_sequence) + 2)

        # Mock tokenizer output
        self.mock_tokens = {
            'input_ids': torch.tensor([[0] + [i for i in range(1, len(self.test_sequence) + 1)] + [2]]),
            'attention_mask': torch.ones(1, len(self.test_sequence) + 2)
        }
        self.mock_tokenizer.return_value = self.mock_tokens

    @patch('models.utils.esm_utils.esm')
    def test_load_model(self, mock_esm):
        """Test model loading functionality."""
        # Configure mock
        mock_esm.pretrained.return_value = (self.mock_model, self.mock_tokenizer)

        model, tokenizer = self.esm_wrapper.load_model()

        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        mock_esm.pretrained.assert_called_once()

    @patch('models.utils.esm_utils.esm')
    def test_get_sequence_embeddings(self, mock_esm):
        """Test sequence embedding generation."""
        # Configure mocks
        mock_esm.pretrained.return_value = (self.mock_model, self.mock_tokenizer)
        self.mock_model.return_value = {
            'representations': {33: self.mock_embeddings},
            'attentions': self.mock_attention
        }

        embeddings = self.esm_wrapper.get_sequence_embeddings(self.test_sequence)

        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[1], 1280)  # ESM-2 hidden dim
        self.assertEqual(embeddings.shape[0], len(self.test_sequence))

    @patch('models.utils.esm_utils.esm')
    def test_get_attention_maps(self, mock_esm):
        """Test attention map generation."""
        # Configure mocks
        mock_esm.pretrained.return_value = (self.mock_model, self.mock_tokenizer)
        self.mock_model.return_value = {
            'representations': {33: self.mock_embeddings},
            'attentions': self.mock_attention
        }

        attention_maps = self.esm_wrapper.get_attention_maps(self.test_sequence)

        self.assertIsInstance(attention_maps, torch.Tensor)
        self.assertEqual(attention_maps.shape[2], len(self.test_sequence))
        self.assertEqual(attention_maps.shape[3], len(self.test_sequence))

    def test_process_attention_maps(self):
        """Test attention map processing."""
        attention_maps = torch.randn(1, 33, len(self.test_sequence), len(self.test_sequence))

        processed_maps = self.esm_wrapper.process_attention_maps(attention_maps)

        self.assertIsInstance(processed_maps, torch.Tensor)
        self.assertEqual(processed_maps.shape[0], len(self.test_sequence))
        self.assertEqual(processed_maps.shape[1], len(self.test_sequence))

    @patch('models.utils.esm_utils.esm')
    def test_batch_processing(self, mock_esm):
        """Test batch processing of sequences."""
        # Configure mocks
        mock_esm.pretrained.return_value = (self.mock_model, self.mock_tokenizer)
        self.mock_model.return_value = {
            'representations': {33: self.mock_embeddings.repeat(2, 1, 1)},
            'attentions': self.mock_attention.repeat(2, 1, 1, 1)
        }

        sequences = [self.test_sequence, self.test_sequence]
        embeddings = self.esm_wrapper.batch_process_sequences(sequences)

        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), len(sequences))
        self.assertEqual(embeddings[0].shape[1], 1280)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        with self.assertRaises(ValueError):
            self.esm_wrapper.get_sequence_embeddings("")

        with self.assertRaises(ValueError):
            self.esm_wrapper.get_attention_maps(None)

        with self.assertRaises(ValueError):
            self.esm_wrapper.batch_process_sequences([])

    @patch('models.utils.esm_utils.esm')
    def test_device_handling(self, mock_esm):
        """Test device handling (CPU/GPU)."""
        # Configure mocks
        mock_esm.pretrained.return_value = (self.mock_model, self.mock_tokenizer)
        self.mock_model.return_value = {
            'representations': {33: self.mock_embeddings},
            'attentions': self.mock_attention
        }

        # Test with CPU
        embeddings_cpu = self.esm_wrapper.get_sequence_embeddings(
            self.test_sequence,
            device='cpu'
        )
        self.assertEqual(embeddings_cpu.device.type, 'cpu')

        # Test with CUDA if available
        if torch.cuda.is_available():
            embeddings_gpu = self.esm_wrapper.get_sequence_embeddings(
                self.test_sequence,
                device='cuda'
            )
            self.assertEqual(embeddings_gpu.device.type, 'cuda')

if __name__ == '__main__':
    unittest.main()
