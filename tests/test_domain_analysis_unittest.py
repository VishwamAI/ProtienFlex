import unittest
from unittest.mock import patch, MagicMock, Mock
import torch
import torch.nn as nn
import logging
from models.domain_analysis import DomainAnalyzer

logging.basicConfig(level=logging.INFO)

class TestDomainAnalyzer(unittest.TestCase):
    """Test cases for DomainAnalyzer class."""

    @patch('esm.pretrained.esm2_t33_650M_UR50D')
    def setUp(self, mock_esm_factory):
        """Set up test fixtures before each test method."""
        try:
            # Create a proper mock class for ESM model
            class MockESMModel(MagicMock):
                def __call__(self, tokens, repr_layers=None, *args, **kwargs):
                    print(f"MockESMModel called with tokens shape: {tokens.shape}")

                    batch_size, seq_length = tokens.shape
                    embedding_dim = 1280

                    embeddings = torch.ones((batch_size, seq_length, embedding_dim), dtype=torch.float32)
                    embeddings = embeddings + torch.randn_like(embeddings) * 0.1
                    embeddings = embeddings.clone().detach().requires_grad_(True)

                    num_heads = 12
                    attentions = torch.ones((batch_size, num_heads, seq_length, seq_length))
                    attentions = attentions + torch.randn_like(attentions) * 0.1
                    attentions = torch.nn.functional.softmax(attentions, dim=-1)

                    vocab_size = 33
                    logits = torch.ones((batch_size, seq_length, vocab_size))
                    logits = logits + torch.randn_like(logits) * 0.1

                    print(f"Created embeddings with shape: {embeddings.shape}")
                    return {
                        "representations": {33: embeddings},
                        "attentions": [attentions],
                        "logits": logits
                    }

                def eval(self):
                    return self

            # Create mock objects
            self.mock_model = MockESMModel()
            self.mock_alphabet = MagicMock()

            # Configure batch_converter
            def mock_batch_converter(*args):
                if len(args) > 0 and isinstance(args[0], list):
                    sequences = args[0]
                else:
                    sequences = [("0", "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG")]

                batch_size = len(sequences)
                max_seq_len = max(len(seq[1]) for seq in sequences)
                seq_len = max(max_seq_len, 32)

                batch_tokens = torch.ones((batch_size, seq_len), dtype=torch.long)
                batch_labels = [seq[0] for seq in sequences]
                batch_strs = [seq[1] for seq in sequences]

                print(f"mock_batch_converter created tokens with shape: {batch_tokens.shape}")
                return batch_labels, batch_strs, batch_tokens

            self.mock_alphabet.batch_converter = mock_batch_converter

            # Set up mock factory to return our mock objects
            mock_esm_factory.return_value = (self.mock_model, self.mock_alphabet)

            # Create analyzer with mock objects
            self.analyzer = DomainAnalyzer()

        except Exception as e:
            print(f"Error in setUp: {str(e)}")
            raise

    def tearDown(self):
        """Clean up after each test method."""
        patch.stopall()

    def test_identify_domains(self):
        """Test domain identification in protein sequences."""
        test_sequences = [
            "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
            "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK"
        ]

        for sequence in test_sequences:
            result = self.analyzer.identify_domains(sequence)

            # Verify the result structure
            self.assertIsInstance(result, dict)
            self.assertIn("domains", result)
            self.assertIsInstance(result["domains"], list)

            # Each domain should have required fields
            for domain in result["domains"]:
                self.assertIsInstance(domain, dict)
                self.assertIn("start", domain)
                self.assertIn("end", domain)
                self.assertIn("score", domain)
                self.assertIsInstance(domain["start"], int)
                self.assertIsInstance(domain["end"], int)
                self.assertIsInstance(domain["score"], float)

    def test_analyze_domain_interactions(self):
        """Test analysis of domain interactions."""
        test_cases = [
            ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK"),
            ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG")
        ]

        for sequence1, sequence2 in test_cases:
            result = self.analyzer.analyze_domain_interactions(sequence1, sequence2)

            # Verify the result structure
            self.assertIsInstance(result, dict)
            self.assertIn("interaction_score", result)
            self.assertIn("contact_probability", result)
            self.assertIn("binding_energy", result)
            self.assertIn("sequence1_length", result)
            self.assertIn("sequence2_length", result)

            # Verify value types and ranges
            self.assertIsInstance(result["interaction_score"], float)
            self.assertIsInstance(result["contact_probability"], float)
            self.assertIsInstance(result["binding_energy"], float)
            self.assertGreaterEqual(result["interaction_score"], -1.0)
            self.assertLessEqual(result["interaction_score"], 1.0)
            self.assertGreaterEqual(result["contact_probability"], 0.0)
            self.assertLessEqual(result["contact_probability"], 1.0)

    def test_predict_domain_function(self):
        """Test domain function prediction."""
        test_cases = [
            ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", "binding"),
            ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", "catalytic")
        ]

        for sequence, domain_type in test_cases:
            result = self.analyzer.predict_domain_function(sequence, domain_type)

            # Verify the result structure
            self.assertIsInstance(result, dict)
            self.assertIn("predictions", result)
            self.assertIsInstance(result["predictions"], list)

            # Each prediction should have required fields
            for prediction in result["predictions"]:
                self.assertIsInstance(prediction, dict)
                self.assertIn("function", prediction)
                self.assertIn("confidence", prediction)
                self.assertIsInstance(prediction["confidence"], float)

    def test_calculate_domain_stability(self):
        """Test domain stability calculation."""
        test_sequences = [
            "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG",
            "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK"
        ]

        for sequence in test_sequences:
            result = self.analyzer.calculate_domain_stability(sequence)

            # Verify the result structure
            self.assertIsInstance(result, dict)
            self.assertIn("stability_scores", result)
            self.assertIsInstance(result["stability_scores"], list)

            # Each stability score should have required fields
            for score in result["stability_scores"]:
                self.assertIsInstance(score, dict)
                self.assertIn("start", score)
                self.assertIn("end", score)
                self.assertIn("score", score)
                self.assertIsInstance(score["start"], int)
                self.assertIsInstance(score["end"], int)
                self.assertIsInstance(score["score"], float)



    def test_error_handling_none(self):
        """Test error handling for None input."""
        with self.assertRaises(ValueError):
            self.analyzer.calculate_domain_stability(None)

    def test_error_handling_empty(self):
        """Test error handling for empty string."""
        with self.assertRaises(ValueError):
            self.analyzer.calculate_domain_stability("")

    def test_error_handling_invalid_chars(self):
        """Test error handling for invalid characters."""
        with self.assertRaises(ValueError):
            self.analyzer.calculate_domain_stability("INVALID123")

    def test_error_handling_invalid_amino_acids(self):
        """Test error handling for invalid amino acids."""
        with self.assertRaises(ValueError):
            # Use a sequence containing truly invalid amino acids (B, J, O, U, X, Z)
            self.analyzer.calculate_domain_stability("ABJOXUZ")

    def test_error_handling_mixed_case(self):
        """Test error handling for mixed case."""
        with self.assertRaises(ValueError):
            self.analyzer.calculate_domain_stability("AcDeFG")

    def test_error_handling_special_chars(self):
        """Test error handling for special characters."""
        with self.assertRaises(ValueError):
            self.analyzer.calculate_domain_stability("ACE*FG")

    def test_scan_domain_boundaries(self):
        """Test scanning for domain boundaries."""
        test_cases = [
            ("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG", 5),
            ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAK", 7)
        ]

        for sequence, window_size in test_cases:
            with self.subTest(sequence=sequence, window_size=window_size):
                boundaries = self.analyzer.scan_domain_boundaries(sequence, window_size)

                self.assertIsInstance(boundaries, list)
                for boundary in boundaries:
                    self.assertIsInstance(boundary, dict)
                    self.assertIn("start", boundary)
                    self.assertIn("end", boundary)
                    self.assertIn("score", boundary)
                    self.assertIn("type", boundary)
                    self.assertGreaterEqual(boundary["score"], 0)
                    self.assertLessEqual(boundary["score"], 1)
                    self.assertGreaterEqual(boundary["start"], 0)
                    self.assertLess(boundary["end"], len(sequence))

if __name__ == '__main__':
    unittest.main()
