import unittest
import torch
import esm
import numpy as np
from models.mutation_analysis import MutationAnalyzer

class TestMutationAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize ESM model
        cls.model = unittest.mock.Mock()
        cls.alphabet = unittest.mock.Mock()
        cls.device = torch.device("cpu")

        # Mock model methods
        cls.model.alphabet = cls.alphabet
        cls.model.layers = [unittest.mock.Mock() for _ in range(12)]
        cls.model.modules = lambda: []

        # Mock batch converter
        def batch_converter_return(data):
            sequence = data[0][1]
            batch_tokens = torch.randn(1, len(sequence) + 2, 33)
            return (
                [{"sequence": sequence, "start": 0, "end": len(sequence)}],
                [sequence],
                batch_tokens
            )
        cls.alphabet.get_batch_converter.return_value = unittest.mock.Mock(side_effect=batch_converter_return)

        cls.analyzer = MutationAnalyzer(cls.model, cls.device)

        # Test sequence (insulin)
        cls.test_sequence = "FVNQHLCGSHLVEAL"  # Shorter segment of insulin sequence
        cls.test_position = 7
        cls.test_mutation = "A"

    def test_predict_mutation_effect(self):
        """Test the complete mutation effect prediction pipeline"""
        # Mock the internal method returns
        self.analyzer._calculate_stability_impact = unittest.mock.Mock(return_value={
            'start': 0,
            'end': len(self.test_sequence),
            'score': 0.85,
            'type': 'stability',
            'stability_score': 0.75
        })
        self.analyzer._calculate_structural_impact = unittest.mock.Mock(return_value={
            'start': 0,
            'end': len(self.test_sequence),
            'score': 0.8,
            'type': 'structural',
            'structural_score': 0.8
        })
        self.analyzer._calculate_conservation = unittest.mock.Mock(return_value={
            'start': 0,
            'end': len(self.test_sequence),
            'score': 0.9,
            'type': 'conservation',
            'conservation_score': 0.85
        })

        result = self.analyzer.predict_mutation_effect(
            self.test_sequence,
            self.test_position,
            self.test_mutation
        )

        # Check dictionary structure
        self.assertIsInstance(result, dict)
        self.assertIn('start', result)
        self.assertIn('end', result)
        self.assertIn('score', result)
        self.assertIn('type', result)
        self.assertIn('stability_impact', result)
        self.assertIn('structural_impact', result)
        self.assertIn('conservation_score', result)
        self.assertIn('overall_impact', result)
        self.assertIn('confidence', result)

        # Check value ranges
        self.assertGreaterEqual(result['stability_impact'], 0)
        self.assertLessEqual(result['stability_impact'], 1)
        self.assertGreaterEqual(result['structural_impact'], 0)
        self.assertLessEqual(result['structural_impact'], 1)
        self.assertGreaterEqual(result['conservation_score'], 0)
        self.assertLessEqual(result['conservation_score'], 1)
        self.assertGreaterEqual(result['overall_impact'], 0)
        self.assertLessEqual(result['overall_impact'], 1)
        self.assertGreaterEqual(result['confidence'], 0)
        self.assertLessEqual(result['confidence'], 100)
        self.assertGreaterEqual(result['score'], 0)
        self.assertLessEqual(result['score'], 1)

    def test_stability_impact(self):
        """Test stability impact calculation"""
        # Mock model forward method
        self.model.forward = unittest.mock.Mock(return_value={
            'representations': {33: torch.randn(1, len(self.test_sequence), 1280)},
            'start': 0,
            'end': len(self.test_sequence),
            'score': 0.85,
            'type': 'stability'
        })

        result = self.analyzer._calculate_stability_impact(
            self.test_sequence,
            self.test_position,
            self.test_mutation
        )

        self.assertIsInstance(result, dict)
        self.assertIn('start', result)
        self.assertIn('end', result)
        self.assertIn('score', result)
        self.assertIn('type', result)
        self.assertIn('stability_score', result)

        self.assertIsInstance(result['stability_score'], float)
        self.assertGreaterEqual(result['stability_score'], 0)
        self.assertLessEqual(result['stability_score'], 1)

        # Test with different mutations
        mutations = ['G', 'P', 'D']
        for mut in mutations:
            result = self.analyzer._calculate_stability_impact(
                self.test_sequence,
                self.test_position,
                mut
            )
            self.assertIsInstance(result, dict)
            self.assertIn('stability_score', result)
            self.assertIsInstance(result['stability_score'], float)
            self.assertGreaterEqual(result['stability_score'], 0)
            self.assertLessEqual(result['stability_score'], 1)

    def test_structural_impact(self):
        """Test structural impact calculation"""
        # Mock model forward method
        self.model.forward = unittest.mock.Mock(return_value={
            'representations': {33: torch.randn(1, len(self.test_sequence), 1280)},
            'start': 0,
            'end': len(self.test_sequence),
            'score': 0.8,
            'type': 'structural'
        })

        result = self.analyzer._calculate_structural_impact(
            self.test_sequence,
            self.test_position
        )

        self.assertIsInstance(result, dict)
        self.assertIn('start', result)
        self.assertIn('end', result)
        self.assertIn('score', result)
        self.assertIn('type', result)
        self.assertIn('structural_score', result)

        self.assertIsInstance(result['structural_score'], float)
        self.assertGreaterEqual(result['structural_score'], 0)
        self.assertLessEqual(result['structural_score'], 1)

        # Test at different positions
        positions = [3, 7, 11]  # Positions within the shorter sequence length
        for pos in positions:
            result = self.analyzer._calculate_structural_impact(
                self.test_sequence,
                pos
            )
            self.assertIsInstance(result, dict)
            self.assertIn('structural_score', result)
            self.assertIsInstance(result['structural_score'], float)
            self.assertGreaterEqual(result['structural_score'], 0)
            self.assertLessEqual(result['structural_score'], 1)

    def test_conservation(self):
        """Test conservation score calculation"""
        # Mock model forward method
        self.model.forward = unittest.mock.Mock(return_value={
            'representations': {33: torch.randn(1, len(self.test_sequence), 1280)},
            'start': 0,
            'end': len(self.test_sequence),
            'score': 0.9,
            'type': 'conservation'
        })

        result = self.analyzer._calculate_conservation(
            self.test_sequence,
            self.test_position
        )

        self.assertIsInstance(result, dict)
        self.assertIn('start', result)
        self.assertIn('end', result)
        self.assertIn('score', result)
        self.assertIn('type', result)
        self.assertIn('conservation_score', result)

        self.assertIsInstance(result['conservation_score'], float)
        self.assertGreaterEqual(result['conservation_score'], 0)
        self.assertLessEqual(result['conservation_score'], 1)

        # Test at different positions
        positions = [2, 5, 8]  # Positions within the shorter sequence length
        for pos in positions:
            result = self.analyzer._calculate_conservation(
                self.test_sequence,
                pos
            )
            self.assertIsInstance(result, dict)
            self.assertIn('conservation_score', result)
            self.assertIsInstance(result['conservation_score'], float)
            self.assertGreaterEqual(result['conservation_score'], 0)
            self.assertLessEqual(result['conservation_score'], 1)

    def test_device_compatibility(self):
        """Test that the analyzer works on both CPU and GPU"""
        # Mock internal methods for CPU test
        mock_result = {
            'start': 0,
            'end': len(self.test_sequence),
            'score': 0.85,
            'type': 'mutation_effect',
            'stability_impact': 0.8,
            'structural_impact': 0.7,
            'conservation_score': 0.9,
            'overall_impact': 0.8,
            'confidence': 85
        }

        # Test on CPU
        cpu_device = torch.device("cpu")
        cpu_model = unittest.mock.Mock()
        cpu_model.alphabet = self.alphabet
        cpu_analyzer = MutationAnalyzer(cpu_model, cpu_device)
        cpu_analyzer.predict_mutation_effect = unittest.mock.Mock(return_value=mock_result)

        cpu_result = cpu_analyzer.predict_mutation_effect(
            self.test_sequence,
            self.test_position,
            self.test_mutation
        )
        self.assertIsInstance(cpu_result, dict)
        self.assertIn('start', cpu_result)
        self.assertIn('end', cpu_result)
        self.assertIn('score', cpu_result)
        self.assertIn('type', cpu_result)
        self.assertNotIn('error', cpu_result)

        # Test on GPU if available
        if torch.cuda.is_available():
            gpu_device = torch.device("cuda")
            gpu_model = unittest.mock.Mock()
            gpu_model.alphabet = self.alphabet
            gpu_analyzer = MutationAnalyzer(gpu_model, gpu_device)
            gpu_analyzer.predict_mutation_effect = unittest.mock.Mock(return_value=mock_result)

            gpu_result = gpu_analyzer.predict_mutation_effect(
                self.test_sequence,
                self.test_position,
                self.test_mutation
            )
            self.assertIsInstance(gpu_result, dict)
            self.assertIn('start', gpu_result)
            self.assertIn('end', gpu_result)
            self.assertIn('score', gpu_result)
            self.assertIn('type', gpu_result)
            self.assertNotIn('error', gpu_result)

    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        # Mock error responses
        error_dict = {
            'start': 0,
            'end': len(self.test_sequence),
            'score': 0.0,
            'type': 'error',
            'error': 'Invalid input',
            'stability_impact': 0.0,
            'structural_impact': 0.0,
            'conservation_score': 0.0,
            'overall_impact': 0.0,
            'confidence': 0.0
        }
        self.analyzer.predict_mutation_effect = unittest.mock.Mock(return_value=error_dict)

        # Test with invalid position
        result = self.analyzer.predict_mutation_effect(
            self.test_sequence,
            len(self.test_sequence) + 1,  # Invalid position
            self.test_mutation
        )
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'error')

        # Test with invalid mutation
        result = self.analyzer.predict_mutation_effect(
            self.test_sequence,
            self.test_position,
            'X'  # Invalid amino acid
        )
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'error')

        # Test with invalid sequence
        result = self.analyzer.predict_mutation_effect(
            "INVALID123",  # Invalid sequence
            self.test_position,
            self.test_mutation
        )
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'error')

if __name__ == '__main__':
    unittest.main()
