import unittest
import torch
import esm
import numpy as np
from models.mutation_analysis import MutationAnalyzer

class TestMutationAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize ESM model
        cls.model, cls.alphabet = esm.pretrained.esm2_t6_8M_UR50D()  # Using smaller model
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = cls.model.to(cls.device)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        cls.analyzer = MutationAnalyzer(cls.model, cls.device)

        # Test sequence (insulin)
        cls.test_sequence = "FVNQHLCGSHLVEAL"  # Shorter segment of insulin sequence
        cls.test_position = 7
        cls.test_mutation = "A"

    def test_predict_mutation_effect(self):
        """Test the complete mutation effect prediction pipeline"""
        result = self.analyzer.predict_mutation_effect(
            self.test_sequence,
            self.test_position,
            self.test_mutation
        )

        # Check result structure
        self.assertIsInstance(result, dict)
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

    def test_stability_impact(self):
        """Test stability impact calculation"""
        stability_score = self.analyzer._calculate_stability_impact(
            self.test_sequence,
            self.test_position,
            self.test_mutation
        )

        self.assertIsInstance(stability_score, float)
        self.assertGreaterEqual(stability_score, 0)
        self.assertLessEqual(stability_score, 1)

        # Test with different mutations
        mutations = ['G', 'P', 'D']
        for mut in mutations:
            score = self.analyzer._calculate_stability_impact(
                self.test_sequence,
                self.test_position,
                mut
            )
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_structural_impact(self):
        """Test structural impact calculation"""
        structural_score = self.analyzer._calculate_structural_impact(
            self.test_sequence,
            self.test_position
        )

        self.assertIsInstance(structural_score, float)
        self.assertGreaterEqual(structural_score, 0)
        self.assertLessEqual(structural_score, 1)

        # Test at different positions
        positions = [3, 7, 11]  # Positions within the shorter sequence length
        for pos in positions:
            score = self.analyzer._calculate_structural_impact(
                self.test_sequence,
                pos
            )
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_conservation(self):
        """Test conservation score calculation"""
        conservation_score = self.analyzer._calculate_conservation(
            self.test_sequence,
            self.test_position
        )

        self.assertIsInstance(conservation_score, float)
        self.assertGreaterEqual(conservation_score, 0)
        self.assertLessEqual(conservation_score, 1)

        # Test at different positions
        positions = [2, 5, 8]  # Positions within the shorter sequence length
        for pos in positions:
            score = self.analyzer._calculate_conservation(
                self.test_sequence,
                pos
            )
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_device_compatibility(self):
        """Test that the analyzer works on both CPU and GPU"""
        # Test on CPU
        cpu_device = torch.device("cpu")
        cpu_model = self.model.to(cpu_device)
        cpu_analyzer = MutationAnalyzer(cpu_model, cpu_device)
        cpu_result = cpu_analyzer.predict_mutation_effect(
            self.test_sequence,
            self.test_position,
            self.test_mutation
        )
        self.assertIsInstance(cpu_result, dict)
        self.assertNotIn('error', cpu_result)

        # Test on GPU if available
        if torch.cuda.is_available():
            gpu_device = torch.device("cuda")
            gpu_model = self.model.to(gpu_device)
            gpu_analyzer = MutationAnalyzer(gpu_model, gpu_device)
            gpu_result = gpu_analyzer.predict_mutation_effect(
                self.test_sequence,
                self.test_position,
                self.test_mutation
            )
            self.assertIsInstance(gpu_result, dict)
            self.assertNotIn('error', gpu_result)

    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        # Test with invalid position
        result = self.analyzer.predict_mutation_effect(
            self.test_sequence,
            len(self.test_sequence) + 1,  # Invalid position
            self.test_mutation
        )
        self.assertIn('error', result)

        # Test with invalid mutation
        result = self.analyzer.predict_mutation_effect(
            self.test_sequence,
            self.test_position,
            'X'  # Invalid amino acid
        )
        self.assertIn('error', result)

        # Test with invalid sequence
        result = self.analyzer.predict_mutation_effect(
            "INVALID123",  # Invalid sequence
            self.test_position,
            self.test_mutation
        )
        self.assertIn('error', result)

if __name__ == '__main__':
    unittest.main()
