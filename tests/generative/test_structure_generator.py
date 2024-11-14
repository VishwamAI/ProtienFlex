"""
Tests for the structure-aware generator implementation
"""
import torch
import unittest
from models.generative.structure_generator import StructureAwareGenerator

class TestStructureAwareGenerator(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_length = 10
        self.hidden_size = 768
        self.generator = StructureAwareGenerator(
            hidden_size=self.hidden_size,
            num_attention_heads=8,
            num_layers=6,
            dropout_prob=0.1
        )

    def test_initialization(self):
        """Test proper initialization of generator components"""
        self.assertEqual(self.generator.hidden_size, 768)
        self.assertEqual(self.generator.num_attention_heads, 8)
        self.assertEqual(self.generator.num_layers, 6)
        self.assertEqual(len(self.generator.attention_layers), 6)
        self.assertEqual(len(self.generator.ff_layers), 6)

    def test_forward_pass(self):
        """Test forward pass with basic inputs"""
        input_ids = torch.randint(0, 22, (self.batch_size, self.seq_length))
        outputs = self.generator(input_ids)

        # Check output dictionary keys
        self.assertIn("logits", outputs)
        self.assertIn("hidden_states", outputs)
        self.assertIn("attention_weights", outputs)

        # Check shapes
        self.assertEqual(
            outputs["logits"].shape,
            (self.batch_size, self.seq_length, 22)
        )
        self.assertEqual(
            outputs["hidden_states"].shape,
            (self.batch_size, self.seq_length, self.hidden_size)
        )
        self.assertEqual(len(outputs["attention_weights"]), 6)

    def test_structure_aware_generation(self):
        """Test forward pass with structural information"""
        input_ids = torch.randint(0, 22, (self.batch_size, self.seq_length))
        distance_matrix = torch.randn(
            self.batch_size, self.seq_length, self.seq_length
        )
        angle_matrix = torch.randn(
            self.batch_size, self.seq_length, self.seq_length
        )

        outputs = self.generator(
            input_ids,
            distance_matrix=distance_matrix,
            angle_matrix=angle_matrix
        )

        # Check that attention weights reflect structural information
        attention_weights = outputs["attention_weights"]
        self.assertEqual(len(attention_weights), 6)
        for layer_weights in attention_weights:
            self.assertEqual(
                layer_weights.shape,
                (self.batch_size, 8, self.seq_length, self.seq_length)
            )

    def test_sequence_generation(self):
        """Test protein sequence generation"""
        start_tokens = torch.randint(0, 22, (self.batch_size, 2))
        max_length = 10

        generated = self.generator.generate(
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=0.8
        )

        # Check generated sequence properties
        self.assertEqual(
            generated.shape,
            (self.batch_size, max_length)
        )
        self.assertTrue(torch.all(generated >= 0))
        self.assertTrue(torch.all(generated < 22))

    def test_structure_guided_generation(self):
        """Test generation with structural guidance"""
        start_tokens = torch.randint(0, 22, (self.batch_size, 2))
        max_length = 10
        distance_matrix = torch.randn(
            self.batch_size, max_length, max_length
        )
        angle_matrix = torch.randn(
            self.batch_size, max_length, max_length
        )

        generated = self.generator.generate(
            start_tokens=start_tokens,
            max_length=max_length,
            temperature=0.8,
            distance_matrix=distance_matrix,
            angle_matrix=angle_matrix
        )

        # Check generated sequence properties
        self.assertEqual(
            generated.shape,
            (self.batch_size, max_length)
        )
        self.assertTrue(torch.all(generated >= 0))
        self.assertTrue(torch.all(generated < 22))

    def test_gradient_flow(self):
        """Test gradient flow through the generator"""
        input_ids = torch.randint(
            0, 22, (self.batch_size, self.seq_length),
            requires_grad=False
        )

        # Forward pass
        outputs = self.generator(input_ids)
        loss = outputs["logits"].sum()
        loss.backward()

        # Check gradients
        for param in self.generator.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.all(torch.isfinite(param.grad)))

if __name__ == '__main__':
    unittest.main()
