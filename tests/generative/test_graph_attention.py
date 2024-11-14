"""
Tests for the graph attention layer implementation
"""
import torch
import unittest
from models.generative.graph_attention import GraphAttentionLayer

class TestGraphAttentionLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_length = 10
        self.hidden_size = 768
        self.layer = GraphAttentionLayer(
            hidden_size=self.hidden_size,
            num_attention_heads=8,
            dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )

    def test_initialization(self):
        """Test proper initialization of layer components"""
        self.assertEqual(self.layer.hidden_size, 768)
        self.assertEqual(self.layer.num_attention_heads, 8)
        self.assertEqual(self.layer.attention_head_size, 96)
        self.assertEqual(self.layer.all_head_size, 768)

    def test_forward_pass(self):
        """Test forward pass with only hidden states"""
        hidden_states = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size
        )
        output, attention_probs = self.layer(hidden_states)

        # Check output shapes
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_length, self.hidden_size)
        )
        self.assertEqual(
            attention_probs.shape,
            (self.batch_size, 8, self.seq_length, self.seq_length)
        )

    def test_structure_aware_attention(self):
        """Test forward pass with structural information"""
        hidden_states = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size
        )
        distance_matrix = torch.randn(
            self.batch_size, self.seq_length, self.seq_length
        )
        angle_matrix = torch.randn(
            self.batch_size, self.seq_length, self.seq_length
        )

        output, attention_probs = self.layer(
            hidden_states,
            distance_matrix=distance_matrix,
            angle_matrix=angle_matrix
        )

        # Check output shapes
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_length, self.hidden_size)
        )
        self.assertEqual(
            attention_probs.shape,
            (self.batch_size, 8, self.seq_length, self.seq_length)
        )

    def test_attention_mask(self):
        """Test attention masking"""
        hidden_states = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size
        )
        attention_mask = torch.ones(
            self.batch_size, self.seq_length
        )
        attention_mask[:, 5:] = 0  # Mask out second half of sequence

        output, attention_probs = self.layer(
            hidden_states,
            attention_mask=attention_mask
        )

        # Check that masked positions have near-zero attention
        self.assertTrue(
            torch.all(attention_probs[:, :, :, 5:] < 1e-4)
        )

    def test_gradient_flow(self):
        """Test gradient flow through the layer"""
        hidden_states = torch.randn(
            self.batch_size, self.seq_length, self.hidden_size,
            requires_grad=True
        )

        output, _ = self.layer(hidden_states)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(torch.all(torch.isfinite(hidden_states.grad)))

if __name__ == '__main__':
    unittest.main()
