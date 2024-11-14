import unittest
import torch
import math
from models.generative.concept_bottleneck import ConceptBottleneckLayer, LoRALayer

class TestConceptBottleneckLayer(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 768
        self.num_concepts = 64
        self.batch_size = 4
        self.seq_length = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.concept_layer = ConceptBottleneckLayer(
            hidden_size=self.hidden_size,
            num_concepts=self.num_concepts
        ).to(self.device)

        self.hidden_states = torch.randn(
            self.batch_size,
            self.seq_length,
            self.hidden_size,
            device=self.device
        )

    def test_concept_bottleneck_forward(self):
        """Test forward pass of concept bottleneck layer"""
        # Test without returning concepts
        output, _ = self.concept_layer(self.hidden_states, return_concepts=False)
        self.assertEqual(output.shape, self.hidden_states.shape)

        # Test with returning concepts
        output, concepts = self.concept_layer(self.hidden_states, return_concepts=True)
        self.assertEqual(output.shape, self.hidden_states.shape)
        self.assertEqual(len(concepts), 4)  # structure, chemistry, function, interaction

        # Verify concept shapes
        for concept_type, concept_tensor in concepts.items():
            self.assertEqual(
                concept_tensor.shape,
                (self.batch_size, self.seq_length, self.num_concepts // 4)
            )

    def test_concept_interpretability(self):
        """Test interpretability of concept activations"""
        _, concepts = self.concept_layer(self.hidden_states, return_concepts=True)

        # Check concept activation ranges
        for concept_type, concept_tensor in concepts.items():
            # Concepts should be bounded between 0 and 1 after sigmoid
            self.assertTrue(torch.all(concept_tensor >= 0))
            self.assertTrue(torch.all(concept_tensor <= 1))

            # Check if concepts are well-distributed
            mean_activation = concept_tensor.mean().item()
            self.assertTrue(0.2 <= mean_activation <= 0.8)

    def test_gradient_flow(self):
        """Test gradient flow through concept bottleneck"""
        self.hidden_states.requires_grad_(True)
        output, concepts = self.concept_layer(self.hidden_states, return_concepts=True)

        # Compute loss using both output and concepts
        output_loss = output.mean()
        concept_loss = sum(c.mean() for c in concepts.values())
        total_loss = output_loss + concept_loss

        # Check gradient flow
        total_loss.backward()
        self.assertIsNotNone(self.hidden_states.grad)
        self.assertTrue(torch.all(self.hidden_states.grad != 0))

class TestLoRALayer(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 768
        self.lora_rank = 8
        self.batch_size = 4
        self.seq_length = 16

        # Initialize model components
        self.lora_layer = LoRALayer(
            hidden_size=self.hidden_size,
            lora_rank=self.lora_rank
        )

        # Move to available device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lora_layer = self.lora_layer.to(self.device)

        # Create test inputs
        self.hidden_states = torch.randn(
            self.batch_size,
            self.seq_length,
            self.hidden_size,
            device=self.device
        )

    def test_lora_forward(self):
        """Test forward pass of LoRA layer"""
        output = self.lora_layer(self.hidden_states)
        self.assertEqual(output.shape, self.hidden_states.shape)

        # Check if output differs from input
        self.assertTrue(torch.any(output != self.hidden_states))

        # Check if output magnitude is reasonable
        output_norm = torch.norm(output)
        input_norm = torch.norm(self.hidden_states)
        ratio = (output_norm / input_norm).item()  # Convert to Python scalar
        self.assertTrue(0.1 <= ratio <= 10, f"Output/input ratio {ratio} is outside reasonable bounds [0.1, 10]")

    def test_parameter_efficiency(self):
        """Test parameter efficiency of LoRA layer"""
        total_params = sum(p.numel() for p in self.lora_layer.parameters())
        full_layer_params = self.hidden_size * self.hidden_size

        # LoRA should use significantly fewer parameters
        self.assertTrue(total_params < full_layer_params * 0.1)

    def test_gradient_flow(self):
        """Test gradient flow through LoRA layer"""
        self.hidden_states.requires_grad_(True)
        output = self.lora_layer(self.hidden_states)
        loss = output.mean()

        # Check gradient flow
        loss.backward()
        self.assertIsNotNone(self.hidden_states.grad)
        grad_norm = torch.norm(self.hidden_states.grad)
        self.assertGreater(grad_norm.item(), 0, "Gradient norm should be positive")

if __name__ == '__main__':
    unittest.main()
