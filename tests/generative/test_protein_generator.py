import unittest
import torch
import math
from models.generative.protein_generator import ProteinGenerativeModel, ProteinGenerativeConfig

class TestProteinGenerator(unittest.TestCase):
    def setUp(self):
        self.config = ProteinGenerativeConfig(
            vocab_size=25,  # Standard amino acid vocabulary
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            num_concepts=64
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ProteinGenerativeModel(self.config).to(self.device)

        # Sample input data
        self.batch_size = 4
        self.seq_length = 16
        self.input_ids = torch.randint(
            0, self.config.vocab_size,
            (self.batch_size, self.seq_length),
            device=self.device
        )

    def test_forward_pass(self):
        """Test forward pass with concept bottleneck"""
        outputs = self.model(
            input_ids=self.input_ids,
            output_attentions=True,
            output_hidden_states=True,
            return_concepts=True
        )

        # Check output shapes
        self.assertEqual(
            outputs["logits"].shape,
            (self.batch_size, self.seq_length, self.config.vocab_size)
        )
        self.assertIn("concepts", outputs)
        self.assertEqual(len(outputs["concepts"]), 4)  # Four concept groups

        # Check structural angles
        self.assertIn("structural_angles", outputs)
        self.assertEqual(
            outputs["structural_angles"].shape[-1],
            3  # phi, psi, omega angles
        )

    def test_generate_with_concepts(self):
        """Test protein generation with concept guidance"""
        prompt_text = "Generate a stable alpha-helical protein"
        target_concepts = {
            "structure": torch.tensor([0.8, 0.2, 0.1], device=self.device),  # Alpha helix preference
            "chemistry": torch.tensor([0.6, 0.4, 0.5], device=self.device),
            "function": torch.tensor([0.7, 0.3, 0.4], device=self.device),
            "interaction": torch.tensor([0.5, 0.5, 0.5], device=self.device)
        }

        sequences = self.model.generate(
            prompt_text=prompt_text,
            max_length=32,
            num_return_sequences=2,
            temperature=0.8,
            concept_guidance=True,
            target_concepts=target_concepts
        )

        # Check generated sequences
        self.assertEqual(len(sequences), 2)
        for seq in sequences:
            # Verify sequence contains valid amino acids
            valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
            self.assertTrue(all(aa in valid_aas for aa in seq))

            # Check sequence length
            self.assertTrue(len(seq) <= 32)

    def test_structural_validation(self):
        """Test structural validation during generation"""
        angles = torch.randn(self.batch_size, self.seq_length, 3, device=self.device)
        scores = self.model._evaluate_structural_validity(angles)

        # Check score shape and range
        self.assertEqual(scores.shape, (self.batch_size,))
        self.assertTrue(torch.all(scores >= 0))
        self.assertTrue(torch.all(scores <= 1))

    def test_concept_alignment(self):
        """Test concept alignment evaluation"""
        current_concepts = {
            "structure": torch.rand(self.batch_size, self.seq_length, 16, device=self.device),
            "chemistry": torch.rand(self.batch_size, self.seq_length, 16, device=self.device),
            "function": torch.rand(self.batch_size, self.seq_length, 16, device=self.device),
            "interaction": torch.rand(self.batch_size, self.seq_length, 16, device=self.device)
        }

        target_concepts = {
            "structure": torch.tensor([0.8, 0.2, 0.1], device=self.device),
            "chemistry": torch.tensor([0.6, 0.4, 0.5], device=self.device)
        }

        scores = self.model._evaluate_concept_alignment(current_concepts, target_concepts)

        # Check score shape and range
        self.assertEqual(scores.shape, (self.batch_size,))
        self.assertTrue(torch.all(scores >= 0))
        self.assertTrue(torch.all(scores <= 1))

    def test_template_guidance(self):
        """Test template-guided generation"""
        template_sequence = "MLKFVAVVVL"
        sequences = self.model.generate(
            prompt_text="Generate a protein similar to the template",
            max_length=32,
            num_return_sequences=2,
            template_sequence=template_sequence
        )

        # Check generated sequences
        self.assertEqual(len(sequences), 2)
        for seq in sequences:
            # Verify sequence contains valid amino acids
            valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
            self.assertTrue(all(aa in valid_aas for aa in seq))

            # Check sequence length
            self.assertTrue(len(seq) <= 32)


if __name__ == '__main__':
    unittest.main()
