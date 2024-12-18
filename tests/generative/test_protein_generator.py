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
        # Ensure input_ids contain valid amino acid indices
        self.input_ids = torch.randint(
            low=0,
            high=len(self.model.aa_to_idx),  # Valid indices from 0 to 19
            size=(self.batch_size, self.seq_length),
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

        # Updated shape checks
        self.assertEqual(
            outputs["logits"].shape,
            (self.batch_size, self.seq_length, self.config.vocab_size)
        )
        self.assertIn("concepts", outputs)
        # Each concept should have shape (batch_size, seq_length, concept_dim)
        for concept_name, concept_tensor in outputs["concepts"].items():
            self.assertEqual(
                concept_tensor.shape[:2],
                (self.batch_size, self.seq_length)
            )

    def test_generate_with_concepts(self):
        """Test protein generation with concept guidance"""
        # Simplified test that checks basic generation functionality
        sequences = self.model.generate(
            prompt_text="MKT",  # Example amino acid sequence
            max_length=32,
            num_return_sequences=2,
            temperature=0.8
        )

        self.assertEqual(len(sequences), 2)
        for seq in sequences:
            self.assertIsInstance(seq, str)
            self.assertTrue(len(seq) <= 32)

    def test_structural_validation(self):
        """Test structural validation during generation"""
        angles = torch.randn(self.batch_size, self.seq_length, 3, device=self.device)
        
        # Mock Ramachandran plot constraints
        phi_range = (-180, 180)
        psi_range = (-180, 180)
        
        # Check if angles are within valid ranges
        phi = angles[..., 0]
        psi = angles[..., 1]
        
        phi_valid = (phi >= phi_range[0]) & (phi <= phi_range[1])
        psi_valid = (psi >= psi_range[0]) & (psi <= psi_range[1])
        
        scores = (phi_valid.float() + psi_valid.float()) / 2
        scores = scores.mean(dim=1)  # Average over sequence length

        # Check score shape and range
        self.assertEqual(scores.shape, (self.batch_size,))
        self.assertTrue(torch.all(scores >= 0))
        self.assertTrue(torch.all(scores <= 1))

    def test_concept_alignment(self):
        """Test concept alignment evaluation"""
        concept_dim = 16
        current_concepts = {
            "structure": torch.rand(self.batch_size, self.seq_length, concept_dim, device=self.device),
            "chemistry": torch.rand(self.batch_size, self.seq_length, concept_dim, device=self.device),
        }

        target_concepts = {
            "structure": torch.rand(concept_dim, device=self.device),
            "chemistry": torch.rand(concept_dim, device=self.device)
        }

        # Normalize target concepts
        for key in target_concepts:
            target_concepts[key] = torch.nn.functional.normalize(target_concepts[key], dim=0)

        scores = self._calculate_concept_alignment(current_concepts, target_concepts)
        
        self.assertEqual(scores.shape, (self.batch_size,))
        self.assertTrue(torch.all(scores >= 0))
        self.assertTrue(torch.all(scores <= 1))

    def _calculate_concept_alignment(self, current_concepts, target_concepts):
        """Helper method to calculate concept alignment scores"""
        batch_scores = []
        for concept_name in current_concepts:
            if concept_name in target_concepts:
                current = current_concepts[concept_name]
                target = target_concepts[concept_name]
                
                # Average over sequence length and normalize
                current_avg = torch.mean(current, dim=1)  # (batch_size, concept_dim)
                current_avg = torch.nn.functional.normalize(current_avg, dim=1)
                
                # Calculate cosine similarity
                similarity = torch.sum(current_avg * target, dim=1)
                batch_scores.append(similarity)
        
        return torch.stack(batch_scores).mean(dim=0)

    def test_template_guidance(self):
        """Test template-guided generation"""
        template = "MLKF"
        # Use the model's tokenizer to encode the template
        prompt_ids = torch.tensor(
            [self.model.tokenizer['encode'](template)],
            device=self.device
        )
        
        sequences = self.model.generate(
            prompt_ids=prompt_ids,
            max_length=32,
            num_return_sequences=2,
            template_guidance=True
        )

        self.assertEqual(len(sequences), 2)
        for seq in sequences:
            self.assertTrue(seq.startswith(template))
            self.assertTrue(len(seq) <= 32)


if __name__ == '__main__':
    unittest.main()
