import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from Bio.PDB import Structure, Model, Chain, Residue
from models.analysis.domain_analysis import DomainAnalyzer

class TestDomainAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.domain_analyzer = DomainAnalyzer()
        self.test_sequence = "MKLLVLGLCALIISASCKS"
        
        # Mock ESM model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_embeddings = torch.randn(1, len(self.test_sequence), 1280)
        self.mock_attention = torch.randn(1, len(self.test_sequence), len(self.test_sequence))
        
        # Mock model output
        self.mock_model.return_value = {
            'representations': {33: self.mock_embeddings},
            'attentions': self.mock_attention
        }
        
        # Set up mock structure
        self.structure = Structure.Structure('test')
        model = Model.Model(0)
        chain = Chain.Chain('A')
        for i, aa in enumerate(self.test_sequence):
            res = Residue.Residue((' ', i, ' '), aa, '')
            chain.add(res)
        model.add(chain)
        self.structure.add(model)

    @patch('models.analysis.domain_analysis.esm')
    def test_predict_domains(self, mock_esm):
        """Test domain prediction functionality."""
        mock_esm.pretrained.return_value = (self.mock_model, self.mock_tokenizer)
        domains = self.domain_analyzer.predict_domains(self.test_sequence)
        
        self.assertIsInstance(domains, dict)
        self.assertIn('domains', domains)
        self.assertIsInstance(domains['domains'], list)

    def test_analyze_domain_contacts(self):
        """Test domain contact analysis."""
        contact_map = np.random.rand(len(self.test_sequence), len(self.test_sequence))
        contact_map = (contact_map + contact_map.T) / 2
        contacts = self.domain_analyzer.analyze_domain_contacts(
            self.structure,
            contact_map,
            distance_threshold=8.0
        )
        
        self.assertIsInstance(contacts, dict)
        self.assertIn('inter_domain_contacts', contacts)
        self.assertIn('contact_strength', contacts)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        with self.assertRaises(ValueError):
            self.domain_analyzer.predict_domains("")
        with self.assertRaises(ValueError):
            self.domain_analyzer.analyze_domain_contacts(None, np.array([]))

if __name__ == '__main__':
    unittest.main()
