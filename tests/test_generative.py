"""
Tests for generative AI components
"""
import unittest
import os
import torch
import numpy as np
from Bio.PDB import *
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO
from models.generative.protein_generator import ProteinGenerativeModel, ProteinGenerativeConfig
from models.generative.structure_predictor import StructurePredictor
from models.generative.virtual_screening import VirtualScreening

# Ensure test directory exists
os.makedirs('tests', exist_ok=True)

class TestProteinGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.config = ProteinGenerativeConfig(
            vocab_size=30,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
        )
        cls.model = ProteinGenerativeModel(cls.config)

    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, ProteinGenerativeModel)
        self.assertEqual(self.model.config.vocab_size, 30)
        self.assertEqual(self.model.config.hidden_size, 256)

    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, self.config.vocab_size - 1, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))

        outputs = self.model(input_ids, attention_mask=attention_mask)
        self.assertIsInstance(outputs, dict)
        self.assertIn('last_hidden_state', outputs)
        self.assertIn('logits', outputs)
        self.assertEqual(outputs['last_hidden_state'].shape, (batch_size, seq_length, self.config.hidden_size))
        self.assertEqual(outputs['logits'].shape, (batch_size, seq_length, self.config.vocab_size))

    def test_sequence_generation(self):
        """Test protein sequence generation"""
        input_ids = torch.randint(0, self.config.vocab_size - 1, (1, 5))
        attention_mask = torch.ones((1, 5))

        generated = self.model.generate(
            input_ids=input_ids,
            max_length=20,
            temperature=0.8,
            attention_mask=attention_mask,
        )

        self.assertIsInstance(generated, torch.Tensor)
        self.assertEqual(len(generated.shape), 2)
        self.assertGreater(generated.shape[1], input_ids.shape[1])
        self.assertTrue(torch.all(generated < self.config.vocab_size))

class TestStructurePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.model = StructurePredictor(
            hidden_size=256,
            num_layers=4,
        )

    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, StructurePredictor)
        self.assertEqual(self.model.hidden_size, 256)

    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 2
        seq_length = 10
        sequence_features = torch.randn(batch_size, seq_length, 20)

        outputs = self.model(sequence_features)
        self.assertIsInstance(outputs, dict)
        self.assertIn('distances', outputs)
        self.assertIn('angles', outputs)
        self.assertIn('coordinates', outputs)
        self.assertIn('hidden_states', outputs)

        # Verify tensor dimensions
        self.assertEqual(outputs['distances'].shape[:2], (batch_size, seq_length))
        self.assertEqual(outputs['angles'].shape, (batch_size, seq_length, 3))  # phi, psi, omega
        self.assertEqual(outputs['coordinates'].shape, (batch_size, seq_length, 3))  # x, y, z coordinates

    def test_structure_prediction(self):
        """Test structure prediction"""
        sequence = "ACDEFGHIKLM"
        structure = self.model.predict_structure(sequence)

        # Check if structure is a Bio.PDB.Structure object
        self.assertTrue(isinstance(structure, Structure))

        # Verify structure contents
        self.assertTrue(len(structure) > 0)  # Has at least one model
        self.assertTrue(len(list(structure.get_models())) > 0)  # Has at least one model
        model = list(structure.get_models())[0]
        self.assertTrue(len(list(model.get_chains())) > 0)  # Has at least one chain
        chain = list(model.get_chains())[0]
        self.assertEqual(len(list(chain.get_residues())), len(sequence))  # Has correct number of residues

        # Verify basic structure properties
        for residue in chain:
            self.assertTrue(isinstance(residue, Residue))
            self.assertTrue(len(residue) >= 3)  # Should have at least backbone atoms
            self.assertTrue('CA' in residue)  # Should have alpha carbon
            self.assertTrue('N' in residue)   # Should have backbone nitrogen
            self.assertTrue('C' in residue)   # Should have backbone carbon

class TestVirtualScreening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.screening = VirtualScreening(device='cpu')

        # Create a simple test structure using Bio.PDB classes
        parser = PDBParser()
        cls.structure = parser.get_structure('test', 'tests/alanine-dipeptide.pdb')

        if not os.path.exists('tests/alanine-dipeptide.pdb'):
            # Create a simple structure file if it doesn't exist
            structure_builder = PDBIO()
            structure = Structure.Structure('test')
            model = Model.Model(0)
            chain = Chain.Chain('A')

            for i in range(5):
                res = Residue.Residue((' ', i, ' '), 'ALA', '')
                ca = Atom.Atom('CA', np.array([float(i), 0.0, 0.0]), 20.0, 1.0, ' ', 'CA', i)
                n = Atom.Atom('N', np.array([float(i), 0.5, 0.0]), 20.0, 1.0, ' ', 'N', i)
                c = Atom.Atom('C', np.array([float(i), -0.5, 0.0]), 20.0, 1.0, ' ', 'C', i)
                o = Atom.Atom('O', np.array([float(i), -0.5, 0.5]), 20.0, 1.0, ' ', 'O', i)

                res.add(ca)
                res.add(n)
                res.add(c)
                res.add(o)
                chain.add(res)

            model.add(chain)
            structure.add(model)

            structure_builder.set_structure(structure)
            structure_builder.save('tests/alanine-dipeptide.pdb')
            cls.structure = structure

    def test_molecule_preparation(self):
        """Test molecule preparation"""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        features = self.screening.prepare_molecule(smiles)

        self.assertIsInstance(features, dict)
        self.assertIn('atom_features', features)
        self.assertIn('bond_features', features)
        self.assertIn('adjacency', features)

    def test_compound_screening(self):
        """Test compound screening"""
        compounds = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC1=C(C=C(C=C1)O)C(=O)O",    # Salicylic acid
        ]

        results = self.screening.screen_compounds(
            self.structure,
            compounds,
            batch_size=2,
        )

        self.assertEqual(len(results), len(compounds))
        for result in results:
            self.assertIn('binding_score', result)
            self.assertIn('binding_affinity', result)
            self.assertIn('interaction_sites', result)
            self.assertIsInstance(result['binding_score'], float)
            self.assertIsInstance(result['binding_affinity'], float)
            self.assertEqual(len(result['interaction_sites']), 3)

if __name__ == '__main__':
    unittest.main()
