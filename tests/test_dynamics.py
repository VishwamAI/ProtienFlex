"""Tests for molecular dynamics simulations"""
import unittest
import torch
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pathlib import Path
from models.dynamics.simulation import MolecularDynamics

class TestMolecularDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.dynamics = MolecularDynamics()

        # Create a simple test PDB file
        cls.test_pdb = "test_protein.pdb"
        cls._create_test_pdb()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        if Path(cls.test_pdb).exists():
            Path(cls.test_pdb).unlink()

    @classmethod
    def _create_test_pdb(cls):
        """Create a simple test PDB file"""
        # Create a simple alanine dipeptide system
        pdb = app.PDBFile('alanine-dipeptide.pdb')
        with open(cls.test_pdb, 'w') as f:
            app.PDBFile.writeFile(
                pdb.topology,
                pdb.positions,
                f
            )

    def test_platform_selection(self):
        """Test automatic platform selection and device compatibility"""
        # Test CPU platform
        cpu_dynamics = MolecularDynamics(device='cpu')
        self.assertEqual(cpu_dynamics.platform.getName(), 'CPU')

        # Test CUDA platform if available
        if 'CUDA' in [mm.Platform.getPlatform(i).getName()
                     for i in range(mm.Platform.getNumPlatforms())]:
            cuda_dynamics = MolecularDynamics(device='cuda')
            self.assertEqual(cuda_dynamics.platform.getName(), 'CUDA')

    def test_simulation_setup(self):
        """Test simulation setup and system preparation"""
        simulation, modeller = self.dynamics.setup_simulation(self.test_pdb)

        # Verify simulation components
        self.assertIsInstance(simulation, app.Simulation)
        self.assertIsInstance(modeller, app.Modeller)

        # Check system setup
        system = simulation.system
        self.assertTrue(system.usesPeriodicBoundaryConditions())
        self.assertGreater(system.getNumParticles(), 0)

    def test_minimize_and_equilibrate(self):
        """Test energy minimization and equilibration"""
        simulation, _ = self.dynamics.setup_simulation(self.test_pdb)
        result = self.dynamics.minimize_and_equilibrate(simulation)

        # Check result structure
        self.assertIn('potential_energy', result)
        self.assertIn('kinetic_energy', result)
        self.assertIn('temperature', result)

        # Verify energy values
        self.assertIsInstance(result['potential_energy'], float)
        self.assertIsInstance(result['kinetic_energy'], float)
        self.assertGreater(result['temperature'], 0)

    def test_run_dynamics(self):
        """Test molecular dynamics simulation"""
        simulation, _ = self.dynamics.setup_simulation(self.test_pdb)
        self.dynamics.minimize_and_equilibrate(simulation)
        result = self.dynamics.run_dynamics(simulation, steps=100)

        # Check result structure
        self.assertIn('potential_energy', result)
        self.assertIn('kinetic_energy', result)
        self.assertIn('temperature', result)
        self.assertIn('positions', result)

        # Verify trajectory data
        self.assertIsInstance(result['positions'], np.ndarray)
        self.assertEqual(len(result['positions'].shape), 2)  # (n_atoms, 3)

    def test_trajectory_analysis(self):
        """Test trajectory analysis"""
        # Create mock trajectory data
        n_frames = 10
        n_atoms = 5
        positions = np.random.rand(n_frames, n_atoms, 3)

        result = self.dynamics.analyze_trajectory(positions)

        # Check result structure
        self.assertIn('rmsd', result)
        self.assertIn('average_structure', result)
        self.assertIn('structure_variance', result)

        # Verify analysis results
        self.assertIsInstance(result['rmsd'], float)
        self.assertEqual(result['average_structure'].shape, (n_atoms, 3))
        self.assertEqual(result['structure_variance'].shape, (n_atoms, 3))

    def test_integration_with_mutation_analysis(self):
        """Test integration with mutation analysis workflow"""
        from models.mutation_analysis import MutationAnalyzer
        import esm

        # Initialize mutation analyzer
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        analyzer = MutationAnalyzer(model, device)

        # Run mutation analysis
        sequence = "FVNQHLCGSHLVEAL"
        position = 7
        mutation = "A"
        mutation_result = analyzer.predict_mutation_effect(sequence, position, mutation)


        # Verify mutation analysis results
        self.assertIsInstance(mutation_result, dict)
        self.assertIn('stability_impact', mutation_result)
        self.assertIn('structural_impact', mutation_result)

        # Setup and run dynamics
        simulation, _ = self.dynamics.setup_simulation(self.test_pdb)
        dynamics_result = self.dynamics.minimize_and_equilibrate(simulation)

        # Verify combined results
        self.assertIsInstance(dynamics_result, dict)
        self.assertIn('potential_energy', dynamics_result)
        self.assertIn('temperature', dynamics_result)

if __name__ == '__main__':
    unittest.main()
