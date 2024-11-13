"""Tests for molecular dynamics simulations"""
import unittest
import torch
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pathlib import Path
from models.dynamics.simulation import MolecularDynamics, EnhancedSampling

class TestMolecularDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.dynamics = MolecularDynamics()

        # Set up paths
        cls.test_dir = Path(__file__).parent
        cls.test_pdb = cls.test_dir / "data" / "test_protein.pdb"
        cls.alanine_pdb = cls.test_dir / "data" / "alanine-dipeptide.pdb"
        cls._create_test_pdb()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        if cls.test_pdb.exists():
            cls.test_pdb.unlink()

    @classmethod
    def _create_test_pdb(cls):
        """Create a test PDB file with proper terminal residues"""
        with open(cls.test_pdb, 'w') as f:
            f.write("ATOM      1  N   ALA     1      -0.529   1.359   0.000  1.00  0.00           N  \n")
            f.write("ATOM      2  H1  ALA     1      -1.540   1.359   0.000  1.00  0.00           H  \n")
            f.write("ATOM      3  H2  ALA     1      -0.190   1.822   0.825  1.00  0.00           H  \n")
            f.write("ATOM      4  H3  ALA     1      -0.190   1.822  -0.825  1.00  0.00           H  \n")
            f.write("ATOM      5  CA  ALA     1       0.000   0.000   0.000  1.00  0.00           C  \n")
            f.write("ATOM      6  HA  ALA     1      -0.481  -0.526   0.825  1.00  0.00           H  \n")
            f.write("ATOM      7  CB  ALA     1      -0.507  -0.785  -1.207  1.00  0.00           C  \n")
            f.write("ATOM      8  HB1 ALA     1      -0.082  -1.786  -1.207  1.00  0.00           H  \n")
            f.write("ATOM      9  HB2 ALA     1      -0.267  -0.315  -2.150  1.00  0.00           H  \n")
            f.write("ATOM     10  HB3 ALA     1      -1.593  -0.869  -1.150  1.00  0.00           H  \n")
            f.write("ATOM     11  C   ALA     1       1.520   0.000   0.000  1.00  0.00           C  \n")
            f.write("ATOM     12  O   ALA     1       2.145   1.057   0.000  1.00  0.00           O  \n")
            f.write("ATOM     13  N   ALA     2       2.116  -1.186   0.000  1.00  0.00           N  \n")
            f.write("ATOM     14  H   ALA     2       1.573  -2.037   0.000  1.00  0.00           H  \n")
            f.write("ATOM     15  CA  ALA     2       3.571  -1.268   0.000  1.00  0.00           C  \n")
            f.write("ATOM     16  HA  ALA     2       3.923  -0.742  -0.887  1.00  0.00           H  \n")
            f.write("ATOM     17  CB  ALA     2       4.167  -0.668   1.272  1.00  0.00           C  \n")
            f.write("ATOM     18  HB1 ALA     2       3.815  -1.194   2.159  1.00  0.00           H  \n")
            f.write("ATOM     19  HB2 ALA     2       3.923   0.389   1.336  1.00  0.00           H  \n")
            f.write("ATOM     20  HB3 ALA     2       5.253  -0.768   1.272  1.00  0.00           H  \n")
            f.write("ATOM     21  C   ALA     2       4.097  -2.453   0.791  1.00  0.00           C  \n")
            f.write("ATOM     22  O   ALA     2       5.311  -2.655   0.827  1.00  0.00           O  \n")
            f.write("ATOM     23  OXT ALA     2       3.311  -3.186   1.582  1.00  0.00           O  \n")
            f.write("TER      24      ALA     2\n")
            f.write("END\n")

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
        simulation, modeller = self.dynamics.setup_simulation(str(self.test_pdb))

        # Verify simulation components
        self.assertIsInstance(simulation, app.Simulation)
        self.assertIsInstance(modeller, app.Modeller)

        # Check system setup
        system = simulation.system
        self.assertTrue(system.usesPeriodicBoundaryConditions())
        self.assertGreater(system.getNumParticles(), 0)

    def test_minimize_and_equilibrate(self):
        """Test energy minimization and equilibration"""
        simulation, _ = self.dynamics.setup_simulation(str(self.test_pdb))
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
        simulation, _ = self.dynamics.setup_simulation(str(self.test_pdb))
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
        simulation, _ = self.dynamics.setup_simulation(str(self.test_pdb))
        dynamics_result = self.dynamics.minimize_and_equilibrate(simulation)

        # Verify combined results
        self.assertIsInstance(dynamics_result, dict)
        self.assertIn('potential_energy', dynamics_result)
        self.assertIn('temperature', dynamics_result)


class TestEnhancedSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.sampling = EnhancedSampling()

        # Set up paths
        cls.test_dir = Path(__file__).parent
        cls.test_pdb = cls.test_dir / "test_protein.pdb"
        cls.alanine_pdb = cls.test_dir / "alanine-dipeptide.pdb"
        cls._create_test_pdb()

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        if cls.test_pdb.exists():
            cls.test_pdb.unlink()

    @classmethod
    def _create_test_pdb(cls):
        """Create a simple test PDB file"""
        pdb = app.PDBFile(str(cls.alanine_pdb))
        with open(cls.test_pdb, 'w') as f:
            app.PDBFile.writeFile(
                pdb.topology,
                pdb.positions,
                f
            )

    def test_replica_exchange_setup(self):
        """Test replica exchange setup"""
        replicas = self.sampling.setup_replica_exchange(
            str(self.test_pdb),
            n_replicas=2,
            temp_range=(300.0, 350.0)
        )

        # Verify replicas
        self.assertEqual(len(replicas), 2)
        for replica in replicas:
            self.assertIsInstance(replica, app.Simulation)

        # Verify temperatures
        temp1 = replicas[0].integrator.getTemperature()
        temp2 = replicas[1].integrator.getTemperature()
        self.assertLess(temp1, temp2)

    def test_replica_exchange_run(self):
        """Test running replica exchange"""
        # Setup replicas
        self.sampling.setup_replica_exchange(
            str(self.test_pdb),
            n_replicas=2,
            temp_range=(300.0, 350.0)
        )

        # Run replica exchange
        results = self.sampling.run_replica_exchange(
            exchange_steps=2,
            dynamics_steps=10
        )

        # Verify results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)  # Two exchange steps
        for states in results:
            self.assertEqual(len(states), 2)  # Two replicas
            for state in states:
                self.assertIn('potential_energy', state)
                self.assertIn('kinetic_energy', state)
                self.assertIn('temperature', state)

if __name__ == '__main__':
    unittest.main()
