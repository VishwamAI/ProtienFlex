from openmm import app, unit, LangevinMiddleIntegrator
import numpy as np

class MolecularDynamics:
    def __init__(self, device='cpu'):
        self.simulation = None
        self.device = device

    def setup_simulation(self, pdb_file):
        pdb = app.PDBFile(pdb_file)
        force_field = app.ForceField('amber14-all.xml')
        system = force_field.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds
        )
        integrator = LangevinMiddleIntegrator(
            300*unit.kelvin,
            1.0/unit.picosecond,
            0.002*unit.picoseconds
        )
        self.simulation = app.Simulation(pdb.topology, system, integrator)
        self.simulation.context.setPositions(pdb.positions)
        return self.simulation, pdb

    def analyze_trajectory(self, positions):
        """Analyze trajectory data"""
        rmsd = np.sqrt(np.mean(np.sum((positions - positions[0])**2, axis=2), axis=1))
        return {
            'rmsd': rmsd,
            'avg_structure': np.mean(positions, axis=0)
        }

class EnhancedSampling:
    def __init__(self):
        self.replicas = []
        self.temperatures = []

    def setup_replica_exchange(self, pdb_file, n_replicas=4, temp_range=(300.0, 400.0)):
        min_temp, max_temp = temp_range

        # Generate temperature ladder using geometric progression
        beta_min = 1.0 / (0.0083144621 * max_temp)
        beta_max = 1.0 / (0.0083144621 * min_temp)
        betas = np.exp(np.linspace(np.log(beta_min), np.log(beta_max), n_replicas))
        self.temperatures = [1.0 / (0.0083144621 * beta) * unit.kelvin for beta in reversed(betas)]

        # Create replicas
        self.replicas = []
        for temp in self.temperatures:
            pdb = app.PDBFile(pdb_file)
            system = self._create_system(pdb.topology)
            integrator = self._create_integrator(temp)
            simulation = app.Simulation(pdb.topology, system, integrator)
            simulation.context.setPositions(pdb.positions)
            self.replicas.append(simulation)

        return self.replicas

    def _create_system(self, topology):
        force_field = app.ForceField('amber14-all.xml')
        system = force_field.createSystem(
            topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds
        )
        return system

    def _create_integrator(self, temperature):
        return LangevinMiddleIntegrator(
            temperature,
            1.0/unit.picosecond,
            0.002*unit.picoseconds
        )
