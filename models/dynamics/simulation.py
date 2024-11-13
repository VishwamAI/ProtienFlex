from openmm import app, unit, LangevinMiddleIntegrator, Platform
import numpy as np
import logging

class MolecularDynamics:
    def __init__(self, device='cpu'):
        self.simulation = None
        self.device = device
        self.platform = Platform.getPlatformByName('CPU' if device == 'cpu' else 'CUDA')

    def setup_simulation(self, pdb_file):
        pdb = app.PDBFile(pdb_file)
        modeller = app.Modeller(pdb.topology, pdb.positions)
        force_field = app.ForceField('amber14-all.xml')
        system = force_field.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0*unit.nanometers,
            constraints=app.HBonds
        )
        integrator = LangevinMiddleIntegrator(
            300*unit.kelvin,
            1.0/unit.picosecond,
            0.002*unit.picoseconds
        )
        self.simulation = app.Simulation(modeller.topology, system, integrator, self.platform)
        self.simulation.context.setPositions(modeller.positions)
        return self.simulation, modeller

    def minimize_and_equilibrate(self, simulation, min_steps=100, equil_steps=1000):
        """Minimize and equilibrate the system"""
        simulation.minimizeEnergy(maxIterations=min_steps)
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
        simulation.step(equil_steps)
        state = simulation.context.getState(getEnergy=True, getTemperature=True)
        return {
            'potential_energy': state.getPotentialEnergy(),
            'kinetic_energy': state.getKineticEnergy(),
            'temperature': state.getTemperature()
        }

    def run_dynamics(self, simulation, steps=1000):
        """Run molecular dynamics simulation"""
        simulation.step(steps)
        state = simulation.context.getState(getEnergy=True, getTemperature=True, getPositions=True)
        return {
            'potential_energy': state.getPotentialEnergy(),
            'kinetic_energy': state.getKineticEnergy(),
            'temperature': state.getTemperature(),
            'positions': state.getPositions()
        }

    def analyze_trajectory(self, positions):
        """Analyze trajectory data"""
        rmsd = np.sqrt(np.mean(np.sum((positions - positions[0])**2, axis=2), axis=1))
        avg_structure = np.mean(positions, axis=0)
        structure_variance = np.var(positions, axis=0)
        return {
            'rmsd': rmsd,
            'average_structure': avg_structure,
            'structure_variance': structure_variance
        }

class EnhancedSampling:
    def __init__(self):
        self.replicas = []
        self.temperatures = []

    def setup_replica_exchange(self, pdb_file, n_replicas=4, temp_range=(300.0, 400.0)):
        """Setup replica exchange simulation"""
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

    def run_replica_exchange(self, exchange_steps=100, dynamics_steps=1000):
        """Run replica exchange molecular dynamics"""
        results = []
        for step in range(exchange_steps):
            # Run dynamics for each replica
            for replica in self.replicas:
                replica.step(dynamics_steps)

            # Attempt exchanges between neighboring replicas
            for i in range(len(self.replicas)-1):
                success = self._attempt_exchange(i, i+1)
                if success:
                    logging.info(f"Exchange successful between replicas {i} and {i+1}")

            # Record state
            state = self._get_replica_states()
            results.append(state)

        return results

    def _attempt_exchange(self, i, j):
        """Attempt exchange between replicas i and j"""
        energy_i = self.replicas[i].context.getState(getEnergy=True).getPotentialEnergy()._value
        energy_j = self.replicas[j].context.getState(getEnergy=True).getPotentialEnergy()._value

        beta_i = 1.0 / (0.0083144621 * self.temperatures[i].value_in_unit(unit.kelvin))
        beta_j = 1.0 / (0.0083144621 * self.temperatures[j].value_in_unit(unit.kelvin))

        delta = float((beta_i - beta_j) * (energy_i - energy_j))

        if delta <= 0.0 or np.random.random() < np.exp(-delta):
            self._swap_replicas(i, j)
            return True
        return False

    def _swap_replicas(self, i, j):
        """Swap positions between replicas i and j"""
        pos_i = self.replicas[i].context.getState(getPositions=True).getPositions()
        pos_j = self.replicas[j].context.getState(getPositions=True).getPositions()

        self.replicas[i].context.setPositions(pos_j)
        self.replicas[j].context.setPositions(pos_i)

    def _get_replica_states(self):
        """Get current state of all replicas"""
        return [{
            'temperature': temp,
            'potential_energy': replica.context.getState(getEnergy=True).getPotentialEnergy(),
            'positions': replica.context.getState(getPositions=True).getPositions()
        } for temp, replica in zip(self.temperatures, self.replicas)]

    def _create_system(self, topology):
        """Create OpenMM system with force field"""
        force_field = app.ForceField('amber14-all.xml')
        system = force_field.createSystem(
            topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0*unit.nanometers,
            constraints=app.HBonds
        )
        return system

    def _create_integrator(self, temperature):
        """Create Langevin integrator with specified temperature"""
        return LangevinMiddleIntegrator(
            temperature,
            1.0/unit.picosecond,
            0.002*unit.picoseconds
        )
