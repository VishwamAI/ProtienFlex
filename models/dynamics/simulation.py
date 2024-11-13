# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Module for molecular dynamics simulations using OpenMM"""
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MolecularDynamics:
    def __init__(self, device: Optional[str] = None):
        """Initialize molecular dynamics simulation with device detection"""
        # Automatic device detection for OpenMM
        self.platform = self._setup_platform(device)
        self.properties = {}
        if self.platform.getName() == 'CUDA':
            self.properties = {'CudaPrecision': 'mixed'}

        logger.info(f"Using OpenMM platform: {self.platform.getName()}")

    def _setup_platform(self, device: Optional[str] = None) -> mm.Platform:
        """Setup OpenMM platform with device detection"""
        available_platforms = [mm.Platform.getPlatform(i).getName()
                             for i in range(mm.Platform.getNumPlatforms())]

        if device == 'cuda' and 'CUDA' in available_platforms:
            return mm.Platform.getPlatformByName('CUDA')
        elif device == 'cpu' and 'CPU' in available_platforms:
            return mm.Platform.getPlatformByName('CPU')

        # Automatic selection based on availability
        for platform in ['CUDA', 'OpenCL', 'CPU']:
            if platform in available_platforms:
                return mm.Platform.getPlatformByName(platform)

        raise RuntimeError("No suitable OpenMM platform found")

    def setup_simulation(self, pdb_file: str, forcefield: str = 'amber14-all.xml') -> Tuple[app.Simulation, app.Modeller]:
        """Setup molecular dynamics simulation"""
        try:
            # Load protein structure
            pdb = app.PDBFile(pdb_file)

            # Initial vacuum minimization to remove bad contacts
            vacuum_ff = app.ForceField('amber14/protein.ff14SB.xml')
            vacuum_modeller = app.Modeller(pdb.topology, pdb.positions)

            # Define residue templates
            templates = {}
            for chain in vacuum_modeller.topology.chains():
                residues = list(chain.residues())
                for i, res in enumerate(residues):
                    if i == 0:  # First residue in chain
                        templates[res] = "NALA" if res.name in ["ALA", "NAL"] else f"N{res.name}"
                    elif i == len(residues) - 1:  # Last residue
                        templates[res] = "CALA" if res.name in ["ALA", "CAL"] else f"C{res.name}"

            # Create vacuum system for initial minimization
            vacuum_system = vacuum_ff.createSystem(
                vacuum_modeller.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                rigidWater=True,
                residueTemplates=templates
            )

            # Add harmonic restraints to heavy atoms
            force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            force.addGlobalParameter("k", 100.0 * unit.kilojoules_per_mole/unit.nanometer**2)
            force.addPerParticleParameter("x0")
            force.addPerParticleParameter("y0")
            force.addPerParticleParameter("z0")

            # Add restraints to all heavy atoms
            positions = vacuum_modeller.positions
            for i, atom in enumerate(vacuum_modeller.topology.atoms()):
                if atom.element.mass > 2.0 * unit.daltons:
                    force.addParticle(i, positions[i].value_in_unit(unit.nanometers))
            vacuum_system.addForce(force)

            # Simple integrator for vacuum minimization with smaller timestep
            vacuum_integrator = mm.LangevinIntegrator(
                300*unit.kelvin,
                1.0/unit.picosecond,
                0.0005*unit.picoseconds
            )
            vacuum_integrator.setConstraintTolerance(1e-6)

            # Create vacuum simulation
            vacuum_sim = app.Simulation(
                vacuum_modeller.topology,
                vacuum_system,
                vacuum_integrator,
                self.platform,
                self.properties
            )

            # Perform staged vacuum minimization
            vacuum_sim.context.setPositions(vacuum_modeller.positions)
            logger.info("Starting vacuum minimization...")
            vacuum_sim.minimizeEnergy(maxIterations=50, tolerance=100.0)
            vacuum_sim.minimizeEnergy(maxIterations=50, tolerance=10.0)

            # Get minimized positions
            minimized_positions = vacuum_sim.context.getState(getPositions=True).getPositions()

            # Now set up full system with solvent
            forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
            modeller = app.Modeller(pdb.topology, minimized_positions)

            # Add solvent with explicit templates and proper periodic box
            modeller.addSolvent(
                forcefield,
                model='tip3p',
                padding=2.0*unit.nanometers,  # Increased padding for stability
                neutralize=True,
                positiveIon='Na+',
                negativeIon='Cl-',
                ionicStrength=0.1*unit.molar,
                residueTemplates=templates
            )

            # Create system with periodic boundary conditions and constraints
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=0.9*unit.nanometers,  # Reduced cutoff for stability
                constraints=app.HBonds,
                rigidWater=True,
                ewaldErrorTolerance=0.0005,
                hydrogenMass=1.5*unit.amu,
                residueTemplates=templates
            )

            # Add barostat for pressure control with reduced frequency
            barostat = mm.MonteCarloBarostat(1.0*unit.atmospheres, 300*unit.kelvin, 50)
            system.addForce(barostat)

            # Create integrator with appropriate parameters for stability
            integrator = mm.LangevinMiddleIntegrator(
                300*unit.kelvin,
                1.0/unit.picosecond,
                0.002*unit.picoseconds
            )
            integrator.setConstraintTolerance(1e-5)

            # Create simulation
            simulation = app.Simulation(
                modeller.topology,
                system,
                integrator,
                self.platform,
                self.properties
            )

            # Set positions
            simulation.context.setPositions(modeller.positions)

            return simulation, modeller

        except Exception as e:
            logger.error(f"Error setting up simulation: {e}")
            raise

    def minimize_and_equilibrate(self, simulation: app.Simulation) -> Dict:
        """Minimize and equilibrate the system"""
        try:
            # Energy minimization with careful constraints
            logger.info("Starting energy minimization...")
            simulation.context.setPositions(simulation.context.getState(getPositions=True).getPositions())

            # Initial minimization with strong position restraints
            force = mm.HarmonicBondForce()
            force.setUsesPeriodicBoundaryConditions(True)

            # Add restraints to all heavy atoms
            positions = simulation.context.getState(getPositions=True).getPositions()
            reference_positions = []
            atom_indices = []
            for i, atom in enumerate(simulation.topology.atoms()):
                if atom.element.mass > 2.0 * unit.daltons:
                    reference_positions.append(positions[i])
                    atom_indices.append(i)

            # Add harmonic restraints
            k = 1000.0 * unit.kilojoules_per_mole/unit.nanometer**2
            for i, atom_idx in enumerate(atom_indices):
                force.addBond(atom_idx, atom_idx, 0.0 * unit.nanometer, k)

            # Add force to system and get index
            force_index = simulation.system.addForce(force)

            # Careful minimization in stages
            simulation.minimizeEnergy(maxIterations=100, tolerance=100.0)
            simulation.minimizeEnergy(maxIterations=100, tolerance=10.0)
            simulation.minimizeEnergy(maxIterations=100, tolerance=1.0)

            # Get minimized state
            state = simulation.context.getState(
                getPositions=True,
                getVelocities=True,
                getForces=True,
                getEnergy=True,
                enforcePeriodicBox=True
            )

            # Create new simulation for equilibration
            logger.info("Starting equilibration...")
            equil_integrator = mm.LangevinMiddleIntegrator(
                300*unit.kelvin,           # Temperature
                1/unit.picosecond,         # Friction coefficient
                0.002*unit.picoseconds     # Time step
            )
            equil_integrator.setConstraintTolerance(1e-5)

            equil_simulation = app.Simulation(
                simulation.topology,
                simulation.system,
                equil_integrator,
                self.platform,
                self.properties
            )

            # Transfer state to new simulation
            equil_simulation.context.setState(state)
            equil_simulation.context.setVelocitiesToTemperature(300*unit.kelvin)

            # Get force from equilibration context
            restraint_force = equil_simulation.system.getForce(force_index)

            # Staged equilibration with gradually relaxing restraints
            stages = [(1000.0, 100), (100.0, 100), (10.0, 100), (1.0, 100), (0.1, 100)]
            for k_restraint, steps in stages:
                # Update force constants for all restraints
                k = k_restraint * unit.kilojoules_per_mole/unit.nanometer**2
                for i in range(restraint_force.getNumBonds()):
                    particle1, particle2, length, _ = restraint_force.getBondParameters(i)
                    restraint_force.setBondParameters(i, particle1, particle2, length, k)
                restraint_force.updateParametersInContext(equil_simulation.context)
                equil_simulation.step(steps)

                # Check for numerical stability
                state = equil_simulation.context.getState(getEnergy=True, getPositions=True)
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)

                if np.any(np.isnan(energy)) or np.any(np.isnan(positions)):
                    logger.error("NaN detected in energy or positions during equilibration")
                    logger.debug(f"Energy: {energy}")
                    logger.debug(f"Positions shape: {positions.shape}")
                    raise ValueError("Numerical instability detected during equilibration")

            # Get system state
            state = equil_simulation.context.getState(
                getEnergy=True,
                getPositions=True,
                getVelocities=True
            )

            # Calculate temperature and extract numerical values
            kinetic_energy = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
            n_atoms = len(list(simulation.topology.atoms()))
            temperature = (2.0 * kinetic_energy) / (3.0 * n_atoms * unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoules_per_mole/unit.kelvin))

            return {
                'potential_energy': state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
                'kinetic_energy': kinetic_energy,
                'temperature': temperature
            }

        except Exception as e:
            logger.error(f"Error in minimization/equilibration: {e}")
            raise

    def run_dynamics(self, simulation: app.Simulation, steps: int = 5000) -> Dict:
        """Run molecular dynamics simulation"""
        try:
            # Run dynamics
            logger.info(f"Running dynamics for {steps} steps...")
            simulation.step(steps)

            # Get final state
            state = simulation.context.getState(
                getEnergy=True,
                getPositions=True,
                getVelocities=True
            )

            return {
                'potential_energy': state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
                'kinetic_energy': state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole),
                'temperature': state.getKineticEnergy() / (0.5 * unit.MOLAR_GAS_CONSTANT_R * len(list(simulation.topology.atoms()))),
                'positions': state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
            }

        except Exception as e:
            logger.error(f"Error running dynamics: {e}")
            raise

    def analyze_trajectory(self, positions: np.ndarray) -> Dict:
        """Analyze molecular dynamics trajectory"""
        try:
            # Calculate RMSD and other structural properties
            rmsd = np.sqrt(np.mean(np.sum((positions - positions[0])**2, axis=1)))

            return {
                'rmsd': rmsd,
                'average_structure': np.mean(positions, axis=0),
                'structure_variance': np.var(positions, axis=0)
            }

        except Exception as e:
            logger.error(f"Error analyzing trajectory: {e}")
            raise

class EnhancedSampling(MolecularDynamics):
    """Enhanced sampling methods for molecular dynamics simulations"""

    def __init__(self, device: Optional[str] = None):
        """Initialize enhanced sampling with device detection"""
        super().__init__(device)
        self.replicas = []
        self.umbrella_windows = []

    def setup_replica_exchange(self, pdb_file: str, n_replicas: int = 4,
                             temp_range: Tuple[float, float] = (300.0, 400.0)) -> List[app.Simulation]:
        """Setup replica exchange molecular dynamics (REMD)"""
        try:
            # Create replicas at different temperatures
            temperatures = np.linspace(temp_range[0], temp_range[1], n_replicas)
            self.replicas = []

            for temp in temperatures:
                # Setup base simulation
                sim, modeller = self.setup_simulation(pdb_file)

                # Modify integrator temperature
                integrator = mm.LangevinMiddleIntegrator(
                    temp*unit.kelvin,
                    1.0/unit.picosecond,
                    0.002*unit.picoseconds
                )
                integrator.setConstraintTolerance(1e-5)

                # Create new simulation with modified temperature
                replica = app.Simulation(
                    sim.topology,
                    sim.system,
                    integrator,
                    self.platform,
                    self.properties
                )

                # Set positions and velocities
                replica.context.setPositions(sim.context.getState(getPositions=True).getPositions())
                replica.context.setVelocitiesToTemperature(temp*unit.kelvin)

                self.replicas.append(replica)

            return self.replicas

        except Exception as e:
            logger.error(f"Error setting up replica exchange: {e}")
            raise

    def run_replica_exchange(self, exchange_steps: int = 1000,
                           dynamics_steps: int = 1000) -> List[Dict]:
        """Run replica exchange molecular dynamics"""
        try:
            if not self.replicas:
                raise ValueError("No replicas found. Run setup_replica_exchange first.")

            results = []

            # Run dynamics and exchanges
            for _ in range(exchange_steps):
                # Run dynamics for each replica
                for replica in self.replicas:
                    replica.step(dynamics_steps)

                # Attempt exchanges between neighboring replicas
                for i in range(len(self.replicas)-1):
                    success = self._attempt_exchange(self.replicas[i], self.replicas[i+1])
                    if success:
                        logger.info(f"Exchange successful between replicas {i} and {i+1}")

                # Collect state information
                states = []
                for replica in self.replicas:
                    state = replica.context.getState(getEnergy=True)
                    states.append({
                        'potential_energy': state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
                        'kinetic_energy': state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole),
                        'temperature': state.getKineticEnergy() / (0.5 * unit.MOLAR_GAS_CONSTANT_R *
                                     len(list(replica.topology.atoms())))
                    })
                results.append(states)

            return results

        except Exception as e:
            logger.error(f"Error running replica exchange: {e}")
            raise

    def _attempt_exchange(self, replica1: app.Simulation,
                         replica2: app.Simulation) -> bool:
        """Attempt a replica exchange between two simulations"""
        try:
            # Get states and energies
            state1 = replica1.context.getState(getEnergy=True, getPositions=True)
            state2 = replica2.context.getState(getEnergy=True, getPositions=True)

            energy1 = state1.getPotentialEnergy()
            energy2 = state2.getPotentialEnergy()

            # Get temperatures
            temp1 = replica1.integrator.getTemperature()
            temp2 = replica2.integrator.getTemperature()

            # Calculate Metropolis criterion
            delta = (1/temp1 - 1/temp2) * (energy1 - energy2)
            delta = delta.value_in_unit(unit.kilojoules_per_mole/unit.kelvin)

            # Accept or reject exchange
            if delta < 0 or np.random.random() < np.exp(-delta):
                # Exchange positions
                pos1 = state1.getPositions()
                pos2 = state2.getPositions()
                replica1.context.setPositions(pos2)
                replica2.context.setPositions(pos1)
                return True

            return False

        except Exception as e:
            logger.error(f"Error attempting replica exchange: {e}")
            raise
