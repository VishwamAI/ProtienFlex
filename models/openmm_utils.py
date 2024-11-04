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

import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OpenMMSimulator:
    """Class for handling OpenMM molecular dynamics simulations"""
    def __init__(self):
        self.system = None
        self.pdb = None
        self.positions = None

    def setup_system(self, pdb_string):
        """Set up OpenMM system for molecular dynamics"""
        try:
            # Create a temporary PDB file
            with open('temp.pdb', 'w') as f:
                f.write(pdb_string)

            # Load the PDB file
            self.pdb = app.PDBFile('temp.pdb')

            # Create force field
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')

            # Create system
            self.system = forcefield.createSystem(
                self.pdb.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                rigidWater=True
            )

            # Add harmonic restraints
            force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            force.addGlobalParameter("k", 5.0*unit.kilocalories_per_mole/unit.angstroms**2)
            force.addPerParticleParameter("x0")
            force.addPerParticleParameter("y0")
            force.addPerParticleParameter("z0")

            for atom in self.pdb.positions:
                force.addParticle(self.system.getNumParticles()-1, [atom.x, atom.y, atom.z])

            self.system.addForce(force)
            return True
        except Exception as e:
            logger.error(f"Error setting up OpenMM system: {e}")
            return False

    def minimize_structure(self):
        """Minimize the structure using OpenMM"""
        try:
            if self.system is None or self.pdb is None:
                return None

            integrator = mm.LangevinIntegrator(
                300*unit.kelvin,
                1/unit.picosecond,
                0.002*unit.picoseconds
            )

            platform = mm.Platform.getPlatformByName('CPU')
            simulation = app.Simulation(self.pdb.topology, self.system, integrator, platform)
            simulation.context.setPositions(self.pdb.positions)

            # Minimize
            simulation.minimizeEnergy(maxIterations=100)

            # Get minimized positions
            state = simulation.context.getState(getPositions=True)
            self.positions = state.getPositions()
            return self.positions
        except Exception as e:
            logger.error(f"Error minimizing structure: {e}")
            return None

    def calculate_contact_map(self, cutoff=8.0):
        """Calculate contact map from atomic positions"""
        try:
            if self.positions is None or self.pdb is None:
                return None

            positions = self.positions.value_in_unit(unit.angstrom)
            n_residues = len(list(self.pdb.topology.residues()))
            contact_map = np.zeros((n_residues, n_residues))

            for i, res1 in enumerate(self.pdb.topology.residues()):
                for j, res2 in enumerate(self.pdb.topology.residues()):
                    if i < j:
                        min_dist = float('inf')
                        for atom1 in res1.atoms():
                            for atom2 in res2.atoms():
                                pos1 = positions[atom1.index]
                                pos2 = positions[atom2.index]
                                dist = np.sqrt(np.sum((pos1 - pos2)**2))
                                min_dist = min(min_dist, dist)

                        if min_dist <= cutoff:
                            contact_map[i,j] = contact_map[j,i] = 1

            return contact_map
        except Exception as e:
            logger.error(f"Error calculating contact map: {e}")
            return None

    def calculate_structure_confidence(self, contact_map=None):
        """Calculate confidence score based on contact density"""
        try:
            if contact_map is None:
                contact_map = self.calculate_contact_map()
            if contact_map is None:
                return 50.0

            sequence_length = len(list(self.pdb.topology.residues()))
            contact_density = np.sum(contact_map) / (sequence_length * sequence_length)
            # Scale to 0-100 range with sigmoid
            confidence = 100 / (1 + np.exp(-10 * (contact_density - 0.2)))
            return float(confidence)
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 50.0
