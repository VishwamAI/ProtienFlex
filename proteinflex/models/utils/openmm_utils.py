from proteinflex.models.utils.lazy_imports import numpy as np, torch, openmm
from proteinflex.models.utils.lazy_imports import app, mm, unit
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

import logging
from Bio.PDB import PDBIO, Structure, Model, Chain
from io import StringIO

logger = logging.getLogger(__name__)

def setup_openmm_system(pdb_string):
    """Set up OpenMM system for molecular dynamics"""
    try:
        # Create a temporary PDB file
        from io import StringIO
        import tempfile
        import os

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp.write(pdb_string)
            tmp_path = tmp.name

        try:
            # Load PDB file
            pdb = app.PDBFile(tmp_path)

            # Create force field
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')

            # Create system
            system = forcefield.createSystem(
                pdb.topology,
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

            for i, pos in enumerate(pdb.positions):
                force.addParticle(i, [pos.x, pos.y, pos.z])

            system.addForce(force)
            return system, pdb

        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    except Exception as e:
        logging.error(f"Error setting up OpenMM system: {str(e)}")
        return None, None

def minimize_structure(system, pdb):
    """Minimize the structure using OpenMM"""
    try:
        if system is None or pdb is None:
            return None

        integrator = mm.LangevinIntegrator(
            300*unit.kelvin,
            1/unit.picosecond,
            0.002*unit.picoseconds
        )

        platform = mm.Platform.getPlatformByName('CPU')
        simulation = app.Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)

        # Minimize
        simulation.minimizeEnergy(maxIterations=100)

        # Get minimized positions
        state = simulation.context.getState(getPositions=True)
        return state.getPositions()
    except Exception as e:
        logger.error(f"Error minimizing structure: {e}")
        return None

def calculate_contact_map(positions, topology, cutoff=8.0):
    """Calculate contact map from atomic positions"""
    try:
        # Convert positions to numpy array in angstroms
        if hasattr(positions[0], 'value_in_unit'):
            positions = np.array([pos.value_in_unit(unit.angstrom) for pos in positions])
        else:
            positions = np.array(positions)

        # Get residue list and create mapping from atom index to residue index
        residues = list(topology.residues())
        n_residues = len(residues)
        atom_to_residue = {}
        for i, residue in enumerate(residues):
            for atom in residue.atoms():
                atom_to_residue[atom.index] = i

        contact_map = np.zeros((n_residues, n_residues))

        # Calculate distances between all atoms
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                if dist <= cutoff:
                    res_i = atom_to_residue[i]
                    res_j = atom_to_residue[j]
                    contact_map[res_i, res_j] = contact_map[res_j, res_i] = 1

        return contact_map
    except Exception as e:
        logger.error(f"Error calculating contact map: {e}")
        return None

def calculate_structure_confidence(contact_map, sequence_length):
    """Calculate confidence score based on contact density"""
    try:
        if contact_map is None:
            return 50.0

        # Calculate contact density excluding diagonal
        total_contacts = np.sum(contact_map) - np.trace(contact_map)  # Exclude diagonal
        max_contacts = sequence_length * (sequence_length - 1)  # Maximum possible non-diagonal contacts
        contact_density = total_contacts / max_contacts if max_contacts > 0 else 0

        # Scale to 50-100 range using linear scaling
        # No contacts -> 50%, all possible contacts -> 100%
        confidence = 50.0 + (contact_density * 50.0)
        return min(100.0, max(50.0, confidence))  # Ensure between 50 and 100

    except Exception as e:
        logger.error(f"Error calculating confidence: {e}")
        return 50.0

def setup_simulation(protein_structure, ligand=None, force_field_name='amber14-all.xml', temperature=300.0):
    """Set up OpenMM simulation with protein and optional ligand"""
    try:
        # Convert Bio.PDB structure to PDB string
        pdb_io = StringIO()
        io = PDBIO()
        io.set_structure(protein_structure)
        io.save(pdb_io)
        pdb_string = pdb_io.getvalue()
        pdb_io.close()

        # Set up system using existing function
        system, pdb = setup_openmm_system(pdb_string)
        if system is None or pdb is None:
            raise ValueError("Failed to set up OpenMM system")

        # Create integrator
        integrator = mm.LangevinIntegrator(
            temperature * unit.kelvin,
            1/unit.picosecond,
            0.002*unit.picoseconds
        )

        # Create simulation
        platform = mm.Platform.getPlatformByName('CPU')
        simulation = app.Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)

        return simulation

    except Exception as e:
        logger.error(f"Error setting up simulation: {str(e)}")
        raise ValueError(f"Failed to set up simulation: {str(e)}")

def minimize_and_equilibrate(simulation):
    """Minimize and equilibrate the system"""
    try:
        # Minimize
        simulation.minimizeEnergy(maxIterations=100)

        # Short equilibration
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
        simulation.step(1000)  # 2 ps equilibration

        # Get state
        state = simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getForces=True,
            getEnergy=True
        )
        return state

    except Exception as e:
        logger.error(f"Error in minimization and equilibration: {str(e)}")
        raise ValueError(f"Failed minimization and equilibration: {str(e)}")
