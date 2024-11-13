"""
Structure Converter Module

This module handles conversion between different protein structure formats,
particularly focusing on converting between AlphaFold3's JAX-based structure
representations and OpenMM/MDTraj formats used in flexibility analysis.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import mdtraj as md
import openmm.app as app
import openmm.unit as unit
import jax.numpy as jnp
from Bio.PDB import Structure, Model, Chain, Residue, Atom
from Bio.PDB.Structure import Structure as BiopythonStructure

class StructureConverter:
    """Handles conversion between different protein structure formats."""

    def __init__(self):
        """Initialize the converter with necessary mappings."""
        # Standard residue names mapping
        self.residue_name_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        self.reverse_residue_map = {v: k for k, v in self.residue_name_map.items()}

    def alphafold_to_openmm(self,
                           positions: jnp.ndarray,
                           sequence: str,
                           confidence: jnp.ndarray) -> Tuple[app.Modeller, np.ndarray]:
        """Convert AlphaFold3 output to OpenMM format.

        Args:
            positions: Atom positions from AlphaFold3 [n_atoms, 3]
            sequence: Amino acid sequence
            confidence: Per-residue confidence scores

        Returns:
            OpenMM Modeller object and confidence scores
        """
        # Create PDB structure from positions
        structure = self._create_pdb_structure(positions, sequence)

        # Convert to OpenMM
        pdb = app.PDBFile(structure)
        modeller = app.Modeller(pdb.topology, pdb.positions)

        return modeller, confidence

    def openmm_to_mdtraj(self,
                        modeller: app.Modeller) -> md.Trajectory:
        """Convert OpenMM Modeller to MDTraj trajectory.

        Args:
            modeller: OpenMM Modeller object

        Returns:
            MDTraj trajectory object
        """
        # Convert positions to nanometers
        positions = modeller.positions.value_in_unit(unit.nanometers)

        # Create MDTraj topology
        top = md.Topology.from_openmm(modeller.topology)

        # Create trajectory
        traj = md.Trajectory(
            xyz=positions.reshape(1, -1, 3),
            topology=top
        )

        return traj

    def mdtraj_to_alphafold(self,
                           trajectory: md.Trajectory) -> Tuple[jnp.ndarray, str]:
        """Convert MDTraj trajectory to AlphaFold3 format.

        Args:
            trajectory: MDTraj trajectory

        Returns:
            Tuple of (positions array, sequence string)
        """
        # Extract positions (first frame)
        positions = jnp.array(trajectory.xyz[0])

        # Extract sequence
        sequence = ''.join(
            self.residue_name_map.get(r.name, 'X')
            for r in trajectory.topology.residues
        )

        return positions, sequence

    def _create_pdb_structure(self,
                            positions: np.ndarray,
                            sequence: str) -> BiopythonStructure:
        """Create Biopython Structure from positions and sequence.

        Args:
            positions: Atom positions [n_atoms, 3]
            sequence: Amino acid sequence

        Returns:
            Biopython Structure object
        """
        structure = Structure.Structure('0')
        model = Model.Model(0)
        chain = Chain.Chain('A')

        atom_index = 0
        for res_idx, res_code in enumerate(sequence):
            res_name = self.reverse_residue_map[res_code]
            residue = Residue.Residue((' ', res_idx, ' '), res_name, '')

            # Add backbone atoms
            for atom_name in ['N', 'CA', 'C', 'O']:
                coord = positions[atom_index]
                atom = Atom.Atom(atom_name,
                               coord,
                               20.0,  # B-factor
                               1.0,   # Occupancy
                               ' ',   # Altloc
                               atom_name,
                               atom_index,
                               'C')   # Element
                residue.add(atom)
                atom_index += 1

            chain.add(residue)

        model.add(chain)
        structure.add(model)

        return structure

    def add_confidence_to_structure(self,
                                  structure: Union[app.Modeller, md.Trajectory],
                                  confidence: np.ndarray) -> None:
        """Add confidence scores to structure as B-factors.

        Args:
            structure: Structure object (OpenMM or MDTraj)
            confidence: Per-residue confidence scores
        """
        if isinstance(structure, app.Modeller):
            # Add to OpenMM structure
            for atom in structure.topology.atoms():
                res_idx = atom.residue.index
                atom.bfactor = float(confidence[res_idx])
        elif isinstance(structure, md.Trajectory):
            # Add to MDTraj structure
            for atom in structure.topology.atoms:
                res_idx = atom.residue.index
                atom.bfactor = float(confidence[res_idx])

    def get_atom_positions(self,
                         structure: Union[app.Modeller, md.Trajectory]) -> np.ndarray:
        """Extract atom positions from structure.

        Args:
            structure: Structure object (OpenMM or MDTraj)

        Returns:
            Numpy array of atom positions [n_atoms, 3]
        """
        if isinstance(structure, app.Modeller):
            return structure.positions.value_in_unit(unit.nanometers)
        elif isinstance(structure, md.Trajectory):
            return structure.xyz[0]
        else:
            raise ValueError(f"Unsupported structure type: {type(structure)}")

    def get_sequence(self,
                    structure: Union[app.Modeller, md.Trajectory]) -> str:
        """Extract amino acid sequence from structure.

        Args:
            structure: Structure object (OpenMM or MDTraj)

        Returns:
            Amino acid sequence string
        """
        if isinstance(structure, app.Modeller):
            topology = structure.topology
        elif isinstance(structure, md.Trajectory):
            topology = structure.topology
        else:
            raise ValueError(f"Unsupported structure type: {type(structure)}")

        sequence = ''
        for residue in topology.residues():
            res_name = residue.name if hasattr(residue, 'name') else residue.resname
            sequence += self.residue_name_map.get(res_name, 'X')

        return sequence
