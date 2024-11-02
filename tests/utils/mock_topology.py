"""Mock topology objects for OpenMM testing."""
import numpy as np
from unittest.mock import Mock
from .mock_base import MockQuantity

class MockAtom:
    """Mock Atom class for OpenMM testing."""
    def __init__(self, index, element_mass=12.0):
        """Initialize atom with index and mass."""
        self.index = index
        self.element = Mock(mass=MockQuantity(element_mass, "daltons"))
        self.name = f"ATOM_{index}"
        self.residue = None

class MockResidue:
    """Mock Residue class for OpenMM testing."""
    def __init__(self, index, name="ALA"):
        """Initialize residue with index and name."""
        self.index = index
        self.name = name
        self._atoms = []
        self.chain = None

    def addAtom(self, atom):
        """Add atom to residue."""
        self._atoms.append(atom)
        atom.residue = self

    def atoms(self):
        """Return iterator over atoms."""
        return iter(self._atoms)

class MockChain:
    """Mock Chain class for OpenMM testing."""
    def __init__(self, index, name="A"):
        """Initialize chain with index and name."""
        self.index = index
        self.name = name
        self._residues = []

    def addResidue(self, residue):
        """Add residue to chain."""
        self._residues.append(residue)
        residue.chain = self

    def residues(self):
        """Return iterator over residues."""
        return iter(self._residues)

class MockTopology:
    """Mock Topology class for OpenMM testing."""
    def __init__(self, n_atoms=100):
        """Initialize topology with given number of atoms."""
        self._chains = []
        self._residues = []
        self._atoms = []

        # Create a simple topology structure
        chain = MockChain(0)
        self._chains.append(chain)

        # Create residues (3 atoms per residue)
        for i in range(0, n_atoms, 3):
            residue = MockResidue(i // 3)
            chain.addResidue(residue)
            self._residues.append(residue)

            # Create atoms for this residue
            for j in range(min(3, n_atoms - i)):
                atom = MockAtom(i + j)
                residue.addAtom(atom)
                self._atoms.append(atom)

    def atoms(self):
        """Return iterator over atoms."""
        return iter(self._atoms)

    def residues(self):
        """Return iterator over residues."""
        return iter(self._residues)

    def chains(self):
        """Return iterator over chains."""
        return iter(self._chains)

    def getNumAtoms(self):
        """Get number of atoms."""
        return len(self._atoms)

    def getNumResidues(self):
        """Get number of residues."""
        return len(self._residues)

    def getNumChains(self):
        """Get number of chains."""
        return len(self._chains)
