"""Mock force objects for OpenMM testing."""
import numpy as np
from unittest.mock import Mock
from .mock_base import MockQuantity

class MockHarmonicBondForce:
    """Mock HarmonicBondForce class."""
    def __init__(self):
        """Initialize force."""
        self._bonds = []
        self._periodic = False
        self._num_bonds = 0  # Explicit integer counter

    def addBond(self, atom1, atom2, length, k):
        """Add a bond to the force."""
        if isinstance(length, Mock):
            length = MockQuantity(0.0, "nanometers")
        if isinstance(k, Mock):
            k = MockQuantity(1000.0, "kilojoules_per_mole/nanometer^2")
        self._bonds.append((int(atom1), int(atom2), length, k))
        self._num_bonds = len(self._bonds)  # Update integer counter
        return self._num_bonds - 1

    def setUsesPeriodicBoundaryConditions(self, periodic):
        """Set periodic boundary conditions."""
        self._periodic = bool(periodic)

    def getNumBonds(self):
        """Get number of bonds."""
        return int(self._num_bonds)  # Ensure integer return

    def getBondParameters(self, index):
        """Get bond parameters."""
        if 0 <= index < self._num_bonds:
            return self._bonds[index]
        raise IndexError(f"Bond index {index} out of range")

    def setBondParameters(self, index, atom1, atom2, length, k):
        """Set bond parameters."""
        if 0 <= index < self._num_bonds:
            if isinstance(length, Mock):
                length = MockQuantity(0.0, "nanometers")
            if isinstance(k, Mock):
                k = MockQuantity(1000.0, "kilojoules_per_mole/nanometer^2")
            self._bonds[index] = (int(atom1), int(atom2), length, k)
        else:
            raise IndexError(f"Bond index {index} out of range")

class MockCustomExternalForce:
    """Mock CustomExternalForce class."""
    def __init__(self, energy_function):
        """Initialize force."""
        self._energy_function = energy_function
        self._global_parameters = {}
        self._particle_parameters = []
        self._particles = []

    def addGlobalParameter(self, name, default_value):
        """Add a global parameter."""
        self._global_parameters[name] = default_value

    def addPerParticleParameter(self, name):
        """Add a per-particle parameter."""
        self._particle_parameters.append(name)

    def addParticle(self, particle_index, parameters):
        """Add a particle."""
        if len(parameters) != len(self._particle_parameters):
            raise ValueError("Wrong number of parameters")
        self._particles.append((particle_index, parameters))
        return len(self._particles) - 1

    def getNumParticles(self):
        """Get number of particles."""
        return len(self._particles)

    def getParticleParameters(self, index):
        """Get particle parameters."""
        if 0 <= index < len(self._particles):
            return self._particles[index]
        raise IndexError(f"Particle index {index} out of range")

    def setParticleParameters(self, index, particle_index, parameters):
        """Set particle parameters."""
        if 0 <= index < len(self._particles):
            self._particles[index] = (particle_index, parameters)
        else:
            raise IndexError(f"Particle index {index} out of range")

class MockSystem:
    """Mock System class."""
    def __init__(self):
        """Initialize system."""
        self._forces = []
        self._num_particles = 100  # Default number of particles
        self._periodic = True  # Default to using periodic boundary conditions

    def addForce(self, force):
        """Add a force to the system."""
        self._forces.append(force)
        return len(self._forces) - 1

    def getForce(self, index):
        """Get force by index."""
        if 0 <= index < len(self._forces):
            return self._forces[index]
        raise IndexError(f"Force index {index} out of range")

    def getNumForces(self):
        """Get number of forces."""
        return len(self._forces)

    def getNumParticles(self):
        """Get number of particles."""
        return self._num_particles

    def usesPeriodicBoundaryConditions(self):
        """Check if system uses periodic boundary conditions."""
        return self._periodic

    def setDefaultPeriodicBoxVectors(self, *args):
        """Set default periodic box vectors."""
        pass  # Not needed for testing
