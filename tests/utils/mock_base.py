"""Base mock objects for OpenMM testing."""
import numpy as np
from unittest.mock import Mock

class MockQuantity:
    """Mock Quantity class for unit handling."""
    def __init__(self, value, unit_str="dimensionless"):
        """Initialize quantity with value and unit."""
        self.value = value
        self.unit_str = unit_str
        self._array = None
        self._unit_conversions = {
            'nanometers': 1e-9,
            'angstroms': 1e-10,
            'picoseconds': 1e-12,
            'femtoseconds': 1e-15,
            'kelvin': 1.0,
            'kilojoules_per_mole': 1.0,
            'kilojoules/mole': 1.0,
            'nanometer': 1e-9,
            'angstrom': 1e-10,
            'picosecond': 1e-12,
            'femtosecond': 1e-15,
            'daltons': 1.66053907e-27,
            'nanometer**2': 1e-18,
            'nanometer^2': 1e-18,
            'molar': 1.0,
            'molars': 1.0,
            'atmospheres': 1.0
        }
        # Support array operations
        if isinstance(self.value, np.ndarray):
            self._array = self.value.astype(float)
        elif isinstance(self.value, (list, tuple)):
            self._array = np.array(self.value, dtype=float)
        elif hasattr(self.value, 'value_in_unit'):
            self.value = self.value.value_in_unit(unit_str)
            if isinstance(self.value, np.ndarray):
                self._array = self.value.astype(float)
            else:
                self._array = np.array([float(self.value)])
        else:
            try:
                self._array = np.array([float(self.value)])
            except (TypeError, ValueError):
                self._array = np.array([1.0])  # Default value for non-numeric types

        # Add numpy array attributes
        self.shape = self._array.shape
        self.ndim = self._array.ndim
        self.size = self._array.size
        self.dtype = self._array.dtype

    def __hash__(self):
        """Make hashable for dictionary operations."""
        return hash((float(self._array.item(0)), self.unit_str))

    def __getitem__(self, key):
        """Support array indexing."""
        return MockQuantity(self._array[key], self.unit_str)

    def __setitem__(self, key, value):
        """Support array item assignment."""
        if isinstance(value, MockQuantity):
            self._array[key] = value._array
        else:
            try:
                self._array[key] = float(value)
            except (TypeError, ValueError):
                self._array[key] = 1.0

    def value_in_unit(self, unit_str):
        """Get value in specified unit."""
        if unit_str in self._unit_conversions:
            return self._array * self._unit_conversions[unit_str]
        return self._array

    def __mul__(self, other):
        """Multiply with another quantity or scalar."""
        if isinstance(other, MockQuantity):
            new_value = self._array * other._array
            return MockQuantity(new_value, f"{self.unit_str}*{other.unit_str}")
        elif isinstance(other, np.ndarray):
            new_value = self._array * other
            return MockQuantity(new_value, self.unit_str)
        elif isinstance(other, Mock):
            return MockQuantity(self._array * 1.0, self.unit_str)
        try:
            return MockQuantity(self._array * float(other), self.unit_str)
        except (TypeError, ValueError):
            return MockQuantity(self._array * 1.0, self.unit_str)

    def __rmul__(self, other):
        """Right multiply."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide by another quantity or scalar."""
        if isinstance(other, MockQuantity):
            new_value = self._array / other._array
            return MockQuantity(new_value, f"{self.unit_str}/{other.unit_str}")
        elif isinstance(other, np.ndarray):
            new_value = self._array / other
            return MockQuantity(new_value, self.unit_str)
        elif isinstance(other, Mock):
            return MockQuantity(self._array / 1.0, self.unit_str)
        try:
            return MockQuantity(self._array / float(other), self.unit_str)
        except (TypeError, ValueError):
            return MockQuantity(self._array / 1.0, self.unit_str)

    def __rtruediv__(self, other):
        """Right divide."""
        if isinstance(other, MockQuantity):
            new_value = other._array / self._array
            return MockQuantity(new_value, f"{other.unit_str}/{self.unit_str}")
        elif isinstance(other, np.ndarray):
            new_value = other / self._array
            return MockQuantity(new_value, f"1/{self.unit_str}")
        elif isinstance(other, Mock):
            return MockQuantity(1.0 / self._array, f"1/{self.unit_str}")
        try:
            return MockQuantity(float(other) / self._array, f"1/{self.unit_str}")
        except (TypeError, ValueError):
            return MockQuantity(1.0 / self._array, f"1/{self.unit_str}")

    def __pow__(self, power):
        """Power operation."""
        try:
            new_value = self._array ** float(power)
            return MockQuantity(new_value, f"{self.unit_str}^{power}")
        except (TypeError, ValueError):
            return MockQuantity(self._array ** 1.0, self.unit_str)

    def __float__(self):
        """Convert to float."""
        return float(self._array.item())

    def __gt__(self, other):
        """Greater than comparison."""
        if isinstance(other, MockQuantity):
            return np.all(self._array > other._array)
        try:
            return np.all(self._array > float(other))
        except (TypeError, ValueError):
            return False

    def __lt__(self, other):
        """Less than comparison."""
        if isinstance(other, MockQuantity):
            return np.all(self._array < other._array)
        try:
            return np.all(self._array < float(other))
        except (TypeError, ValueError):
            return False

    def __eq__(self, other):
        """Equal comparison."""
        if isinstance(other, MockQuantity):
            return np.all(self._array == other._array)
        try:
            return np.all(self._array == float(other))
        except (TypeError, ValueError):
            return False

    def __ge__(self, other):
        """Greater than or equal comparison."""
        if isinstance(other, MockQuantity):
            return np.all(self._array >= other._array)
        try:
            return np.all(self._array >= float(other))
        except (TypeError, ValueError):
            return False

    def __le__(self, other):
        """Less than or equal comparison."""
        if isinstance(other, MockQuantity):
            return np.all(self._array <= other._array)
        try:
            return np.all(self._array <= float(other))
        except (TypeError, ValueError):
            return False

    def __add__(self, other):
        """Add operation."""
        if isinstance(other, MockQuantity):
            if isinstance(other._array, np.ndarray) and isinstance(self._array, np.ndarray):
                return MockQuantity(self._array + other._array, self.unit_str)
            return MockQuantity(np.array(self._array) + np.array(other._array), self.unit_str)
        elif isinstance(other, (np.ndarray, list)):
            return MockQuantity(np.array(self._array) + np.array(other), self.unit_str)
        try:
            return MockQuantity(self._array + float(other), self.unit_str)
        except (TypeError, ValueError):
            return MockQuantity(self._array + 0.0, self.unit_str)

    def __sub__(self, other):
        """Subtract operation."""
        if isinstance(other, MockQuantity):
            if isinstance(other._array, np.ndarray) and isinstance(self._array, np.ndarray):
                return MockQuantity(self._array - other._array, self.unit_str)
            return MockQuantity(np.array(self._array) - np.array(other._array), self.unit_str)
        elif isinstance(other, (np.ndarray, list)):
            return MockQuantity(np.array(self._array) - np.array(other), self.unit_str)
        try:
            return MockQuantity(self._array - float(other), self.unit_str)
        except (TypeError, ValueError):
            return MockQuantity(self._array - 0.0, self.unit_str)

class MockTopology:
    """Mock Topology class for OpenMM testing."""
    def __init__(self, n_atoms=100):
        """Initialize topology with given number of atoms."""
        self._n_atoms = n_atoms
        self._atoms = [Mock(element=Mock(mass=12.0)) for _ in range(n_atoms)]
        self._residues = [Mock(atoms=self._atoms[i:i+3]) for i in range(0, n_atoms, 3)]
        self._chains = [Mock(residues=self._residues)]

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
        return self._n_atoms

    def __radd__(self, other):
        """Right add operation."""
        return self.__add__(other)

    def __rsub__(self, other):
        """Right subtract operation."""
        if isinstance(other, MockQuantity):
            return MockQuantity(other._array - self._array, self.unit_str)
        elif isinstance(other, np.ndarray):
            return MockQuantity(other - self._array, self.unit_str)
        return MockQuantity(float(other) - self._array, self.unit_str)
