import numpy as np
from unittest.mock import Mock

class MockQuantity:
    """Mock OpenMM Quantity with proper numpy array handling."""
    def __init__(self, value, unit=None):
        self.value = np.asarray(value) if not isinstance(value, MockQuantity) else value.value
        self.unit = unit
        self._mock = Mock()

    def value_in_unit(self, unit=None):
        """Get value in specified unit."""
        if unit is None or self.unit is None:
            return np.asarray(self.value)
        # In test mode, just return the value since we don't do real unit conversion
        return np.asarray(self.value)

    def __array__(self):
        """Convert to numpy array."""
        return np.asarray(self.value)

    def __mul__(self, other):
        """Multiply with proper unit handling."""
        if isinstance(other, MockQuantity):
            return MockQuantity(self.value * other.value, None)
        return MockQuantity(self.value * np.asarray(other), self.unit)

    def __rmul__(self, other):
        """Right multiply with proper unit handling."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide with proper unit handling."""
        if isinstance(other, MockQuantity):
            return MockQuantity(self.value / other.value, None)
        return MockQuantity(self.value / np.asarray(other), self.unit)

    def __rtruediv__(self, other):
        """Right divide with proper unit handling."""
        if isinstance(other, MockQuantity):
            return MockQuantity(other.value / self.value, None)
        return MockQuantity(np.asarray(other) / self.value, self.unit)

    def __add__(self, other):
        """Add with proper unit handling."""
        if isinstance(other, MockQuantity):
            return MockQuantity(self.value + other.value, self.unit)
        return MockQuantity(self.value + np.asarray(other), self.unit)

    def __radd__(self, other):
        """Right add with proper unit handling."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract with proper unit handling."""
        if isinstance(other, MockQuantity):
            return MockQuantity(self.value - other.value, self.unit)
        return MockQuantity(self.value - np.asarray(other), self.unit)

    def __rsub__(self, other):
        """Right subtract with proper unit handling."""
        if isinstance(other, MockQuantity):
            return MockQuantity(other.value - self.value, self.unit)
        return MockQuantity(np.asarray(other) - self.value, self.unit)

    def __pow__(self, power):
        """Power operation with proper unit handling."""
        return MockQuantity(np.power(self.value, power), self.unit)

    def __len__(self):
        """Length of underlying array."""
        return len(self.value)

    def __getitem__(self, key):
        """Array indexing."""
        return MockQuantity(self.value[key], self.unit)

    def __iter__(self):
        """Iterator support."""
        for v in np.nditer(self.value):
            yield MockQuantity(v.item(), self.unit)

    def __eq__(self, other):
        """Equal comparison."""
        if isinstance(other, MockQuantity):
            return np.array_equal(self.value, other.value)
        return np.array_equal(self.value, np.asarray(other))

    def __ne__(self, other):
        """Not equal comparison."""
        return not self.__eq__(other)

    def __lt__(self, other):
        """Less than comparison."""
        if isinstance(other, MockQuantity):
            return np.all(self.value < other.value)
        return np.all(self.value < np.asarray(other))

    def __le__(self, other):
        """Less than or equal comparison."""
        if isinstance(other, MockQuantity):
            return np.all(self.value <= other.value)
        return np.all(self.value <= np.asarray(other))

    def __gt__(self, other):
        """Greater than comparison."""
        if isinstance(other, MockQuantity):
            return np.all(self.value > other.value)
        return np.all(self.value > np.asarray(other))

    def __ge__(self, other):
        """Greater than or equal comparison."""
        if isinstance(other, MockQuantity):
            return np.all(self.value >= other.value)
        return np.all(self.value >= np.asarray(other))

    def __repr__(self):
        """String representation."""
        return f"MockQuantity({self.value}, {self.unit})"
