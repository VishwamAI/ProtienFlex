"""Mock implementations for testing imports."""

class MockTensor:
    def __init__(self, data=None):
        self.data = data or []

class MockModel:
    def __init__(self):
        self.tensor = MockTensor()

# Export mock classes
__all__ = ['MockTensor', 'MockModel']
