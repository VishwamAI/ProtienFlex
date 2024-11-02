"""Lazy import utilities for handling dependencies."""
import os

class LazyLoader:
    """Lazy loader for handling dependencies."""

    def __init__(self, lib_name):
        self.lib_name = lib_name
        self._lib = None

    def __getattr__(self, name):
        if self._lib is None:
            # Check if we're in test mode
            if os.environ.get('PROTEINFLEX_TEST_MODE') == '1':
                if self.lib_name == 'numpy' or self.lib_name.startswith('numpy.'):
                    from tests.utils.mock_numpy import np
                    self._lib = np
                elif self.lib_name == 'openmm' or self.lib_name.startswith('openmm.'):
                    from tests.utils.mock_openmm import openmm, app, unit, mm
                    if self.lib_name == 'openmm':
                        self._lib = openmm
                    elif self.lib_name == 'openmm.app':
                        self._lib = app
                    elif self.lib_name == 'openmm.unit':
                        self._lib = unit
                    elif self.lib_name == 'openmm.mm':
                        self._lib = mm
                elif self.lib_name.startswith('transformers'):
                    from unittest.mock import Mock
                    mock_transformers = Mock()
                    mock_transformers.PreTrainedModel = Mock
                    mock_transformers.PretrainedConfig = Mock
                    mock_transformers.AutoModel = Mock()
                    mock_transformers.AutoTokenizer = Mock()
                    self._lib = mock_transformers
                else:
                    import importlib
                    self._lib = importlib.import_module(self.lib_name)
            else:
                import importlib
                self._lib = importlib.import_module(self.lib_name)
                # Special handling for numpy initialization
                if self.lib_name == 'numpy':
                    self._lib._import_array()
        return getattr(self._lib, name)

# Create lazy loaders for common dependencies
numpy = LazyLoader('numpy')
torch = LazyLoader('torch')

# PyTorch components
nn = LazyLoader('torch.nn')  # Neural network modules
F = LazyLoader('torch.nn.functional')  # Neural network functions

# OpenMM components
openmm = LazyLoader('openmm')
app = LazyLoader('openmm.app')  # OpenMM application layer
unit = LazyLoader('openmm.unit')  # OpenMM units
mm = LazyLoader('openmm.mm')  # OpenMM core functionality

# Bioinformatics tools
Bio = LazyLoader('Bio')
transformers = LazyLoader('transformers')
AutoModel = LazyLoader('transformers.AutoModel')  # Transformers auto model
AutoTokenizer = LazyLoader('transformers.AutoTokenizer')  # Transformers auto tokenizer

# Additional scientific computing
np = LazyLoader('numpy')  # Alias for numpy commonly used in code

# Initialize numpy array support
try:
    numpy._import_array()
    np._import_array()  # Also initialize the np alias
except AttributeError:
    pass  # Already initialized or will be initialized on first use

# Ensure OpenMM components are properly initialized
try:
    mm._lib = openmm._lib  # Share the same OpenMM instance
except AttributeError:
    pass  # OpenMM not yet initialized

# Ensure PyTorch components are properly initialized
try:
    nn._lib = torch._lib.nn  # Share the same torch instance
    F._lib = torch._lib.nn.functional  # Share the same torch instance
except AttributeError:
    pass  # PyTorch not yet initialized
