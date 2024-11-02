"""Initialize test environment with system-wide mocks."""
import sys
from unittest.mock import MagicMock

# Mock OpenMM and related modules
mock_openmm = MagicMock()
mock_openmm.app = MagicMock()
mock_openmm.unit = MagicMock()
mock_openmm.Platform = MagicMock()
mock_openmm.System = MagicMock()
mock_openmm.Force = MagicMock()
mock_openmm.Context = MagicMock()
mock_openmm.State = MagicMock()
mock_openmm.Vec3 = MagicMock()

sys.modules['openmm'] = mock_openmm
sys.modules['openmm.app'] = mock_openmm.app
sys.modules['openmm.unit'] = mock_openmm.unit

# Mock simtk.openmm
mock_simtk = MagicMock()
mock_simtk.openmm = mock_openmm
sys.modules['simtk'] = mock_simtk
sys.modules['simtk.openmm'] = mock_openmm

# Mock mdtraj
mock_mdtraj = MagicMock()
sys.modules['mdtraj'] = mock_mdtraj

# Mock numpy array import
import numpy as np
np.import_array = MagicMock()

# Mock transformers
mock_transformers = MagicMock()
sys.modules['transformers'] = mock_transformers

# Mock torch cuda
import torch
if not hasattr(torch, 'cuda'):
    torch.cuda = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)
