import sys
from unittest.mock import MagicMock

# Mock OpenMM
mock_openmm = MagicMock()
mock_openmm.app = MagicMock()
mock_openmm.unit = MagicMock()
mock_openmm.Platform = MagicMock()
mock_openmm.System = MagicMock()
sys.modules['openmm'] = mock_openmm
sys.modules['openmm.app'] = mock_openmm.app
sys.modules['openmm.unit'] = mock_openmm.unit

# Mock simtk
mock_simtk = MagicMock()
mock_simtk.openmm = mock_openmm
sys.modules['simtk'] = mock_simtk
sys.modules['simtk.openmm'] = mock_openmm

# Mock mdtraj
mock_mdtraj = MagicMock()
sys.modules['mdtraj'] = mock_mdtraj

# Mock RDKit
mock_rdkit = MagicMock()
sys.modules['rdkit'] = mock_rdkit
sys.modules['rdkit.Chem'] = MagicMock()
