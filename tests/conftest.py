import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_mock_result(mocker, result_dict):
    """Create a mock result that supports both attribute and dictionary access."""
    mock = mocker.MagicMock()
    # Set up dictionary-style access
    mock.__getitem__ = mocker.MagicMock()
    mock.__getitem__.return_value = mocker.MagicMock()
    mock.__getitem__.side_effect = result_dict.__getitem__

    # Set up attribute access
    for key, value in result_dict.items():
        setattr(mock, key, value)

    # Set up additional dictionary methods
    mock.get = mocker.MagicMock(side_effect=result_dict.get)
    mock.keys = mocker.MagicMock(return_value=result_dict.keys())
    return mock

def create_mock_method(mocker, result_dict):
    """Create a mock method that returns a properly structured mock result."""
    mock_result = create_mock_result(mocker, result_dict)
    mock = mocker.MagicMock()
    mock.side_effect = lambda *args, **kwargs: mock_result
    mock.return_value = mock_result
    return mock

@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    """Mock external dependencies."""
    class MockTensor:
        def __init__(self, shape=(1, 100)):
            self.shape = shape if isinstance(shape, tuple) else (shape,)
            self.data = np.zeros(self.shape)

        def mean(self):
            return np.mean(self.data)

        def numpy(self):
            return self.data

        def dim(self):
            return len(self.shape)

        def __getitem__(self, idx):
            return self.data[idx]

        def get(self, key, default=None):
            return getattr(self, key, default)

        def item(self):
            return self.data.item()

        def to(self, device):
            return self

        def max(self):
            return np.max(self.data)

        def sigmoid(self):
            return 1 / (1 + np.exp(-self.data))

    # Mock torch
    mock_torch = mocker.Mock()
    mock_torch.Tensor = MockTensor
    mock_torch.device = mocker.Mock()
    mock_torch.cuda.is_available = mocker.Mock(return_value=True)
    mock_torch.no_grad = mocker.Mock()
    mock_torch.no_grad.return_value.__enter__ = mocker.Mock()
    mock_torch.no_grad.return_value.__exit__ = mocker.Mock()

    # Mock transformers
    mock_transformers = mocker.Mock()
    mock_transformers.AutoModel.from_pretrained = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'model',
        'model': mocker.Mock()
    })
    mock_transformers.AutoTokenizer.from_pretrained = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'tokenizer',
        'tokenizer': mocker.Mock()
    })
    mock_pipeline = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'pipeline_output',
        'text': 'Mock pipeline output',
        'answer': 'Mock answer',
        'confidence': 0.9
    })
    mock_transformers.pipeline.return_value = mock_pipeline

    # Mock openmm
    mock_openmm = mocker.Mock()
    mock_openmm.app = mocker.Mock()
    mock_openmm.app.PDBFile.return_value = create_mock_result(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'pdb_structure',
        'topology': mocker.Mock(),
        'positions': mocker.Mock()
    })
    mock_openmm.app.ForceField.return_value = create_mock_result(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'force_field',
        'createSystem': create_mock_method(mocker, {
            'start': 0,
            'end': 100,
            'score': 0.95,
            'type': 'system',
            'system': mocker.Mock()
        })
    })
    mock_openmm.unit = mocker.Mock()
    mock_openmm.unit.kelvin = 300.0
    mock_openmm.unit.picosecond = 1.0
    mock_openmm.unit.femtosecond = 0.001
    mock_openmm.unit.nanometer = 1.0
    mock_openmm.Platform.getPlatformByName.return_value = create_mock_result(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'platform',
        'name': 'CPU'
    })
    mock_openmm.LangevinMiddleIntegrator.return_value = create_mock_result(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'integrator',
        'step_size': 0.002
    })
    mock_openmm.Simulation.return_value = create_mock_result(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'simulation',
        'context': mocker.Mock(),
        'minimizeEnergy': create_mock_method(mocker, {
            'start': 0,
            'end': 100,
            'score': 0.95,
            'type': 'energy_minimization'
        })
    })

    # Mock rdkit
    mock_rdkit = mocker.Mock()
    mock_rdkit.Chem = mocker.Mock()
    mock_rdkit.Chem.MolFromSmiles = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'molecule',
        'mol': mocker.Mock()
    })
    mock_rdkit.Chem.MolToSmiles = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'smiles',
        'smiles': 'CC(=O)NC1=CC=C(O)C=C1'
    })
    mock_rdkit.Chem.AllChem = mocker.Mock()

    # Patch modules
    # Patch modules in sys.modules first
    mocker.patch.dict('sys.modules', {
        'torch': mock_torch,
        'transformers': mock_transformers,
        'openmm': mock_openmm,
        'rdkit': mock_rdkit,
        'esm': mock_transformers  # Add ESM mock since it's used in esm_utils
    }, clear=True)  # clear=True ensures no real modules remain

    # Then patch specific module imports
    patches = [
        mocker.patch('models.nlp_analysis.transformers', mock_transformers),
        mocker.patch('models.protein_llm.transformers', mock_transformers),
        mocker.patch('models.qa_system.transformers', mock_transformers),
        mocker.patch('models.openmm_utils.openmm', mock_openmm),
        mocker.patch('models.drug_binding.rdkit', mock_rdkit),
        mocker.patch('models.drug_discovery.rdkit', mock_rdkit),
        mocker.patch('models.esm_utils.esm', mock_transformers)  # Add ESM patch
    ]
    return {
        'torch': mock_torch,
        'transformers': mock_transformers,
        'openmm': mock_openmm,
        'rdkit': mock_rdkit
    }

@pytest.fixture
def mock_esm_model(mocker):
    """Mock ESM model."""
    mock_model = mocker.Mock()
    mock_model.forward = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'embedding',
        'logits': mocker.Mock(),
        'representations': {'last_layer': mocker.Mock()}
    })
    mock_model.get_sequence_embeddings = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'embedding',
        'embeddings': mocker.Mock()
    })
    mock_model.get_attention_maps = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'attention',
        'attention_maps': mocker.Mock()
    })
    return mock_model

@pytest.fixture
def mock_alphabet(mocker):
    """Mock ESM alphabet."""
    mock_alph = mocker.Mock()
    mock_alph.batch_converter = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'batch_conversion',
        'batch_tokens': mocker.Mock(),
        'batch_lengths': mocker.Mock()
    })
    return mock_alph

@pytest.fixture
def domain_analyzer(mocker):
    """Mock domain analyzer."""
    mock_analyzer = mocker.Mock()
    mock_analyzer.identify_domains = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'domain',
        'domains': [
            {'start': 0, 'end': 50, 'score': 0.9, 'type': 'binding'},
            {'start': 51, 'end': 100, 'score': 0.85, 'type': 'catalytic'}
        ]
    })
    mock_analyzer.analyze_domain = create_mock_method(mocker, {
        'start': 0,
        'end': 50,
        'score': 0.9,
        'type': 'domain_analysis',
        'properties': {
            'stability': 0.85,
            'conservation': 0.9,
            'flexibility': 0.75
        }
    })
    return mock_analyzer

@pytest.fixture
def drug_discovery_engine(mocker):
    """Mock drug discovery engine."""
    mock_engine = mocker.Mock()
    mock_engine.analyze_binding_sites = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'binding_sites',
        'sites': [
            {'start': i, 'end': i+20, 'score': 0.9, 'type': 'binding_site'}
            for i in range(0, 80, 20)
        ]
    })
    mock_engine.predict_drug_interactions = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.85,
        'type': 'interactions',
        'interactions': [
            {'start': 0, 'end': 20, 'score': 0.8, 'type': 'hydrogen_bond'},
            {'start': 30, 'end': 50, 'score': 0.75, 'type': 'hydrophobic'}
        ]
    })
    mock_engine.screen_off_targets = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.9,
        'type': 'off_targets',
        'targets': [
            {'start': i, 'end': i+30, 'score': 0.85, 'type': 'off_target'}
            for i in range(0, 70, 30)
        ]
    })
    mock_engine.optimize_binding_site = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'optimization',
        'optimized_site': {
            'start': 0,
            'end': 20,
            'score': 0.9,
            'type': 'binding_site'
        }
    })
    return mock_engine

@pytest.fixture
def drug_binding_simulator(mocker):
    """Mock drug binding simulator."""
    mock_simulator = mocker.Mock()
    mock_simulator.simulate_binding = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'simulation',
        'trajectory': mocker.Mock(),
        'energies': [
            {'start': i, 'end': i+10, 'score': 0.9, 'type': 'energy'}
            for i in range(0, 90, 10)
        ]
    })
    mock_simulator.analyze_trajectory = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.9,
        'type': 'analysis',
        'rmsd': 2.5,
        'contacts': [
            {'start': i, 'end': i+5, 'score': 0.85, 'type': 'contact'}
            for i in range(0, 95, 5)
        ]
    })
    return mock_simulator

@pytest.fixture
def nlp_analyzer(mocker):
    """Mock NLP analyzer."""
    mock_analyzer = mocker.Mock()
    mock_analyzer.answer_protein_question = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.95,
        'type': 'answer',
        'text': 'This is a mock answer about the protein.',
        'confidence': 0.9
    })
    mock_analyzer.generate_sequence_description = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.9,
        'type': 'description',
        'text': 'This is a mock protein sequence description.',
        'features': [
            {'start': i, 'end': i+20, 'score': 0.85, 'type': 'feature'}
            for i in range(0, 80, 20)
        ]
    })
    mock_analyzer.compare_sequences = create_mock_method(mocker, {
        'start': 0,
        'end': 100,
        'score': 0.9,
        'type': 'comparison',
        'similarity': 0.85,
        'differences': [
            {'start': i, 'end': i+10, 'score': 0.8, 'type': 'difference'}
            for i in range(0, 90, 10)
        ]
    })
    return mock_analyzer

@pytest.fixture
def test_protein_sequence():
    """Test protein sequence."""
    return "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"

@pytest.fixture
def test_ligand_smiles():
    """Test ligand SMILES."""
    return "CC(=O)NC1=CC=C(O)C=C1"
