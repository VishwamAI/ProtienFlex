"""
Pytest configuration file for ProteinFlex tests
"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
import torch
import numpy as np
import esm

@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    """Automatically mock external dependencies before any imports."""
    # Mock only the essential imports and methods
    mocker.patch('models.nlp_analysis.transformers.AutoTokenizer.from_pretrained',
                return_value=mocker.MagicMock(
                    encode=lambda *args, **kwargs: [1] * 10,
                    decode=lambda *args, **kwargs: "mock decoded text"
                ))

    mocker.patch('models.nlp_analysis.transformers.AutoModelForSequenceClassification.from_pretrained',
                return_value=mocker.MagicMock(
                    eval=lambda: mocker.MagicMock(),
                    generate=lambda *args, **kwargs: [mocker.MagicMock(sequences=["mock generated text"])],
                    forward=lambda *args, **kwargs: {"logits": torch.zeros(1, 10, 100)}
                ))

    mocker.patch('models.openmm_utils.openmm.Simulation',
                return_value=mocker.MagicMock(
                    step=lambda steps: None,
                    minimizeEnergy=lambda: None
                ))

    # Add project root to Python path
    if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def mock_esm_model():
    """Mock ESM model with expected outputs."""
    mock = Mock()
    mock.eval.return_value = mock

    # Make representations dict subscriptable
    representations = {}
    representations[33] = torch.zeros((1, 100, 1280))  # Match expected shape

    # Create a subscriptable MagicMock for the model output
    model_output = MagicMock()
    model_output.__getitem__.side_effect = lambda x: representations[x] if x == "representations" else {
        "attentions": torch.zeros((1, 33, 100, 100)),  # Match attention map shape
        "contacts": torch.zeros((1, 100, 100))  # Match contact map shape
    }[x]

    # Configure forward method to return properly structured output
    mock.return_value = model_output
    mock.forward.return_value = model_output
    return mock

@pytest.fixture
def mock_alphabet():
    """Mock ESM alphabet with batch converter."""
    mock = Mock()
    batch_converter = MagicMock()
    # Match expected token dimensions for protein sequences
    batch_tokens = torch.zeros((1, 100), dtype=torch.long)

    # Configure batch_converter to return proper tuple with correct dimensions
    def batch_converter_return(data):
        batch_labels = ["protein"]
        batch_strs = [data[0][1]]  # Extract sequence from (idx, sequence) tuple
        return (batch_labels, batch_strs, batch_tokens)
    batch_converter.side_effect = batch_converter_return

    mock.get_batch_converter.return_value = batch_converter
    mock.batch_converter = batch_converter
    return mock

@pytest.fixture
def mock_batch_converter():
    """Mock the ESM batch converter to return expected tuple structure."""
    mock = Mock()
    batch_tokens = torch.zeros((1, 32), dtype=torch.long)

    # Configure mock to handle input data properly
    def converter_return(data):
        return (["protein"], [data[0][1]], batch_tokens)
    mock.side_effect = converter_return
    return mock

@pytest.fixture
def domain_analyzer(mock_esm_model, mock_alphabet):
    """Create a DomainAnalyzer instance with mocked dependencies."""
    from models.domain_analysis import DomainAnalyzer
    # Initialize analyzer with mock dependencies directly
    analyzer = DomainAnalyzer(model=mock_esm_model, alphabet=mock_alphabet)
    return analyzer

@pytest.fixture
def drug_discovery_engine(mock_esm_model, mock_alphabet):
    """Create a DrugDiscoveryEngine instance with mocked dependencies."""
    with patch('models.drug_discovery.esm') as mock_esm:
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_esm_model, mock_alphabet)
        from models.drug_discovery import DrugDiscoveryEngine
        engine = DrugDiscoveryEngine()
        return engine

@pytest.fixture
def drug_binding_simulator():
    """Create a DrugBindingSimulator instance with mocked dependencies."""
    from models.drug_binding import DrugBindingSimulator
    simulator = DrugBindingSimulator()
    # Mock OpenMM simulation setup
    simulator.setup_simulation = MagicMock(return_value=Mock())
    return simulator

@pytest.fixture
def test_protein_sequence():
    """Provide a test protein sequence."""
    return "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"

@pytest.fixture
def test_ligand_smiles():
    """Provide a test ligand SMILES string."""
    return "CC1=CC=C(C=C1)CC(C(=O)O)N"
