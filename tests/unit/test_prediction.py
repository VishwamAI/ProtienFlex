"""
Unit tests for structure prediction components.
"""

import pytest
import numpy as np
from pathlib import Path
import os
import tempfile
import shutil
from unittest.mock import Mock, patch

from models.prediction import structure_converter, alphafold_interface

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / 'data'
ALANINE_PDB = TEST_DATA_DIR / 'alanine-dipeptide.pdb'

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_alphafold_predictor():
    """Create mock AlphaFold predictor."""
    mock = Mock()
    # Mock prediction output
    mock.predict_structure.return_value = {
        'positions': np.random.random((10, 3)),  # 10 atoms, 3 coordinates
        'plddt': np.random.random(10),  # per-residue confidence
        'pae': np.random.random((10, 10)),  # pairwise aligned error
        'mean_plddt': 0.85
    }
    return mock

@pytest.fixture
def sample_sequence():
    """Sample protein sequence for testing."""
    return "MKLLVLGLRSGSGKS"

def test_structure_converter_initialization():
    """Test structure converter initialization."""
    converter = structure_converter.StructureConverter()
    assert converter is not None

def test_alphafold_to_openmm_conversion(mock_alphafold_predictor, sample_sequence):
    """Test conversion from AlphaFold output to OpenMM system."""
    # Get mock prediction
    pred = mock_alphafold_predictor.predict_structure(sample_sequence)

    # Convert to OpenMM
    converter = structure_converter.StructureConverter()
    system = converter.alphafold_to_openmm(
        positions=pred['positions'],
        sequence=sample_sequence,
        plddt=pred['plddt']
    )

    # Check system components
    assert system['topology'] is not None
    assert system['positions'] is not None
    assert system['system'] is not None

def test_pdb_conversion(temp_dir):
    """Test PDB file conversion."""
    converter = structure_converter.StructureConverter()

    # Load and convert PDB
    system = converter.pdb_to_openmm(ALANINE_PDB)

    # Check system components
    assert system['topology'] is not None
    assert system['positions'] is not None
    assert system['system'] is not None

    # Test saving
    output_pdb = os.path.join(temp_dir, 'converted.pdb')
    converter.save_structure(system, output_pdb)
    assert os.path.exists(output_pdb)

def test_confidence_metrics():
    """Test confidence metrics calculation."""
    # Create sample prediction data
    plddt = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    pae = np.random.random((5, 5))

    # Calculate confidence metrics
    metrics = structure_converter.calculate_confidence_metrics(plddt, pae)

    # Check metrics
    assert 'mean_plddt' in metrics
    assert 'median_plddt' in metrics
    assert 'mean_pae' in metrics
    assert 'max_pae' in metrics
    assert 0 <= metrics['mean_plddt'] <= 1
    assert metrics['max_pae'] >= 0

@patch('models.prediction.alphafold_interface.AlphaFoldPredictor')
def test_alphafold_prediction(mock_predictor_class, sample_sequence, temp_dir):
    """Test AlphaFold prediction interface."""
    # Setup mock predictor
    mock_predictor = mock_predictor_class.return_value
    mock_predictor.predict_structure.return_value = {
        'positions': np.random.random((10, 3)),
        'plddt': np.random.random(10),
        'pae': np.random.random((10, 10)),
        'mean_plddt': 0.85
    }

    # Create predictor
    predictor = alphafold_interface.AlphaFoldPredictor(
        model_dir='/path/to/models',
        output_dir=temp_dir
    )

    # Make prediction
    result = predictor.predict_structure(sample_sequence)

    # Check result structure
    assert 'positions' in result
    assert 'plddt' in result
    assert 'pae' in result
    assert 'mean_plddt' in result

def test_structure_validation():
    """Test structure validation functions."""
    # Create sample structure data
    positions = np.random.random((10, 3))
    plddt = np.random.random(10)

    # Validate structure
    validation = structure_converter.validate_structure(positions, plddt)

    # Check validation results
    assert 'is_valid' in validation
    assert 'validation_messages' in validation
    assert isinstance(validation['is_valid'], bool)
    assert isinstance(validation['validation_messages'], list)

def test_error_handling():
    """Test error handling in prediction components."""
    converter = structure_converter.StructureConverter()

    # Test invalid sequence
    with pytest.raises(ValueError):
        converter.alphafold_to_openmm(
            positions=np.random.random((10, 3)),
            sequence="Invalid123",  # Invalid sequence
            plddt=np.random.random(10)
        )

    # Test mismatched dimensions
    with pytest.raises(ValueError):
        converter.alphafold_to_openmm(
            positions=np.random.random((10, 3)),
            sequence="AAAAAA",  # 6 residues
            plddt=np.random.random(10)  # 10 confidence values
        )

@pytest.mark.parametrize("confidence_threshold", [0.5, 0.7, 0.9])
def test_confidence_filtering(confidence_threshold):
    """Test filtering based on confidence scores."""
    # Create sample data
    positions = np.random.random((10, 3))
    plddt = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05])

    # Filter based on confidence
    filtered = structure_converter.filter_by_confidence(
        positions, plddt, threshold=confidence_threshold
    )

    # Check filtering
    assert len(filtered['positions']) <= len(positions)
    assert all(conf >= confidence_threshold for conf in filtered['plddt'])

def test_format_conversion():
    """Test structure format conversion utilities."""
    converter = structure_converter.StructureConverter()

    # Test PDB to internal format
    internal = converter.pdb_to_internal(ALANINE_PDB)
    assert 'positions' in internal
    assert 'topology' in internal

    # Test internal to PDB format
    with tempfile.NamedTemporaryFile(suffix='.pdb') as tmp:
        converter.internal_to_pdb(internal, tmp.name)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0

def test_batch_prediction(mock_alphafold_predictor):
    """Test batch structure prediction."""
    sequences = [
        "MKLLVLGLRSGSGKS",
        "MALWMRLLPLLALLALWGPD"
    ]

    predictor = alphafold_interface.AlphaFoldPredictor(
        model_dir='/path/to/models',
        batch_size=2
    )

    # Mock batch prediction
    with patch.object(predictor, '_predict_batch') as mock_predict:
        mock_predict.return_value = [{
            'positions': np.random.random((len(seq), 3)),
            'plddt': np.random.random(len(seq)),
            'pae': np.random.random((len(seq), len(seq))),
            'mean_plddt': 0.85
        } for seq in sequences]

        results = predictor.predict_structures(sequences)

        # Check results
        assert len(results) == len(sequences)
        for result in results:
            assert all(key in result for key in ['positions', 'plddt', 'pae', 'mean_plddt'])
