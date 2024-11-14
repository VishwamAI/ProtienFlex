"""
Test suite for multi-modal protein understanding integration
"""
import pytest
import torch
from models.analysis.enhanced_sequence_analyzer import EnhancedSequenceAnalyzer
from models.analysis.structure_predictor import StructurePredictor
from models.analysis.function_predictor import FunctionPredictor
from models.analysis.multimodal_integrator import MultiModalProteinAnalyzer

@pytest.fixture
def config():
    return {
        'hidden_size': 768,
        'num_attention_heads': 8,
        'num_go_terms': 1000
    }

@pytest.fixture
def test_sequence():
    return "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

@pytest.fixture
def multimodal_analyzer(config):
    return MultiModalProteinAnalyzer(config)

def test_sequence_analysis(multimodal_analyzer, test_sequence):
    """Test sequence analysis component"""
    results = multimodal_analyzer.sequence_analyzer(test_sequence)
    assert 'features' in results
    assert isinstance(results['features'], torch.Tensor)
    assert results['features'].shape[-1] == multimodal_analyzer.config['hidden_size']

def test_structure_prediction(multimodal_analyzer, test_sequence):
    """Test structure prediction component"""
    sequence_results = multimodal_analyzer.sequence_analyzer(test_sequence)
    structure_results = multimodal_analyzer.structure_predictor(sequence_results['features'])
    assert 'refined_structure' in structure_results
    assert isinstance(structure_results['refined_structure'], torch.Tensor)

def test_function_prediction(multimodal_analyzer, test_sequence):
    """Test function prediction component"""
    sequence_results = multimodal_analyzer.sequence_analyzer(test_sequence)
    structure_results = multimodal_analyzer.structure_predictor(sequence_results['features'])
    function_results = multimodal_analyzer.function_predictor(
        sequence_results['features'],
        structure_results['refined_structure']
    )
    assert 'go_terms' in function_results
    assert 'ppi' in function_results
    assert 'enzyme_activity' in function_results
    assert 'binding_sites' in function_results

def test_multimodal_integration(multimodal_analyzer, test_sequence):
    """Test complete multi-modal integration"""
    results = multimodal_analyzer.analyze_protein(test_sequence)

    # Verify sequence analysis results
    assert 'sequence_analysis' in results
    assert 'features' in results['sequence_analysis']

    # Verify structure prediction results
    assert 'structure_prediction' in results
    assert 'refined_structure' in results['structure_prediction']

    # Verify function prediction results
    assert 'function_prediction' in results
    assert 'go_terms' in results['function_prediction']
    assert 'ppi' in results['function_prediction']

    # Verify unified prediction
    assert 'unified_prediction' in results
    assert 'confidence' in results['unified_prediction']

    # Verify feature integration
    assert 'integrated_features' in results
    assert isinstance(results['integrated_features'], torch.Tensor)

def test_cross_modal_attention(multimodal_analyzer, test_sequence):
    """Test cross-modal attention mechanism"""
    sequence_results = multimodal_analyzer.sequence_analyzer(test_sequence)
    structure_results = multimodal_analyzer.structure_predictor(sequence_results['features'])

    integrated_features = multimodal_analyzer.cross_modal_attention(
        sequence_results['features'],
        structure_results['refined_structure']
    )

    assert isinstance(integrated_features, torch.Tensor)
    assert integrated_features.shape[-1] == multimodal_analyzer.config['hidden_size']

def test_confidence_estimation(multimodal_analyzer, test_sequence):
    """Test confidence estimation for predictions"""
    results = multimodal_analyzer.analyze_protein(test_sequence)
    assert 'confidence' in results['unified_prediction']
    confidence = results['unified_prediction']['confidence']
    assert isinstance(confidence, torch.Tensor)
    assert 0 <= confidence.item() <= 1

if __name__ == '__main__':
    pytest.main([__file__])
