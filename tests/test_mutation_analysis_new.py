import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
from proteinflex.models.mutation_analysis import MutationAnalyzer

@pytest.fixture
def mock_esm_model(mock_esm_enhanced):
    return mock_esm_enhanced.pretrained.esm2_t33_650M_UR50D.return_value[0]

@pytest.fixture
def mutation_analyzer(mock_esm_model):
    return MutationAnalyzer()

@pytest.fixture
def sample_sequence():
    return "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"

@pytest.fixture
def sample_mutation():
    return {
        'position': 5,
        'original': 'E',
        'mutant': 'A'
    }

def test_init(mutation_analyzer):
    """Test initialization of MutationAnalyzer"""
    assert hasattr(mutation_analyzer, 'model')
    assert hasattr(mutation_analyzer, 'device')
    assert str(mutation_analyzer.device) in ['cpu', 'cuda:0']

def test_validate_mutation(mutation_analyzer, sample_sequence, sample_mutation):
    """Test mutation validation"""
    assert mutation_analyzer._validate_mutation(sample_sequence, sample_mutation)

    # Test invalid cases
    with pytest.raises(ValueError):
        mutation_analyzer._validate_mutation(sample_sequence, {})

    with pytest.raises(ValueError):
        mutation_analyzer._validate_mutation(sample_sequence, {
            'position': -1,
            'original': 'E',
            'mutant': 'A'
        })

    with pytest.raises(ValueError):
        mutation_analyzer._validate_mutation(sample_sequence, {
            'position': 5,
            'original': 'X',  # Invalid amino acid
            'mutant': 'A'
        })

def test_calculate_stability_impact(mutation_analyzer, sample_sequence, sample_mutation):
    """Test stability impact calculation"""
    impact = mutation_analyzer.calculate_stability_impact(sample_sequence, sample_mutation)

    assert isinstance(impact, dict)
    assert 'ddG' in impact
    assert 'confidence' in impact
    assert isinstance(impact['ddG'], float)
    assert isinstance(impact['confidence'], float)
    assert 0 <= impact['confidence'] <= 1

def test_predict_structural_impact(mutation_analyzer, sample_sequence, sample_mutation):
    """Test structural impact prediction"""
    impact = mutation_analyzer.predict_structural_impact(sample_sequence, sample_mutation)

    assert isinstance(impact, dict)
    assert 'local_structure_change' in impact
    assert 'global_structure_change' in impact
    assert 'confidence' in impact
    assert isinstance(impact['local_structure_change'], float)
    assert isinstance(impact['global_structure_change'], float)
    assert isinstance(impact['confidence'], float)

def test_analyze_conservation(mutation_analyzer, sample_sequence, sample_mutation):
    """Test conservation analysis"""
    analysis = mutation_analyzer.analyze_conservation(sample_sequence, sample_mutation)

    assert isinstance(analysis, dict)
    assert 'conservation_score' in analysis
    assert 'evolutionary_coupling' in analysis
    assert 'frequency_in_homologs' in analysis
    assert isinstance(analysis['conservation_score'], float)
    assert isinstance(analysis['evolutionary_coupling'], float)
    assert isinstance(analysis['frequency_in_homologs'], float)

def test_batch_analysis(mutation_analyzer, sample_sequence):
    """Test batch mutation analysis"""
    mutations = [
        {'position': 5, 'original': 'E', 'mutant': 'A'},
        {'position': 10, 'original': 'T', 'mutant': 'S'}
    ]

    results = mutation_analyzer.analyze_mutations(sample_sequence, mutations)

    assert isinstance(results, list)
    assert len(results) == len(mutations)
    for result in results:
        assert 'stability_impact' in result
        assert 'structural_impact' in result
        assert 'conservation_analysis' in result

def test_error_handling(mutation_analyzer, sample_sequence):
    """Test error handling"""
    # Test with invalid sequence
    with pytest.raises(ValueError):
        mutation_analyzer.calculate_stability_impact("", sample_mutation)

    # Test with None mutation
    with pytest.raises(ValueError):
        mutation_analyzer.predict_structural_impact(sample_sequence, None)

    # Test with invalid mutation position
    with pytest.raises(ValueError):
        mutation_analyzer.analyze_conservation(sample_sequence, {
            'position': 100,  # Beyond sequence length
            'original': 'E',
            'mutant': 'A'
        })

def test_device_handling():
    """Test device handling (CPU/GPU)"""
    with patch('torch.cuda') as mock_cuda:
        # Test CPU
        mock_cuda.is_available.return_value = False
        analyzer_cpu = MutationAnalyzer()
        assert str(analyzer_cpu.device) == 'cpu'

        # Test GPU
        mock_cuda.is_available.return_value = True
        analyzer_gpu = MutationAnalyzer()
        assert str(analyzer_gpu.device).startswith('cuda')

def test_memory_management(mutation_analyzer, sample_sequence, sample_mutation):
    """Test memory management during analysis"""
    with patch('torch.cuda') as mock_cuda:
        mock_cuda.is_available.return_value = True
        mock_cuda.empty_cache = Mock()

        mutation_analyzer.calculate_stability_impact(sample_sequence, sample_mutation)
        mock_cuda.empty_cache.assert_called()

def test_output_validation(mutation_analyzer, sample_sequence, sample_mutation):
    """Test output validation and normalization"""
    impact = mutation_analyzer.calculate_stability_impact(sample_sequence, sample_mutation)
    assert isinstance(impact['ddG'], float)
    assert -50 <= impact['ddG'] <= 50  # Reasonable range for stability change

    structural = mutation_analyzer.predict_structural_impact(sample_sequence, sample_mutation)
    assert 0 <= structural['local_structure_change'] <= 1
    assert 0 <= structural['global_structure_change'] <= 1

    conservation = mutation_analyzer.analyze_conservation(sample_sequence, sample_mutation)
    assert 0 <= conservation['conservation_score'] <= 1
