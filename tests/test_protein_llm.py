import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from proteinflex.models.generative.protein_llm import ProteinLanguageModel

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    return tokenizer

@pytest.fixture
def mock_model():
    model = Mock()
    model.eval = Mock()
    outputs = Mock()
    # Create tensors matching the sequence length
    seq_length = len("MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG")  # From sample_sequence
    outputs.last_hidden_state = torch.randn(1, seq_length, 768)
    outputs.attentions = [torch.randn(1, 8, seq_length, seq_length)] * 5
    model.return_value = outputs
    # Ensure model.to() works properly
    model.to = Mock(return_value=model)
    return model

@pytest.fixture
def mock_stability_model():
    model = Mock()
    model.eval = Mock()
    outputs = Mock()
    # Create actual tensor for logits that can be used with torch.sigmoid
    outputs.logits = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    # Configure the model to return the outputs when called
    model.return_value = outputs
    # Ensure the model is properly mocked for to() calls
    model.to = Mock(return_value=model)
    return model

@pytest.fixture
def mock_text_pipeline():
    pipeline = Mock()
    pipeline.return_value = [{'generated_text': 'Sample protein description'}]
    return pipeline

@pytest.fixture
def protein_llm(mock_tokenizer, mock_model, mock_stability_model, mock_text_pipeline):
    with patch('proteinflex.models.generative.protein_llm.AutoTokenizer') as mock_auto_tokenizer, \
         patch('proteinflex.models.generative.protein_llm.AutoModel') as mock_auto_model, \
         patch('proteinflex.models.generative.protein_llm.AutoModelForSequenceClassification') as mock_auto_stability, \
         patch('proteinflex.models.generative.protein_llm.pipeline') as mock_pipeline:

        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_stability.from_pretrained.return_value = mock_stability_model
        mock_pipeline.return_value = mock_text_pipeline

        return ProteinLanguageModel()

@pytest.fixture
def sample_sequence():
    return "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"

def test_init_success():
    """Test successful initialization of ProteinLanguageModel"""
    with patch('proteinflex.models.generative.protein_llm.AutoTokenizer') as mock_tokenizer, \
         patch('proteinflex.models.generative.protein_llm.AutoModel') as mock_model, \
         patch('proteinflex.models.generative.protein_llm.AutoModelForSequenceClassification') as mock_stability, \
         patch('proteinflex.models.generative.protein_llm.pipeline') as mock_pipeline:

        # Configure mock returns
        mock_model_instance = Mock(name='model')
        mock_tokenizer_instance = Mock(name='tokenizer')
        mock_stability_instance = Mock(name='stability_model')
        mock_pipeline_instance = Mock(name='text_pipeline')

        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_stability.from_pretrained.return_value = mock_stability_instance
        mock_pipeline.return_value = mock_pipeline_instance

        # Configure to() method returns
        mock_model_instance.to = Mock(return_value=mock_model_instance)
        mock_stability_instance.to = Mock(return_value=mock_stability_instance)

        llm = ProteinLanguageModel()

        # Verify correct models were loaded
        mock_tokenizer.from_pretrained.assert_called_once_with('facebook/esm2_t33_650M_UR50D')
        mock_model.from_pretrained.assert_called_once_with('facebook/esm2_t33_650M_UR50D')
        mock_stability.from_pretrained.assert_called_once_with('facebook/esm-1b-stability-prediction')

        # Verify attributes were set correctly
        assert llm.tokenizer is mock_tokenizer_instance
        assert llm.model is mock_model_instance
        assert llm.stability_model is mock_stability_instance

def test_analyze_sequence(protein_llm, sample_sequence):
    """Test sequence analysis functionality"""
    result = protein_llm.analyze_sequence(sample_sequence)

    assert isinstance(result, dict)
    assert 'embeddings' in result
    assert 'residue_importance' in result
    assert 'basic_properties' in result

    properties = result['basic_properties']
    assert 'molecular_weight' in properties
    assert 'aromaticity' in properties
    assert 'instability_index' in properties
    assert 'isoelectric_point' in properties

def test_predict_mutations(protein_llm, sample_sequence):
    """Test mutation prediction"""
    positions = [0, 5]
    predictions = protein_llm.predict_mutations(sample_sequence, positions)

    assert isinstance(predictions, list)
    assert len(predictions) == len(positions)

    for pred in predictions:
        assert 'position' in pred
        assert 'mutations' in pred
        assert isinstance(pred['mutations'], list)

        for mut in pred['mutations']:
            assert 'mutation' in mut
            assert 'stability_change' in mut
            assert 'impact' in mut
            assert 'confidence' in mut

def test_analyze_drug_binding(protein_llm, sample_sequence):
    """Test drug binding analysis"""
    result = protein_llm.analyze_drug_binding(sample_sequence)

    assert isinstance(result, list)
    if result:  # If binding sites found
        site = result[0]
        assert 'position' in site
        assert 'score' in site
        assert 'properties' in site
        assert 'druggability' in site

def test_predict_stability_change(protein_llm, sample_sequence):
    """Test stability change prediction"""
    original_seq = sample_sequence
    mutated_seq = sample_sequence[:5] + "A" + sample_sequence[6:]

    change = protein_llm._predict_stability_change(original_seq, mutated_seq)
    assert isinstance(change, float)

def test_calculate_mutation_impact(protein_llm, sample_sequence):
    """Test mutation impact calculation"""
    impact = protein_llm._calculate_mutation_impact(sample_sequence, 0, "A")
    assert isinstance(impact, float)
    assert 0 <= impact <= 1

def test_calculate_confidence(protein_llm):
    """Test confidence calculation"""
    confidence = protein_llm._calculate_confidence(0.5, 0.7)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 100

def test_adjust_parameters(protein_llm):
    """Test parameter adjustment"""
    original_threshold = protein_llm.attention_threshold
    importance_scores = torch.tensor([0.8, 0.9, 0.7])

    protein_llm._adjust_parameters(importance_scores)
    assert 0.1 <= protein_llm.attention_threshold <= 0.9
    assert protein_llm.attention_threshold != original_threshold

def test_analyze_binding_site_properties(protein_llm, sample_sequence):
    """Test binding site property analysis"""
    properties = protein_llm._analyze_binding_site_properties(sample_sequence, 5)

    assert isinstance(properties, dict)
    assert 'hydrophobicity' in properties
    assert 'flexibility' in properties
    assert 'surface_accessibility' in properties

def test_calculate_druggability_score(protein_llm, sample_sequence):
    """Test druggability score calculation"""
    score = protein_llm._calculate_druggability_score(sample_sequence, 5)
    assert isinstance(score, float)
    assert -1 <= score <= 1

@pytest.mark.parametrize("sequence,position,expected_type", [
    ("MAEGEITTFT", 0, float),    # Start position
    ("MAEGEITTFT", 5, float),    # Middle position
    ("MAEGEITTFT", 9, float)     # End position
])
def test_surface_accessibility_positions(protein_llm, sequence, position, expected_type):
    """Test surface accessibility calculation at different positions"""
    score = protein_llm._calculate_surface_accessibility(sequence, position)
    assert isinstance(score, expected_type)
    assert 0 <= score <= 1

def test_error_handling(protein_llm):
    """Test error handling for invalid inputs"""
    # Test with None sequence
    with pytest.raises(Exception):
        protein_llm.analyze_sequence(None)

    # Test with empty sequence
    with pytest.raises(Exception):
        protein_llm.predict_mutations("", [0])

    # Test with invalid position
    with pytest.raises(Exception):
        protein_llm.predict_mutations("MAE", [5])

@pytest.mark.parametrize("ligand_features", [
    None,
    {'molecular_weight': 300, 'logp': 2.5},
    {'molecular_weight': 500, 'logp': -1.0}
])
def test_ligand_compatibility(protein_llm, sample_sequence, ligand_features):
    """Test ligand compatibility prediction with different features"""
    result = protein_llm.analyze_drug_binding(sample_sequence, ligand_features)
    assert isinstance(result, list)

    if ligand_features and result:
        assert 'ligand_compatibility' in result[0]
