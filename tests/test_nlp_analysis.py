import pytest
from unittest.mock import MagicMock, Mock, patch
import torch
import numpy as np
from proteinflex.models.analysis.nlp_analysis import ProteinNLPAnalyzer, ProteinAnalysis

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.from_pretrained = Mock(return_value=tokenizer)
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    tokenizer.__call__ = Mock(return_value={
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    })
    return tokenizer

@pytest.fixture
def mock_model():
    model = Mock()
    model.eval = Mock(return_value=model)
    model.from_pretrained = Mock(return_value=model)
    outputs = Mock()
    outputs.logits = torch.randn(1, 5, 768)  # Match ESM2 hidden size
    model.forward = Mock(return_value=outputs)
    return model

@pytest.fixture
def mock_nlp_pipeline():
    pipeline = Mock()
    pipeline.from_pretrained = Mock(return_value=pipeline)
    pipeline.return_value = [{'generated_text': 'This protein sequence consists of X amino acids with a molecular weight of Y Da. The predicted secondary structure composition includes alpha-helix, beta-sheet, and random coil.'}]
    pipeline.__call__ = Mock(side_effect=lambda text: [{'generated_text': 'This protein sequence consists of X amino acids with a molecular weight of Y Da. The predicted secondary structure composition includes alpha-helix, beta-sheet, and random coil.'}])
    return pipeline

@pytest.fixture
def mock_protein_analyzer():
    """Mock protein analyzer with realistic Bio.SeqUtils.ProtParam behavior"""
    mock = MagicMock()

    def mock_molecular_weight():
        return 3789.2

    def mock_isoelectric_point():
        return 7.2

    def mock_aromaticity():
        return 0.15

    def mock_instability_index():
        return 35.0

    def mock_secondary_structure_fraction():
        return (0.35, 0.25, 0.40)

    def mock_flexibility():
        # Return torch tensor instead of numpy array
        return torch.tensor([0.1] * 10, dtype=torch.float32)

    # Assign mock methods
    mock.molecular_weight = Mock(side_effect=mock_molecular_weight)
    mock.isoelectric_point = Mock(side_effect=mock_isoelectric_point)
    mock.aromaticity = Mock(side_effect=mock_aromaticity)
    mock.instability_index = Mock(side_effect=mock_instability_index)
    mock.secondary_structure_fraction = Mock(side_effect=mock_secondary_structure_fraction)
    mock.flexibility = Mock(side_effect=mock_flexibility)

    # Configure mock to handle dictionary-like access
    mock.__getitem__ = lambda self, key: torch.tensor([0.1] * 10, dtype=torch.float32) if key == 'flexibility' else None
    mock.keys = lambda: ['flexibility']

    return mock

@pytest.fixture
def sample_sequence():
    return "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"

def test_init_success():
    """Test successful initialization of ProteinNLPAnalyzer"""
    with patch('proteinflex.models.analysis.nlp_analysis.AutoTokenizer') as mock_tokenizer, \
         patch('proteinflex.models.analysis.nlp_analysis.AutoModelForSequenceClassification') as mock_model, \
         patch('proteinflex.models.analysis.nlp_analysis.pipeline') as mock_pipeline:

        # Setup mock returns
        mock_tokenizer.from_pretrained.return_value = Mock(
            __call__=Mock(return_value={
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
            })
        )

        outputs = Mock()
        outputs.logits = torch.randn(1, 5, 768)
        mock_model.from_pretrained.return_value = Mock(
            eval=Mock(return_value=Mock(
                forward=Mock(return_value=outputs)
            ))
        )

        mock_pipeline.return_value = Mock(
            __call__=Mock(return_value=[{
                'generated_text': 'This protein sequence consists of X amino acids with a molecular weight of Y Da.'
            }])
        )

        analyzer = ProteinNLPAnalyzer()
        assert analyzer.tokenizer == mock_tokenizer.from_pretrained.return_value
        assert analyzer.model == mock_model.from_pretrained.return_value
        assert analyzer.nlp_pipeline == mock_pipeline.return_value

def test_init_error():
    """Test error handling during initialization"""
    with patch('proteinflex.models.analysis.nlp_analysis.AutoTokenizer') as mock_tokenizer:
        mock_tokenizer.from_pretrained.side_effect = Exception("Model loading error")
        with pytest.raises(Exception, match="Model loading error"):
            ProteinNLPAnalyzer()

def test_calculate_hydrophobicity(mock_protein_analyzer, sample_sequence):
    """Test hydrophobicity calculation"""
    analyzer = ProteinNLPAnalyzer()
    hydrophobicity = analyzer._calculate_hydrophobicity(sample_sequence)

    assert isinstance(hydrophobicity, float)
    assert -4.5 <= hydrophobicity <= 4.5  # Range of Kyte-Doolittle scale

def test_calculate_sequence_complexity(mock_protein_analyzer, sample_sequence):
    """Test sequence complexity calculation"""
    analyzer = ProteinNLPAnalyzer()
    complexity = analyzer._calculate_sequence_complexity(sample_sequence)

    assert isinstance(complexity, float)
    assert 0 <= complexity <= 100  # Percentage scale

def test_predict_domains(mock_protein_analyzer, sample_sequence):
    """Test domain prediction"""
    analyzer = ProteinNLPAnalyzer()
    domains = analyzer._predict_domains(sample_sequence)

    assert isinstance(domains, list)
    for domain in domains:
        assert isinstance(domain, str)
        assert "domain" in domain.lower()

def test_analyze_stability(mock_protein_analyzer, sample_sequence):
    """Test stability analysis"""
    analyzer = ProteinNLPAnalyzer()
    stability = analyzer._analyze_stability(sample_sequence)

    assert isinstance(stability, dict)
    assert 'index' in stability
    assert 'is_stable' in stability
    assert 'flexibility' in stability
    assert isinstance(stability['is_stable'], bool)

def test_predict_function(mock_protein_analyzer):
    """Test function prediction"""
    analyzer = ProteinNLPAnalyzer()
    embeddings = torch.tensor([[0.6, -0.2, 0.3]])
    function = analyzer._predict_function(embeddings)

    assert isinstance(function, str)
    assert any(keyword in function.lower() for keyword in ['enzymatic', 'structural', 'regulatory'])

def test_generate_description(mock_protein_analyzer):
    """Test description generation"""
    analyzer = ProteinNLPAnalyzer()
    properties = {
        'sequence_length': 100,
        'molecular_weight': 10000.0,
        'isoelectric_point': 7.0,
        'aromaticity': 0.1,
        'instability_index': 30.0,
        'secondary_structure': (0.3, 0.3, 0.4),
        'hydrophobicity': 0.5,
        'complexity': 75.0,
        'domains': ['Hydrophobic domain (1-50)'],
        'stability': {'index': 30.0, 'is_stable': True, 'flexibility': 0.4},
        'function': 'Likely enzymatic activity'
    }

    description = analyzer._generate_description(**properties)
    assert isinstance(description, str)
    assert len(description) > 0
    assert 'amino acids' in description
    assert 'molecular weight' in description

def test_analyze_sequence_success(mock_protein_analyzer, sample_sequence):
    """Test successful sequence analysis"""
    with patch('proteinflex.models.analysis.nlp_analysis.AutoTokenizer') as mock_tokenizer, \
         patch('proteinflex.models.analysis.nlp_analysis.AutoModelForSequenceClassification') as mock_model, \
         patch('proteinflex.models.analysis.nlp_analysis.pipeline') as mock_pipeline, \
         patch('proteinflex.models.analysis.nlp_analysis.ProteinAnalysis') as mock_protein_analysis, \
         patch('proteinflex.models.analysis.nlp_analysis.torch.mean', wraps=torch.mean) as mock_mean:

        # Setup mock returns for protein analysis
        mock_protein_analyzer.molecular_weight.return_value = 3789.2
        mock_protein_analyzer.isoelectric_point.return_value = 7.2
        mock_protein_analyzer.aromaticity.return_value = 0.15
        mock_protein_analyzer.instability_index.return_value = 35.0
        mock_protein_analyzer.secondary_structure_fraction.return_value = (0.35, 0.25, 0.40)
        mock_protein_analyzer.flexibility.return_value = torch.tensor([0.1] * len(sample_sequence), dtype=torch.float32)
        mock_protein_analysis.return_value = mock_protein_analyzer

        # Setup mock tokenizer
        mock_tokenizer.from_pretrained.return_value = MagicMock(
            __call__=lambda *args, **kwargs: {
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
            }
        )

        # Create mock model with proper tensor outputs for embeddings
        class MockOutput:
            def __init__(self):
                # Create a tensor with shape [batch_size, sequence_length] and values > 0.5
                self.logits = torch.full((1, 5), 0.6, dtype=torch.float32)

        class MockModel(MagicMock):
            def __call__(self, **kwargs):
                return MockOutput()

            def eval(self):
                return self

        mock_model.from_pretrained.return_value = MockModel()

        # Mock pipeline with proper text generation
        def mock_pipeline_call(*args, **kwargs):
            stability = "stable" if mock_protein_analyzer.instability_index() < 40 else "unstable"
            return [{'generated_text': (
                f"This protein sequence consists of {len(sample_sequence)} amino acids "
                f"with a molecular weight of {mock_protein_analyzer.molecular_weight()} Da. "
                f"The predicted secondary structure composition includes {mock_protein_analyzer.secondary_structure_fraction()[0]:.1%} "
                f"alpha-helix, {mock_protein_analyzer.secondary_structure_fraction()[1]:.1%} beta-sheet, and "
                f"{mock_protein_analyzer.secondary_structure_fraction()[2]:.1%} random coil. "
                f"The protein appears to be {stability} with enzymatic activity."
            )}]
        mock_pipeline.return_value = MagicMock(__call__=mock_pipeline_call)

        analyzer = ProteinNLPAnalyzer()
        result = analyzer.analyze_sequence(sample_sequence)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "amino acids" in result
        assert "molecular weight" in result
        assert "3789.2" in result
        assert "stable" in result
        assert "enzymatic activity" in result
        assert "alpha-helix" in result
        assert "beta-sheet" in result
        assert "random coil" in result

        # Verify tensor operations
        mock_mean.assert_called()

def test_analyze_sequence_error():
    """Test error handling in sequence analysis"""
    with patch('proteinflex.models.analysis.nlp_analysis.AutoTokenizer') as mock_tokenizer, \
         patch('proteinflex.models.analysis.nlp_analysis.AutoModelForSequenceClassification') as mock_model, \
         patch('proteinflex.models.analysis.nlp_analysis.pipeline') as mock_pipeline, \
         patch('proteinflex.models.analysis.nlp_analysis.ProteinAnalysis') as mock_protein_analysis:

        mock_tokenizer.from_pretrained.return_value = Mock(side_effect=Exception("Tokenization error"))
        analyzer = ProteinNLPAnalyzer()
        result = analyzer.analyze_sequence(None)
        assert result == "Error analyzing sequence"

@pytest.mark.parametrize("sequence,expected_complexity", [
    ("AAAA", 5.0),           # Single amino acid
    ("ACDEFGHIKLM", 55.0),   # Multiple unique amino acids
    ("", 0.0)                # Empty sequence
])
def test_sequence_complexity_edge_cases(mock_protein_analyzer, sequence, expected_complexity):
    """Test sequence complexity calculation with edge cases"""
    analyzer = ProteinNLPAnalyzer()
    complexity = analyzer._calculate_sequence_complexity(sequence)
    assert abs(complexity - expected_complexity) < 0.1

def test_stability_analysis_error_handling(mock_protein_analyzer):
    """Test error handling in stability analysis"""
    with patch('proteinflex.models.analysis.nlp_analysis.ProteinAnalysis') as mock_protein_analysis:
        mock_protein_analysis.side_effect = ValueError("Invalid amino acid")
        analyzer = ProteinNLPAnalyzer()
        result = analyzer._analyze_stability("X")  # Invalid amino acid
        assert result is None
