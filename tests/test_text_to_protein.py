import pytest
import torch
from unittest.mock import Mock, patch
from proteinflex.models.generative.text_to_protein_generator import TextToProteinGenerator

@pytest.fixture
def mock_llm():
    model = Mock()
    model.eval.return_value = model
    model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    model.forward = Mock(return_value={
        'logits': torch.randn(1, 10, 20),
        'hidden_states': torch.randn(1, 10, 768)
    })
    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = Mock(return_value="MAEGEITT")
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer

@pytest.fixture
def text_to_protein_generator(mock_llm, mock_tokenizer):
    with patch('proteinflex.models.generative.text_to_protein_generator.AutoModelForCausalLM') as mock_auto_model, \
         patch('proteinflex.models.generative.text_to_protein_generator.AutoTokenizer') as mock_auto_tokenizer:
        mock_auto_model.from_pretrained.return_value = mock_llm
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        return TextToProteinGenerator()

def test_init(mock_llm, mock_tokenizer):
    """Test initialization of TextToProteinGenerator"""
    with patch('proteinflex.models.generative.text_to_protein_generator.AutoModelForCausalLM') as mock_auto_model, \
         patch('proteinflex.models.generative.text_to_protein_generator.AutoTokenizer') as mock_auto_tokenizer:
        mock_auto_model.from_pretrained.return_value = mock_llm
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        generator = TextToProteinGenerator()
        assert generator.model == mock_llm
        assert generator.tokenizer == mock_tokenizer
        assert str(generator.device) in ['cpu', 'cuda:0']

def test_generate_from_text(text_to_protein_generator):
    """Test protein generation from text description"""
    description = "Generate a stable protein that binds to ATP"
    sequence = text_to_protein_generator.generate_from_text(description)

    assert isinstance(sequence, str)
    assert len(sequence) > 0
    assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)

def test_generate_with_properties(text_to_protein_generator):
    """Test protein generation with specific properties"""
    properties = {
        'length': 100,
        'stability': 'high',
        'function': 'enzyme',
        'target': 'ATP'
    }

    sequence = text_to_protein_generator.generate_with_properties(properties)

    assert isinstance(sequence, str)
    assert len(sequence) > 0
    assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)

def test_batch_generation(text_to_protein_generator):
    """Test batch generation of proteins"""
    descriptions = [
        "Generate a stable protein that binds to ATP",
        "Design a membrane protein with high stability"
    ]

    sequences = text_to_protein_generator.generate_batch(descriptions)

    assert isinstance(sequences, list)
    assert len(sequences) == len(descriptions)
    assert all(isinstance(seq, str) for seq in sequences)
    assert all(len(seq) > 0 for seq in sequences)

def test_generate_with_constraints(text_to_protein_generator):
    """Test generation with structural constraints"""
    description = "Generate a stable protein"
    constraints = {
        'secondary_structure': 'alpha_helix',
        'solubility': 'high',
        'ph_stability': 'neutral'
    }

    sequence = text_to_protein_generator.generate_with_constraints(description, constraints)

    assert isinstance(sequence, str)
    assert len(sequence) > 0
    assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)

def test_error_handling(text_to_protein_generator):
    """Test error handling"""
    # Test with empty description
    with pytest.raises(ValueError):
        text_to_protein_generator.generate_from_text("")

    # Test with invalid properties
    with pytest.raises(ValueError):
        text_to_protein_generator.generate_with_properties({'invalid_key': 'value'})

    # Test with invalid constraints
    with pytest.raises(ValueError):
        text_to_protein_generator.generate_with_constraints(
            "Generate a protein",
            {'invalid_constraint': 'value'}
        )

def test_sequence_validation(text_to_protein_generator):
    """Test sequence validation"""
    # Test valid sequence
    assert text_to_protein_generator._validate_sequence("MAEGEITT")

    # Test invalid sequences
    assert not text_to_protein_generator._validate_sequence("")
    assert not text_to_protein_generator._validate_sequence("XYZ123")
    assert not text_to_protein_generator._validate_sequence(None)

def test_device_handling():
    """Test device handling (CPU/GPU)"""
    with patch('torch.cuda') as mock_cuda, \
         patch('proteinflex.models.generative.text_to_protein_generator.AutoModelForCausalLM') as mock_auto_model, \
         patch('proteinflex.models.generative.text_to_protein_generator.AutoTokenizer') as mock_auto_tokenizer:

        # Test CPU
        mock_cuda.is_available.return_value = False
        generator_cpu = TextToProteinGenerator()
        assert str(generator_cpu.device) == 'cpu'

        # Test GPU
        mock_cuda.is_available.return_value = True
        generator_gpu = TextToProteinGenerator()
        assert str(generator_gpu.device).startswith('cuda')

def test_memory_management(text_to_protein_generator):
    """Test memory management during generation"""
    with patch('torch.cuda') as mock_cuda:
        mock_cuda.is_available.return_value = True
        mock_cuda.empty_cache = Mock()

        text_to_protein_generator.generate_from_text("Generate a protein")
        mock_cuda.empty_cache.assert_called()

def test_generation_parameters(text_to_protein_generator):
    """Test different generation parameters"""
    description = "Generate a stable protein"

    # Test temperature effect
    seq_high_temp = text_to_protein_generator.generate_from_text(
        description,
        temperature=1.0
    )
    seq_low_temp = text_to_protein_generator.generate_from_text(
        description,
        temperature=0.1
    )
    assert isinstance(seq_high_temp, str)
    assert isinstance(seq_low_temp, str)

    # Test sampling parameters
    seq_with_params = text_to_protein_generator.generate_from_text(
        description,
        temperature=0.7,
        top_p=0.9,
        top_k=50
    )
    assert isinstance(seq_with_params, str)

def test_output_format(text_to_protein_generator):
    """Test output format consistency"""
    sequence = text_to_protein_generator.generate_from_text("Generate a protein")
    assert isinstance(sequence, str)
    assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)

    batch_sequences = text_to_protein_generator.generate_batch(
        ["Description 1", "Description 2"]
    )
    assert isinstance(batch_sequences, list)
    assert all(isinstance(seq, str) for seq in batch_sequences)
    assert all(all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq) for seq in batch_sequences)
