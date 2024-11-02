import pytest
import torch
from unittest.mock import Mock, patch
from proteinflex.models.generative.protein_generator import ProteinGenerator, ProteinGenerativeConfig

@pytest.fixture
def mock_transformer():
    model = Mock()
    model.eval.return_value = model
    model.forward = Mock(return_value={
        'logits': torch.randn(1, 10, 20),  # batch_size, seq_len, vocab_size
        'hidden_states': torch.randn(1, 10, 768),
        'attentions': torch.randn(1, 8, 10, 10)
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
def protein_generator(mock_transformer, mock_tokenizer):
    with patch('proteinflex.models.generative.protein_generator.AutoModel') as mock_auto_model, \
         patch('proteinflex.models.generative.protein_generator.AutoTokenizer') as mock_auto_tokenizer:
        mock_auto_model.from_pretrained.return_value = mock_transformer
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        config = ProteinGenerativeConfig(
            vocab_size=30,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8
        )
        return ProteinGenerator(config)

def test_init(mock_transformer, mock_tokenizer):
    """Test initialization of ProteinGenerator"""
    with patch('proteinflex.models.generative.protein_generator.AutoModel') as mock_auto_model, \
         patch('proteinflex.models.generative.protein_generator.AutoTokenizer') as mock_auto_tokenizer:
        mock_auto_model.from_pretrained.return_value = mock_transformer
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        config = ProteinGenerativeConfig()
        generator = ProteinGenerator(config)
        assert generator.model == mock_transformer
        assert generator.tokenizer == mock_tokenizer
        assert str(generator.device) in ['cpu', 'cuda:0']

def test_generate_sequence(protein_generator):
    """Test protein sequence generation"""
    sequence = protein_generator.generate_sequence(
        prompt="Design a stable protein that",
        max_length=50
    )

    assert isinstance(sequence, str)
    assert len(sequence) > 0
    assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)

def test_generate_with_constraints(protein_generator):
    """Test constrained sequence generation"""
    constraints = {
        'length': 100,
        'hydrophobicity': 'high',
        'secondary_structure': 'alpha_helix'
    }

    sequence = protein_generator.generate_with_constraints(constraints)

    assert isinstance(sequence, str)
    assert len(sequence) == constraints['length']
    assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)

def test_batch_generation(protein_generator):
    """Test batch sequence generation"""
    prompts = [
        "Design a stable protein that",
        "Generate a membrane protein with"
    ]

    sequences = protein_generator.generate_batch(prompts, max_length=50)

    assert isinstance(sequences, list)
    assert len(sequences) == len(prompts)
    assert all(isinstance(seq, str) for seq in sequences)
    assert all(len(seq) > 0 for seq in sequences)

def test_conditional_generation(protein_generator):
    """Test conditional sequence generation"""
    conditions = {
        'domain': 'kinase',
        'organism': 'human',
        'function': 'ATP binding'
    }

    sequence = protein_generator.generate_conditional(conditions, max_length=200)

    assert isinstance(sequence, str)
    assert len(sequence) > 0
    assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)

def test_error_handling(protein_generator):
    """Test error handling"""
    # Test with invalid prompt
    with pytest.raises(ValueError):
        protein_generator.generate_sequence("")

    # Test with invalid constraints
    with pytest.raises(ValueError):
        protein_generator.generate_with_constraints({'length': -1})

    # Test with invalid conditions
    with pytest.raises(ValueError):
        protein_generator.generate_conditional({'invalid_key': 'value'})

def test_sequence_validation(protein_generator):
    """Test sequence validation"""
    # Test valid sequence
    assert protein_generator._validate_sequence("MAEGEITT")

    # Test invalid sequences
    assert not protein_generator._validate_sequence("")
    assert not protein_generator._validate_sequence("XYZ123")
    assert not protein_generator._validate_sequence(None)

def test_device_handling():
    """Test device handling (CPU/GPU)"""
    with patch('torch.cuda') as mock_cuda, \
         patch('proteinflex.models.generative.protein_generator.AutoModel') as mock_auto_model, \
         patch('proteinflex.models.generative.protein_generator.AutoTokenizer') as mock_auto_tokenizer:

        # Test CPU
        mock_cuda.is_available.return_value = False
        config = ProteinGenerativeConfig(
            vocab_size=30,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8
        )
        generator_cpu = ProteinGenerator(config)
        assert str(generator_cpu.device) == 'cpu'

        # Test GPU
        mock_cuda.is_available.return_value = True
        config = ProteinGenerativeConfig(
            vocab_size=30,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8
        )
        generator_gpu = ProteinGenerator(config)
        assert str(generator_gpu.device).startswith('cuda')

def test_memory_management():
    """Test memory management during generation"""
    with patch('torch.cuda') as mock_cuda, \
         patch('proteinflex.models.generative.protein_generator.AutoModel') as mock_auto_model, \
         patch('proteinflex.models.generative.protein_generator.AutoTokenizer') as mock_auto_tokenizer:
        mock_cuda.is_available.return_value = True
        mock_cuda.empty_cache = Mock()

        config = ProteinGenerativeConfig(
            vocab_size=30,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8
        )
        generator = ProteinGenerator(config)
        generator.generate_sequence("Design a protein")
        mock_cuda.empty_cache.assert_called()

def test_generation_parameters():
    """Test different generation parameters"""
    with patch('proteinflex.models.generative.protein_generator.AutoModel') as mock_auto_model, \
         patch('proteinflex.models.generative.protein_generator.AutoTokenizer') as mock_auto_tokenizer:
        config = ProteinGenerativeConfig(
            vocab_size=30,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8
        )
        generator = ProteinGenerator(config)

        # Test temperature effect
        seq_high_temp = generator.generate_sequence(
            "Design a protein",
            temperature=1.0
        )
        seq_low_temp = generator.generate_sequence(
            "Design a protein",
            temperature=0.1
        )
        assert isinstance(seq_high_temp, str)
        assert isinstance(seq_low_temp, str)

        # Test top_p effect
        seq_high_p = generator.generate_sequence(
            "Design a protein",
            top_p=0.9
        )
        seq_low_p = generator.generate_sequence(
            "Design a protein",
            top_p=0.1
        )
        assert isinstance(seq_high_p, str)
        assert isinstance(seq_low_p, str)

def test_output_format():
    """Test output format consistency"""
    with patch('proteinflex.models.generative.protein_generator.AutoModel') as mock_auto_model, \
         patch('proteinflex.models.generative.protein_generator.AutoTokenizer') as mock_auto_tokenizer:
        config = ProteinGenerativeConfig(
            vocab_size=30,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8
        )
        generator = ProteinGenerator(config)
        sequence = generator.generate_sequence("Design a protein")
        assert isinstance(sequence, str)
        assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)

        batch_sequences = generator.generate_batch(["Prompt 1", "Prompt 2"])
        assert isinstance(batch_sequences, list)
        assert all(isinstance(seq, str) for seq in batch_sequences)
        assert all(all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq) for seq in batch_sequences)
