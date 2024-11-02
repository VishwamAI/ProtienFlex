import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from proteinflex.models.analysis.drug_discovery import DrugDiscoveryEngine

# Mock RDKit descriptors at module level
mock_logp = Mock(return_value=2.5)
mock_tpsa = Mock(return_value=60.0)
mock_volume = Mock(return_value=150.0)
mock_hbd = Mock(return_value=2)
mock_hba = Mock(return_value=3)
mock_mw = Mock(return_value=250.0)

# Create a mock molecule
mock_mol = Mock()
mock_mol.GetNumAtoms = Mock(return_value=20)

def mock_mol_from_smiles(smiles):
    """Mock MolFromSmiles to handle valid and invalid SMILES"""
    if smiles in ["CC1=CC=C(C=C1)C2=CN=CN=C2", "CC1=CC=C(C=C1)C2=CN=CN=C2"]:
        return mock_mol
    return None

# Apply patches at module level
patches = [
    patch('rdkit.Chem.Descriptors.MolLogP', mock_logp),
    patch('rdkit.Chem.Descriptors.TPSA', mock_tpsa),
    patch('rdkit.Chem.AllChem.ComputeMolVolume', mock_volume),
    patch('rdkit.Chem.Descriptors.NumHDonors', mock_hbd),
    patch('rdkit.Chem.Descriptors.NumHAcceptors', mock_hba),
    patch('rdkit.Chem.Descriptors.ExactMolWt', mock_mw),
    patch('rdkit.Chem.MolFromSmiles', mock_mol_from_smiles)
]

# Start all patches
for p in patches:
    p.start()

@pytest.fixture
def mock_esm_model():
    """Create a mock ESM model with proper tensor outputs"""
    class MockESMModel(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval_mode = False

        def eval(self):
            self.eval_mode = True
            return self

        def __call__(self, batch_tokens, repr_layers=None, **kwargs):
            if not isinstance(batch_tokens, torch.Tensor):
                raise ValueError("batch_tokens must be a tensor")
            seq_length = batch_tokens.shape[1]
            return {
                "representations": {
                    33: torch.randn(1, seq_length, 768)  # [batch_size, seq_length, hidden_size]
                },
                "attentions": [torch.randn(1, 12, seq_length, seq_length)],  # [batch_size, num_heads, seq_length, seq_length]
                "contacts": torch.randn(1, seq_length, seq_length)  # [batch_size, seq_length, seq_length]
            }

    return MockESMModel()

@pytest.fixture
def mock_alphabet():
    """Create mock ESM alphabet with batch converter"""
    alphabet = Mock()
    batch_converter = Mock()

    def convert_batch(data):
        """Properly convert batch data to tensors"""
        if data is None:
            raise ValueError("Input data cannot be None")

        batch_size = len(data)
        if batch_size == 0:
            raise ValueError("Input data cannot be empty")

        try:
            max_seq_len = max(len(seq) for _, seq in data)
            batch_tokens = torch.randint(0, 100, (batch_size, max_seq_len))  # Create proper tensor
            return [], [seq for _, seq in data], batch_tokens
        except (TypeError, AttributeError):
            raise ValueError("Invalid sequence format")

    batch_converter.side_effect = convert_batch
    alphabet.get_batch_converter.return_value = batch_converter
    return alphabet

@pytest.fixture
def drug_discovery_engine(mock_esm_model, mock_alphabet):
    """Create DrugDiscoveryEngine instance with mocked components"""
    with patch('proteinflex.models.analysis.drug_discovery.esm') as mock_esm:
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_esm_model, mock_alphabet)
        return DrugDiscoveryEngine()

@pytest.fixture
def sample_sequence():
    return "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNG"

@pytest.fixture
def sample_ligand_smiles():
    return "CC1=CC=C(C=C1)C2=CN=CN=C2"

def test_init_success(mock_esm_model, mock_alphabet):
    """Test successful initialization of DrugDiscoveryEngine"""
    with patch('proteinflex.models.analysis.drug_discovery.esm') as mock_esm:
        mock_esm.pretrained.esm2_t33_650M_UR50D.return_value = (mock_esm_model, mock_alphabet)
        engine = DrugDiscoveryEngine()
        assert isinstance(engine.esm_model, MagicMock)
        assert isinstance(engine.batch_converter, Mock)

def test_init_error():
    """Test error handling during initialization"""
    with patch('proteinflex.models.analysis.drug_discovery.esm') as mock_esm:
        mock_esm.pretrained.esm2_t33_650M_UR50D.side_effect = Exception("Model loading error")
        with pytest.raises(Exception):
            DrugDiscoveryEngine()

def test_analyze_binding_sites(drug_discovery_engine, sample_sequence):
    """Test binding site analysis"""
    binding_sites = drug_discovery_engine.analyze_binding_sites(sample_sequence)

    assert isinstance(binding_sites, list)
    if binding_sites:  # If sites found
        site = binding_sites[0]
        assert 'start' in site
        assert 'end' in site
        assert 'confidence' in site
        assert 'hydrophobicity' in site
        assert 'surface_area' in site
        assert 'volume' in site
        assert 'residues' in site

def test_predict_drug_interactions(drug_discovery_engine, sample_sequence, sample_ligand_smiles):
    """Test drug interaction prediction"""
    interactions = drug_discovery_engine.predict_drug_interactions(sample_sequence, sample_ligand_smiles)

    assert isinstance(interactions, dict)
    assert 'binding_affinity' in interactions
    assert 'stability_score' in interactions
    assert 'binding_energy' in interactions
    assert 'key_interactions' in interactions
    assert 'ligand_properties' in interactions

    # Test invalid SMILES
    with pytest.raises(ValueError):
        drug_discovery_engine.predict_drug_interactions(sample_sequence, "invalid_smiles")

def test_screen_off_targets(drug_discovery_engine, sample_sequence, sample_ligand_smiles):
    """Test off-target screening"""
    off_targets = drug_discovery_engine.screen_off_targets(sample_sequence, sample_ligand_smiles)

    assert isinstance(off_targets, list)
    if off_targets:  # If off-targets found
        target = off_targets[0]
        assert 'protein_family' in target
        assert 'similarity_score' in target
        assert 'risk_level' in target
        assert 'predicted_effects' in target
        assert 'confidence' in target

def test_optimize_binding_site(drug_discovery_engine, sample_sequence, sample_ligand_smiles):
    """Test binding site optimization"""
    with patch('rdkit.Chem.AllChem.ComputeMolVolume') as mock_compute_volume, \
         patch('rdkit.Chem.AllChem.EmbedMolecule') as mock_embed:

        # Mock volume computation and molecule embedding
        mock_compute_volume.return_value = 150.0
        mock_embed.return_value = 0  # Success

        result = drug_discovery_engine.optimize_binding_site(
            sample_sequence,
            site_start=0,
            site_end=10,
            ligand_smiles=sample_ligand_smiles
        )

        assert isinstance(result, dict)
        assert 'site_analysis' in result
        assert 'optimization_suggestions' in result
        assert 'optimization_score' in result
        assert 'predicted_improvement' in result

def test_binding_site_analysis_error_handling(drug_discovery_engine):
    """Test error handling in binding site analysis"""
    with pytest.raises(ValueError):
        drug_discovery_engine.analyze_binding_sites(None)

def test_drug_interactions_error_handling(drug_discovery_engine, sample_sequence):
    """Test error handling in drug interaction prediction"""
    with pytest.raises(ValueError):
        drug_discovery_engine.predict_drug_interactions(sample_sequence, "INVALID_SMILES")

@pytest.mark.parametrize("sequence,site_start,site_end", [
    ("MAEGEITTFT", 0, 5),    # Normal case
    ("A" * 100, 0, 10),      # Long sequence
    ("MAE", 0, 3),           # Short sequence
])
def test_optimize_binding_site_different_sizes(
    drug_discovery_engine, sample_ligand_smiles, sequence, site_start, site_end
):
    """Test binding site optimization with different sequence sizes"""
    with patch('rdkit.Chem.AllChem.ComputeMolVolume') as mock_compute_volume, \
         patch('rdkit.Chem.AllChem.EmbedMolecule') as mock_embed:

        # Mock volume computation and molecule embedding
        mock_compute_volume.return_value = 150.0
        mock_embed.return_value = 0  # Success

        result = drug_discovery_engine.optimize_binding_site(
            sequence, site_start, site_end, sample_ligand_smiles
        )

        assert isinstance(result, dict)
        assert 'site_analysis' in result

def test_rdkit_integration(drug_discovery_engine, sample_sequence):
    """Test RDKit integration for molecular analysis"""
    # Test with valid SMILES
    valid_smiles = "CC1=CC=C(C=C1)C2=CN=CN=C2"
    result = drug_discovery_engine.predict_drug_interactions(sample_sequence, valid_smiles)
    assert result['ligand_properties']['molecular_weight'] > 0
    assert isinstance(result['ligand_properties']['logp'], float)

    # Test with invalid SMILES
    with pytest.raises(ValueError):
        drug_discovery_engine.predict_drug_interactions(sample_sequence, "invalid")

@pytest.mark.parametrize("risk_level,expected_effects", [
    ("high", ["Strong binding potential"]),
    ("medium", ["Moderate binding possibility"]),
    ("low", ["Weak binding potential"])
])
def test_off_target_risk_levels(
    drug_discovery_engine, sample_sequence, sample_ligand_smiles, risk_level, expected_effects
):
    """Test off-target screening with different risk levels"""
    with patch.object(torch, 'sigmoid', return_value=torch.tensor(0.8)):
        off_targets = drug_discovery_engine.screen_off_targets(sample_sequence, sample_ligand_smiles)
        matching_targets = [t for t in off_targets if t['risk_level'] == risk_level]
        assert any(effect in target['predicted_effects']
                  for target in matching_targets
                  for effect in expected_effects)
