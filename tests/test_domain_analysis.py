import pytest
import numpy as np
from unittest.mock import Mock, patch
from proteinflex.models.analysis.domain_analysis import DomainAnalyzer

@pytest.fixture
def domain_analyzer():
    return DomainAnalyzer()

@pytest.fixture
def mock_sequence():
    return "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTDGLLYGSQTPNEECLFLERLEENHYNTYISKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV"

def test_init(domain_analyzer):
    """Test initialization of DomainAnalyzer"""
    assert isinstance(domain_analyzer.hydrophobicity_scale, dict)
    assert isinstance(domain_analyzer.active_site_patterns, dict)
    assert len(domain_analyzer.hydrophobicity_scale) == 20  # All amino acids
    assert 'catalytic_triad' in domain_analyzer.active_site_patterns

def test_calculate_hydrophobicity_profile(domain_analyzer):
    """Test hydrophobicity profile calculation"""
    sequence = "ARNDCQEGHILKMFPSTWYV"  # All 20 amino acids
    profile = domain_analyzer._calculate_hydrophobicity_profile(sequence, window_size=3)

    assert isinstance(profile, np.ndarray)
    assert len(profile) == len(sequence)
    assert not np.isnan(profile).any()

def test_identify_domains(domain_analyzer):
    """Test domain identification from hydrophobicity profile"""
    # Mock hydrophobicity profile with clear domains
    mock_profile = np.array([2.0, 2.0, 2.0, 0.0, -2.0, -2.0, -2.0, 0.0, 2.0, 2.0])
    domains = domain_analyzer._identify_domains(mock_profile, threshold=1.5)

    assert len(domains) == 3
    assert domains[0]['type'] == 'hydrophobic'
    assert domains[1]['type'] == 'hydrophilic'
    assert domains[2]['type'] == 'hydrophobic'

def test_find_active_sites(domain_analyzer):
    """Test active site pattern matching"""
    # Test sequence with known patterns
    sequence = "HDSAAPKKKRKVNXSAHEXXH"
    active_sites = domain_analyzer._find_active_sites(sequence)

    assert len(active_sites) > 0
    # Check for catalytic triad
    assert any(site['type'] == 'catalytic_triad' for site in active_sites)
    # Check for nuclear localization signal
    assert any(site['type'] == 'nuclear_localization' for site in active_sites)
    # Check for zinc binding motif
    assert any(site['type'] == 'zinc_binding' for site in active_sites)

def test_match_pattern(domain_analyzer):
    """Test pattern matching function"""
    assert domain_analyzer._match_pattern("HDS", "HDS")  # Exact match
    assert domain_analyzer._match_pattern("NAS", "NXS")  # With wildcard
    assert not domain_analyzer._match_pattern("ABC", "DEF")  # No match
    assert not domain_analyzer._match_pattern("AB", "ABC")  # Length mismatch

def test_generate_heatmap_data(domain_analyzer):
    """Test heatmap data generation"""
    sequence = "ARNDCQEGHILKMFPSTWYV"
    hydrophobicity = domain_analyzer._calculate_hydrophobicity_profile(sequence)
    domains = [{'start': 0, 'end': 5, 'type': 'hydrophobic'}]
    active_sites = [{'type': 'catalytic_triad', 'position': 0, 'length': 3}]
    conservation = np.ones(len(sequence))

    heatmap = domain_analyzer._generate_heatmap_data(
        sequence, hydrophobicity, domains, active_sites, conservation
    )

    assert isinstance(heatmap, list)
    assert len(heatmap) == 4  # Four tracks
    assert len(heatmap[0]) == len(sequence)  # Correct length

def test_generate_annotations(domain_analyzer):
    """Test annotation generation"""
    domains = [
        {'start': 0, 'end': 10, 'type': 'hydrophobic'},
        {'start': 15, 'end': 25, 'type': 'hydrophilic'}
    ]
    active_sites = [
        {'type': 'catalytic_triad', 'position': 5, 'length': 3}
    ]

    annotations = domain_analyzer._generate_annotations(domains, active_sites)

    assert isinstance(annotations, list)
    assert len(annotations) == 3  # Two domains + one active site
    assert annotations[0]['type'] == 'domain'
    assert annotations[2]['type'] == 'active_site'

def test_analyze_domains_integration(domain_analyzer, mock_sequence):
    """Test full domain analysis pipeline"""
    result = domain_analyzer.analyze_domains(mock_sequence)

    assert result is not None
    assert 'domains' in result
    assert 'active_sites' in result
    assert 'heatmap_data' in result
    assert 'annotations' in result

    # Verify structure of results
    assert isinstance(result['domains'], list)
    assert isinstance(result['active_sites'], list)
    assert isinstance(result['heatmap_data'], list)
    assert isinstance(result['annotations'], list)

def test_analyze_domains_error_handling(domain_analyzer):
    """Test error handling in analyze_domains"""
    # Test with invalid sequence
    result = domain_analyzer.analyze_domains(None)
    assert result is None

    # Test with empty sequence
    result = domain_analyzer.analyze_domains("")
    assert result is None

def test_conservation_scores(domain_analyzer, mock_sequence):
    """Test conservation score calculation"""
    scores = domain_analyzer._calculate_conservation_scores(mock_sequence)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == len(mock_sequence)
    assert np.all(scores >= 0) and np.all(scores <= 1)  # Scores should be normalized

def test_predict_domain_interactions(domain_analyzer):
    """Test prediction of domain interactions"""
    sequence = "ARNDCQEGHILKMFPSTWYV"
    interactions = domain_analyzer.predict_domain_interactions(sequence)

    assert isinstance(interactions, dict)
    assert 'interaction_matrix' in interactions
    assert 'domain_pairs' in interactions
    assert isinstance(interactions['interaction_matrix'], np.ndarray)
    assert isinstance(interactions['domain_pairs'], list)

def test_calculate_domain_stability(domain_analyzer):
    """Test calculation of domain stability"""
    sequence = "ARNDCQEGHILKMFPSTWYV"
    stability = domain_analyzer.calculate_domain_stability(sequence)

    assert isinstance(stability, dict)
    assert 'stability_scores' in stability
    assert 'average_stability' in stability
    assert isinstance(stability['stability_scores'], dict)
    assert isinstance(stability['average_stability'], float)

def test_identify_binding_sites(domain_analyzer):
    """Test identification of binding sites"""
    sequence = "ARNDCQEGHILKMFPSTWYV"
    binding_sites = domain_analyzer.identify_binding_sites(sequence)

    assert isinstance(binding_sites, list)
    assert all(isinstance(site, dict) for site in binding_sites)
    assert all('position' in site and 'type' in site for site in binding_sites)

def test_analyze_domain_flexibility(domain_analyzer):
    """Test analysis of domain flexibility"""
    sequence = "ARNDCQEGHILKMFPSTWYV"
    flexibility = domain_analyzer.analyze_domain_flexibility(sequence)

    assert isinstance(flexibility, dict)
    assert 'flexibility_profile' in flexibility
    assert 'flexible_regions' in flexibility
    assert isinstance(flexibility['flexibility_profile'], np.ndarray)
    assert isinstance(flexibility['flexible_regions'], list)

def test_edge_cases_and_errors(domain_analyzer):
    """Test edge cases and error handling"""
    # Test with invalid amino acids
    with pytest.raises(ValueError):
        domain_analyzer.analyze_domains("ABC123")

    # Test with sequence too short
    with pytest.raises(ValueError):
        domain_analyzer.analyze_domains("A")

    # Test with non-string input
    with pytest.raises(TypeError):
        domain_analyzer.analyze_domains(123)

    # Test with None input
    with pytest.raises(TypeError):
        domain_analyzer.analyze_domains(None)

def test_domain_boundary_detection(domain_analyzer):
    """Test detection of domain boundaries"""
    sequence = "ARNDCQEGHILKMFPSTWYV"
    boundaries = domain_analyzer.detect_domain_boundaries(sequence)

    assert isinstance(boundaries, list)
    assert all(isinstance(boundary, dict) for boundary in boundaries)
    assert all('start' in b and 'end' in b and 'confidence' in b for b in boundaries)
    assert all(isinstance(b['confidence'], float) for b in boundaries)

def test_domain_conservation_analysis(domain_analyzer):
    """Test analysis of domain conservation"""
    sequences = ["ARNDCQEGHILKMFPSTWYV", "ARNDCQEGHILKMFPSTWYV"]
    conservation = domain_analyzer.analyze_domain_conservation(sequences)

    assert isinstance(conservation, dict)
    assert 'conservation_scores' in conservation
    assert 'conserved_regions' in conservation
    assert isinstance(conservation['conservation_scores'], np.ndarray)
    assert isinstance(conservation['conserved_regions'], list)

def test_domain_interaction_network(domain_analyzer):
    """Test generation of domain interaction network"""
    sequence = "ARNDCQEGHILKMFPSTWYV"
    network = domain_analyzer.generate_interaction_network(sequence)

    assert isinstance(network, dict)
    assert 'nodes' in network
    assert 'edges' in network
    assert isinstance(network['nodes'], list)
    assert isinstance(network['edges'], list)
    assert all('weight' in edge for edge in network['edges'])
