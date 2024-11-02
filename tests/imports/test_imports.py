import pytest

def test_core_imports():
    """Test that core modules can be imported."""
    from proteinflex.models.utils.esm_utils import ESMModel
    from proteinflex.models.generative import ProteinGenerator
    from proteinflex.models.analysis.mutation_analysis import MutationAnalyzer
    assert True

def test_analysis_imports():
    """Test analysis module imports."""
    from proteinflex.models.analysis.domain_analysis import DomainAnalyzer
    from proteinflex.models.analysis.drug_binding import DrugBindingAnalyzer
    from proteinflex.models.analysis.nlp_analysis import NLPAnalyzer
    assert True

def test_utils_imports():
    """Test utils module imports."""
    from proteinflex.models.utils.openmm_utils import OpenMMUtils
    assert True

def test_package_structure():
    """Test package structure."""
    import proteinflex
    assert hasattr(proteinflex.models, 'generative')
    assert hasattr(proteinflex.models, 'analysis')
    assert hasattr(proteinflex.models, 'utils')
    assert hasattr(proteinflex.models, 'dynamics')
