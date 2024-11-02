"""ProteinFlex package initialization."""
# Define version
__version__ = '0.1.0'

# Define available components
__all__ = [
    'MutationAnalyzer',
    'DomainAnalyzer',
    'DrugBindingAnalyzer',
    'DrugDiscoveryPipeline',
    'NLPAnalyzer',
    'QASystem',
    'MolecularDynamics',
    'ProteinGenerator',
    'ProteinGenerativeConfig',
    'TextToProteinGenerator',
    'ESMModel',
    'OpenMMUtils',
]

# Lazy imports to avoid early dependency loading
def __getattr__(name):
    if name in __all__:
        if name in ['MutationAnalyzer']:
            from proteinflex.models.analysis.mutation_analysis import MutationAnalyzer
            return MutationAnalyzer
        elif name in ['DomainAnalyzer']:
            from proteinflex.models.analysis.domain_analysis import DomainAnalyzer
            return DomainAnalyzer
        elif name in ['DrugBindingAnalyzer']:
            from proteinflex.models.analysis.drug_binding import DrugBindingAnalyzer
            return DrugBindingAnalyzer
        elif name in ['DrugDiscoveryPipeline']:
            from proteinflex.models.analysis.drug_discovery import DrugDiscoveryPipeline
            return DrugDiscoveryPipeline
        elif name in ['NLPAnalyzer']:
            from proteinflex.models.analysis.nlp_analysis import NLPAnalyzer
            return NLPAnalyzer
        elif name in ['QASystem']:
            from proteinflex.models.analysis.qa_system import QASystem
            return QASystem
        elif name in ['MolecularDynamics']:
            from proteinflex.models.dynamics.simulation import MolecularDynamics
            return MolecularDynamics
        elif name in ['ProteinGenerator', 'ProteinGenerativeConfig']:
            from proteinflex.models.generative.protein_generator import ProteinGenerator, ProteinGenerativeConfig
            return ProteinGenerator if name == 'ProteinGenerator' else ProteinGenerativeConfig
        elif name in ['TextToProteinGenerator']:
            from proteinflex.models.generative.text_to_protein_generator import TextToProteinGenerator
            return TextToProteinGenerator
        elif name in ['ESMModel']:
            from proteinflex.models.utils.esm_utils import ESMModel
            return ESMModel
        elif name in ['OpenMMUtils']:
            from proteinflex.models.utils.openmm_utils import OpenMMUtils
            return OpenMMUtils
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
