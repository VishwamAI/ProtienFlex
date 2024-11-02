from proteinflex.models.utils.lazy_imports import numpy, torch, openmm
"""High-level interface for protein generation."""
from typing import Optional
from .protein_generator import ProteinGenerator as BaseProteinGenerator, ProteinGenerativeConfig

class ProteinGenerator:
    """High-level interface for protein generation"""
    def __init__(self, config: Optional[ProteinGenerativeConfig] = None):
        """Initialize the protein generator.

        Args:
            config: Configuration for the generative model. If None, uses default config.
        """
        if config is None:
            config = ProteinGenerativeConfig()
        self.model = BaseProteinGenerator(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def generate_sequence(self, prompt: str, max_length: int = 512) -> str:
        """Generate a protein sequence from a text prompt.

        Args:
            prompt: Text description of the desired protein
            max_length: Maximum length of the generated sequence

        Returns:
            Generated protein sequence as a string
        """
        input_ids = torch.tensor([[self.model.aa_to_idx.get(aa, 0) for aa in prompt]])
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                pad_token_id=self.model.config.pad_token_id,
                eos_token_id=None,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )

        sequence = ''.join([self.model.idx_to_aa.get(idx.item(), 'X')
                          for idx in output[0]])
        return sequence

    def save_model(self, path: str):
        """Save the model to disk."""
        self.model.save_pretrained(path)

    @classmethod
    def load_model(cls, path: str) -> 'ProteinGenerator':
        """Load a model from disk."""
        config = ProteinGenerativeConfig.from_pretrained(path)
        generator = cls(config)
        generator.model = BaseProteinGenerator.from_pretrained(path)
        generator.model.to(generator.device)
        return generator
