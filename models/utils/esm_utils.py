class ESMWrapper:
    """Wrapper for ESM protein language model."""
    def __init__(self):
        pass

    def load_model(self):
        """Load ESM model and tokenizer."""
        return None, None

    def get_sequence_embeddings(self, sequence):
        """Get embeddings for protein sequence."""
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        return None

    def get_attention_maps(self, sequence):
        """Get attention maps for protein sequence."""
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        return None

    def process_attention_maps(self, attention_maps):
        """Process attention maps for analysis."""
        return None
