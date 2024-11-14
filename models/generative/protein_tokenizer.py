import torch
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer

class ProteinTokenizer(PreTrainedTokenizer):
    """Tokenizer for protein sequences with special tokens and text-to-protein conversion."""

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )

        # Standard amino acids
        self.amino_acids = {
            'A': 'Alanine',
            'C': 'Cysteine',
            'D': 'Aspartic Acid',
            'E': 'Glutamic Acid',
            'F': 'Phenylalanine',
            'G': 'Glycine',
            'H': 'Histidine',
            'I': 'Isoleucine',
            'K': 'Lysine',
            'L': 'Leucine',
            'M': 'Methionine',
            'N': 'Asparagine',
            'P': 'Proline',
            'Q': 'Glutamine',
            'R': 'Arginine',
            'S': 'Serine',
            'T': 'Threonine',
            'V': 'Valine',
            'W': 'Tryptophan',
            'Y': 'Tyrosine',
        }

        # Build vocabulary
        self.vocab = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.cls_token: 2,
            self.sep_token: 3,
            self.mask_token: 4,
        }

        # Add amino acids to vocabulary
        for i, (aa, name) in enumerate(self.amino_acids.items()):
            self.vocab[aa] = i + 5  # Start after special tokens
            self.vocab[name.lower()] = i + 25  # Add full names

        # Create reverse mapping
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into subwords."""
        # Convert text to uppercase for amino acid matching
        text = text.upper()
        tokens = []

        i = 0
        while i < len(text):
            if text[i] in self.amino_acids:
                tokens.append(text[i])
                i += 1
            else:
                # Handle unknown characters
                tokens.append(self.unk_token)
                i += 1

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to vocabulary id."""
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert vocabulary id to token."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to protein sequence string."""
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence by adding special tokens."""
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where non-special tokens are 0s and special tokens are 1s."""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create token type IDs for sequence pairs."""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Save the tokenizer vocabulary to a file."""
        import os
        import json

        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)
