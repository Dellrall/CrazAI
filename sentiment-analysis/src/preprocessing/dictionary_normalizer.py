"""Dictionary-backed fuzzy word normalization utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from difflib import get_close_matches


class DictionaryNormalizer:
    """Map out-of-vocabulary words to the closest in-vocabulary word.

    This is intentionally conservative:
    - short tokens are left unchanged so slang abbreviations like "lol" survive
    - only alphabetic tokens are considered for correction
    - candidate matches must be close enough to avoid noisy substitutions
    """

    _DEFAULT_VOCAB_PATHS = (
        Path(__file__).resolve().parents[2] / 'data' / 'imdb' / 'aclImdb' / 'imdb.vocab',
        Path(__file__).resolve().parents[3] / 'data' / 'imdb' / 'aclImdb' / 'imdb.vocab',
    )

    def __init__(self, vocab_path: str | None = None, min_length: int = 4, cutoff: float = 0.84):
        self.min_length = min_length
        self.cutoff = cutoff
        self.vocab = self._load_vocab(vocab_path)

    @classmethod
    @lru_cache(maxsize=4)
    def _load_vocab(cls, vocab_path: str | None) -> set[str]:
        paths = []
        if vocab_path:
            paths.append(Path(vocab_path))
        paths.extend(cls._DEFAULT_VOCAB_PATHS)

        for path in paths:
            if path.exists():
                with path.open('r', encoding='utf-8') as handle:
                    return {line.strip().lower() for line in handle if line.strip()}

        return set()

    def normalize_token(self, token: str) -> str:
        """Return the closest dictionary word for an out-of-vocabulary token."""
        lower = token.lower()

        if len(lower) < self.min_length:
            return token
        if not lower.isalpha():
            return token
        if lower in self.vocab:
            return token

        matches = get_close_matches(lower, self.vocab, n=1, cutoff=self.cutoff)
        if not matches:
            return token

        match = matches[0]
        if token.isupper():
            return match.upper()
        if token[:1].isupper():
            return match.capitalize()
        return match

    def normalize_tokens(self, tokens: list[str]) -> list[str]:
        return [self.normalize_token(token) for token in tokens]