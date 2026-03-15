"""Text cleaning utilities for preprocessing raw text data."""

import re


class TextCleaner:
    """Clean raw text: remove noise, special characters, HTML tags."""

    @staticmethod
    def clean(text: str) -> str:
        """Apply full cleaning pipeline to a text string.

        Steps:
            1. Remove HTML tags
            2. Remove URLs
            3. Remove @mentions and #hashtags
            4. Remove special characters and numbers
            5. Remove extra whitespace
            6. Lowercase
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters and numbers (keep apostrophes for contractions
        # like "don't", and keep clause-boundary punctuation for negation scope detection)
        text = re.sub(r"[^a-zA-Z'.,!?;:\s]", '', text)
        # Collapse stray apostrophes not inside words (e.g. lone ' at start/end)
        text = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Lowercase
        text = text.lower()
        return text
