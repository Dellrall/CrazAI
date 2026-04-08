"""Text cleaning utilities for preprocessing raw text data."""

import re


class TextCleaner:
    """Clean raw text: remove noise, special characters, HTML tags."""

    _ELONGATION_RE = re.compile(r'(.)\1{2,}')

    # Small, high-confidence idiom and slang rewrites. These normalize
    # colloquial phrases and emoticons before tokenization so the sentiment
    # model sees the intended polarity.
    _IDIOM_REWRITES = (
        (re.compile(r'\bbad[\s-]*ass\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\bbadass\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\blit\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\bfire\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\bsick\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\bdope\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\blol\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\blmao\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\brofl\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\bw00t\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\byay\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'\bxoxo\b', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'<3', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r':-?\)|;\-?\)|:d', flags=re.IGNORECASE), 'awesome'),
        (re.compile(r'(?::-?\(|:-?/|:\'\()', flags=re.IGNORECASE), 'awful'),
        (re.compile(r'\bwtf\b', flags=re.IGNORECASE), 'awful'),
        (re.compile(r'\bsmh\b', flags=re.IGNORECASE), 'awful'),
        (re.compile(r'\bugh\b', flags=re.IGNORECASE), 'awful'),
    )

    @staticmethod
    def _normalize_elongations(text: str) -> str:
        """Collapse exaggerated character repetitions into a stable form.

        Runs of 3+ repeated vowels are reduced to two characters so words like
        "goooood" become "good". Runs of 3+ repeated consonants are reduced to
        one character so words like "goooooddddddd" also normalize correctly.
        """

        def replace(match: re.Match[str]) -> str:
            char = match.group(1)
            if char.lower() in 'aeiouy':
                return char * 2
            return char

        return TextCleaner._ELONGATION_RE.sub(replace, text)

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
        # Normalize elongated spellings before idiom rewriting so repeated
        # letters like "goooooddddddd" collapse into a stable sentiment cue.
        text = TextCleaner._normalize_elongations(text)
        # Normalize a few high-confidence idioms before punctuation stripping so
        # slang like "bad ass" is treated as positive rather than negative.
        for pattern, replacement in TextCleaner._IDIOM_REWRITES:
            text = pattern.sub(replacement, text)
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
