"""Tokenization, stop word removal, negation tagging, stemming, and lemmatization."""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data (safe to call multiple times)
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(resource, quiet=True)

# Words that signal negation scope.  We keep these OUT of the stop-word filter
# so they survive long enough to be used by mark_negations().
_NEGATION_WORDS = {
    # Base negation words
    "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
    "hardly", "scarcely", "barely", "without",
    # Contractions WITH apostrophe (tokenized as a separate n't token)
    "n't",
    # Contractions WITHOUT apostrophe — casual/informal writing
    "dont", "doesnt", "didnt", "wont", "wouldnt", "shouldnt", "couldnt",
    "cant", "cannot", "isnt", "arent", "wasnt", "werent", "hasnt", "havent",
    "hadnt", "aint", "neednt", "mustnt",
}

# Punctuation that ends a negation scope (resets the _NEG flag)
_SCOPE_ENDERS = re.compile(r'^[.!?,;:]$')


class TextTokenizer:
    """Tokenize text and apply optional negation tagging, stemming/lemmatization."""

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        # Build a stop-word set that does NOT include negation words, so they
        # are kept when we call remove_stopwords() during preprocessing.
        raw_stops = set(stopwords.words('english'))
        self.stop_words = raw_stops - _NEGATION_WORDS

    # ──────────────────────────────────────────────────
    # Core helpers
    # ──────────────────────────────────────────────────

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stop words from token list (case-insensitive).

        Negation words (not, no, never, n't …) are intentionally kept so that
        mark_negations() can operate on them downstream.
        """
        return [t for t in tokens if t.lower() not in self.stop_words]

    def mark_negations(self, tokens: list[str]) -> list[str]:
        """Tag words that follow a negation trigger with a ``_NEG`` suffix.

        Negation scope starts immediately after a negation trigger word
        (e.g. "not", "never", "n't") and ends at the next clause boundary
        (punctuation: . ! ? , ; :) or at the end of the token list.

        Examples
        --------
        >>> mark_negations(['this', 'movie', 'is', 'not', 'bad'])
        ['this', 'movie', 'is', 'not', 'bad_NEG']

        >>> mark_negations(['good', 'at', 'not', 'making', 'bad', 'movie'])
        ['good', 'at', 'not', 'making_NEG', 'bad_NEG', 'movie_NEG']
        """
        result: list[str] = []
        in_negation = False
        neg_count = 0          # how many words tagged in the current scope
        NEG_SCOPE_LIMIT = 5    # maximum words to tag per negation trigger

        for token in tokens:
            lower = token.lower()

            # n't is always a negation trigger regardless of capitalisation
            is_negation_trigger = (
                lower in _NEGATION_WORDS or lower == "n't"
            )

            if is_negation_trigger:
                # Append the trigger itself unchanged; enable negation scope
                result.append(token)
                in_negation = True
                neg_count = 0
            elif _SCOPE_ENDERS.match(token):
                # Punctuation ends the negation scope
                result.append(token)
                in_negation = False
                neg_count = 0
            elif in_negation:
                result.append(token + '_NEG')
                neg_count += 1
                if neg_count >= NEG_SCOPE_LIMIT:
                    in_negation = False  # scope exhausted
                    neg_count = 0
            else:
                result.append(token)

        return result

    def stem(self, tokens: list[str]) -> list[str]:
        """Apply Porter Stemming (preserves ``_NEG`` suffixes)."""
        out = []
        for t in tokens:
            if t.endswith('_NEG'):
                out.append(self.stemmer.stem(t[:-4]) + '_NEG')
            else:
                out.append(self.stemmer.stem(t))
        return out

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Apply WordNet Lemmatization (preserves ``_NEG`` suffixes)."""
        out = []
        for t in tokens:
            if t.endswith('_NEG'):
                out.append(self.lemmatizer.lemmatize(t[:-4]) + '_NEG')
            else:
                out.append(self.lemmatizer.lemmatize(t))
        return out

    # ──────────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────────

    def preprocess(self, text: str, use_lemma: bool = True) -> list[str]:
        """Full preprocessing pipeline.

        Order: tokenize → negation tagging → remove stopwords → stem/lemmatize.

        Negation tagging MUST happen before stop-word removal so that the
        negation trigger words are still present when we scan for them.
        After tagging, the triggers (not, never …) are removed as stop words
        unless they are immediately needed — in practice keeping them is fine
        too, so we leave them in to aid interpretability.
        """
        tokens = self.tokenize(text)
        tokens = self.mark_negations(tokens)
        tokens = self.remove_stopwords(tokens)
        if use_lemma:
            tokens = self.lemmatize(tokens)
        else:
            tokens = self.stem(tokens)
        return tokens
