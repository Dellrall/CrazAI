"""Tokenization, stop word removal, stemming, and lemmatization."""

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data (safe to call multiple times)
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
    nltk.download(resource, quiet=True)


class TextTokenizer:
    """Tokenize text and apply stemming/lemmatization."""

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stop words from token list (case-insensitive)."""
        return [t for t in tokens if t.lower() not in self.stop_words]

    def stem(self, tokens: list[str]) -> list[str]:
        """Apply Porter Stemming."""
        return [self.stemmer.stem(t) for t in tokens]

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Apply WordNet Lemmatization."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text: str, use_lemma: bool = True) -> list[str]:
        """Full preprocessing pipeline: tokenize → remove stopwords → stem/lemmatize."""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if use_lemma:
            tokens = self.lemmatize(tokens)
        else:
            tokens = self.stem(tokens)
        return tokens
