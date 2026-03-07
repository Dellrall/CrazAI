"""Feature extraction: Bag-of-Words, TF-IDF, and Word2Vec."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False


class FeatureExtractor:
    """Extract features from preprocessed text."""

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.bow_vectorizer = CountVectorizer(max_features=max_features)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

    def extract_bow(self, corpus: list[str]):
        """Bag-of-Words feature extraction.

        Args:
            corpus: List of preprocessed text strings.

        Returns:
            Sparse matrix of BoW features.
        """
        return self.bow_vectorizer.fit_transform(corpus)

    def transform_bow(self, corpus: list[str]):
        """Transform new data using fitted BoW vectorizer."""
        return self.bow_vectorizer.transform(corpus)

    def extract_tfidf(self, corpus: list[str]):
        """TF-IDF feature extraction.

        Args:
            corpus: List of preprocessed text strings.

        Returns:
            Sparse matrix of TF-IDF features.
        """
        return self.tfidf_vectorizer.fit_transform(corpus)

    def transform_tfidf(self, corpus: list[str]):
        """Transform new data using fitted TF-IDF vectorizer."""
        return self.tfidf_vectorizer.transform(corpus)

    @staticmethod
    def extract_word2vec(
        tokenized_corpus: list[list[str]],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2
    ) -> tuple:
        """Word2Vec embeddings — returns model and document vectors.

        Each document is represented as the average of its word vectors.

        Args:
            tokenized_corpus: List of tokenized documents (list of word lists).
            vector_size: Dimensionality of word vectors.
            window: Context window size.
            min_count: Minimum word frequency to include.

        Returns:
            Tuple of (Word2Vec model, document vectors as numpy array).
        """
        if not HAS_GENSIM:
            raise ImportError(
                "gensim is required for Word2Vec. Install with: pip install gensim"
            )

        model = Word2Vec(
            sentences=tokenized_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )

        # Average word vectors to create document vectors
        doc_vectors = []
        for tokens in tokenized_corpus:
            vectors = [model.wv[t] for t in tokens if t in model.wv]
            if vectors:
                doc_vectors.append(np.mean(vectors, axis=0))
            else:
                doc_vectors.append(np.zeros(vector_size))

        return model, np.array(doc_vectors)
