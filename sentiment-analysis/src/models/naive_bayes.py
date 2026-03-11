"""Naïve Bayes sentiment classifier with TF-IDF features."""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.evaluation.evaluator import evaluate


class NaiveBayesModel:
    """Sentiment classifier using Multinomial Naïve Bayes with TF-IDF."""

    def __init__(self, max_features: int = 10000, alpha: float = 1.0):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self.model = MultinomialNB(alpha=alpha)
        self.is_trained = False

    def train(
        self,
        texts: list[str],
        labels: list[int],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> dict:
        """Train the Naïve Bayes model and return evaluation metrics.

        Args:
            texts: List of preprocessed text strings.
            labels: List of integer labels (0=negative, 1=positive).
            test_size: Fraction of data to use for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Dict with accuracy, precision, recall, f1_score, report, confusion_matrix.
        """
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics['n_train'] = X_train.shape[0]
        metrics['n_test'] = X_test.shape[0]
        return metrics

    def predict(self, texts: list[str]) -> list[int]:
        """Predict sentiment for new texts."""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return class probabilities for new texts."""
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def update(self, texts: list[str], labels: list[int]) -> None:
        """Incrementally update classifier with feedback samples.

        Keeps the existing TF-IDF vocabulary — only updates the NB
        classifier weights via partial_fit. Very fast (no vectorizer refit).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling update().")
        X = self.vectorizer.transform(texts)
        self.model.partial_fit(X, np.array(labels), classes=[0, 1])

