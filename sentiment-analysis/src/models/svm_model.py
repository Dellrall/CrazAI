"""SVM sentiment classifier with TF-IDF features and GridSearch tuning."""

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

from src.evaluation.evaluator import evaluate


class SVMModel:
    """Sentiment classifier using LinearSVC with TF-IDF features.

    Uses LinearSVC (faster than SVC for text) wrapped with
    CalibratedClassifierCV for probability estimates.
    """

    def __init__(self, max_features: int = 10000, C: float = 1.0):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
            ('svm', CalibratedClassifierCV(LinearSVC(C=C, max_iter=2000), cv=3))
        ])
        self.is_trained = False

    def train(
        self,
        texts: list[str],
        labels: list[int],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> dict:
        """Train the SVM model and return evaluation metrics.

        Args:
            texts: List of preprocessed text strings.
            labels: List of integer labels (0=negative, 1=positive).
            test_size: Fraction of data to use for testing.
            random_state: Random seed for reproducibility.

        Returns:
            Dict with accuracy, precision, recall, f1_score, report, confusion_matrix.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics['n_train'] = len(X_train)
        metrics['n_test'] = len(X_test)
        return metrics

    def train_with_tuning(
        self,
        texts: list[str],
        labels: list[int],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> dict:
        """Train SVM with hyperparameter tuning via GridSearchCV.

        Note: This is much slower than plain train() — use on a smaller
        subset first to find good parameters.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        param_grid = {
            'tfidf__max_features': [5000, 10000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'svm__estimator__C': [0.1, 1.0, 10.0],
        }

        grid_search = GridSearchCV(
            self.pipeline, param_grid, cv=3,
            scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        self.pipeline = grid_search.best_estimator_
        self.is_trained = True

        print(f"Best params: {grid_search.best_params_}")

        y_pred = self.pipeline.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics['n_train'] = len(X_train)
        metrics['n_test'] = len(X_test)
        metrics['best_params'] = grid_search.best_params_
        return metrics

    def predict(self, texts: list[str]) -> list[int]:
        """Predict sentiment for new texts."""
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return class probabilities for new texts."""
        return self.pipeline.predict_proba(texts)
