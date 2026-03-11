"""Online learning with incremental model updates using partial_fit."""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer


class OnlineLearner:
    """Incremental sentiment model that improves from user feedback.

    Uses HashingVectorizer (stateless) + SGDClassifier (supports partial_fit)
    so the model can be updated one batch at a time without full retraining.
    """

    def __init__(self, model_path: str = 'outputs/online_model.pkl'):
        # HashingVectorizer: stateless, no need to fit - works for online learning
        self.vectorizer = HashingVectorizer(
            n_features=2 ** 17,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm='l2'
        )
        # SGDClassifier: supports incremental partial_fit
        self.model = SGDClassifier(
            loss='log_loss',        # gives probability estimates
            penalty='l2',
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            warm_start=True
        )
        self.model_path = model_path
        self.is_initialized = False
        self.update_history: list[dict] = []
        self.total_feedback = 0

    def initialize(self, texts: list[str], labels: list[int]):
        """Seed the online model with the initial training data."""
        X = self.vectorizer.transform(texts)
        self.model.partial_fit(X, labels, classes=[0, 1])
        self.is_initialized = True
        self.update_history.append({
            'type': 'initial_training',
            'n_samples': len(texts),
            'timestamp': datetime.now().isoformat()
        })

    def update(self, texts: list[str], labels: list[int]) -> dict:
        """Incrementally update the model with new feedback samples.

        Args:
            texts: Preprocessed text strings.
            labels: Corrected labels (0=negative, 1=positive).

        Returns:
            Dict with update info.
        """
        if not self.is_initialized:
            raise RuntimeError("Call initialize() before update().")

        X = self.vectorizer.transform(texts)
        self.model.partial_fit(X, labels, classes=[0, 1])
        self.total_feedback += len(texts)

        record = {
            'type': 'feedback_update',
            'n_samples': len(texts),
            'total_feedback': self.total_feedback,
            'timestamp': datetime.now().isoformat()
        }
        self.update_history.append(record)
        self.save()
        return record

    def predict(self, texts: list[str]) -> list[int]:
        """Predict sentiment for new texts."""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return class probabilities [P(negative), P(positive)]."""
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def predict_with_confidence(self, text: str) -> tuple[int, float, bool]:
        """Predict and flag if confidence is low (uncertain prediction).

        Returns:
            (label, confidence, is_uncertain) where is_uncertain=True
            means confidence < 0.7 — good candidate for user feedback.
        """
        probs = self.predict_proba([text])[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
        return pred, conf, conf < 0.7

    def save(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'history': self.update_history,
            'total_feedback': self.total_feedback,
            'is_initialized': self.is_initialized
        }, self.model_path)

    def load(self) -> bool:
        """Load previously saved model. Returns True if loaded."""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.update_history = data['history']
            self.total_feedback = data['total_feedback']
            self.is_initialized = data['is_initialized']
            return True
        return False
