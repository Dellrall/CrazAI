"""Online learning with incremental model updates and feedback persistence."""

import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer

# Get absolute paths
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / 'outputs'
FEEDBACK_DIR = BASE_DIR / 'data' / 'feedback'


class FeedbackCollector:
    """Collect, store, and retrieve user feedback corrections.

    Feedback is stored as JSONL (one JSON object per line) for easy
    append-only persistence without loading the entire file.
    """

    def __init__(self, feedback_path: str = None):
        if feedback_path is None:
            feedback_path = FEEDBACK_DIR / 'corrections.jsonl'
        self.feedback_path = Path(feedback_path).resolve()
        try:
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            pass  # Ignore on read-only filesystems

    def record(
        self,
        text: str,
        original_prediction: int,
        corrected_label: int,
        model_name: str = 'unknown',
        confidence: float = 0.0
    ):
        """Append a single correction to the feedback log.

        Args:
            text: The original input text.
            original_prediction: What the model predicted (0 or 1).
            corrected_label: The correct label provided by the user.
            model_name: Which model made the wrong prediction.
            confidence: Model confidence at time of prediction.
        """
        entry = {
            'text': text,
            'original_prediction': original_prediction,
            'corrected_label': corrected_label,
            'model_name': model_name,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(self.feedback_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except (OSError, PermissionError) as e:
            print(f"⚠️ Warning: Could not save feedback: {e}")

    def load_all(self) -> list[dict]:
        """Load all feedback entries from the JSONL file."""
        if not self.feedback_path.exists():
            return []
        try:
            entries = []
            with open(self.feedback_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            return entries
        except (OSError, PermissionError) as e:
            print(f"⚠️ Warning: Could not load feedback: {e}")
            return []

    def load_pending(self, min_count: int = 5) -> list[dict]:
        """Load feedback that hasn't been used for training yet.

        Simple heuristic: load the last N entries if total >= min_count.

        Args:
            min_count: Minimum entries before returning any (avoids
                       retraining on every single correction).

        Returns:
            List of feedback entries ready for model update.
        """
        entries = self.load_all()
        if len(entries) < min_count:
            return []
        return entries

    def count(self) -> int:
        """Return total number of feedback entries collected."""
        return len(self.load_all())

    def summary(self) -> dict:
        """Return a summary of collected feedback."""
        entries = self.load_all()
        if not entries:
            return {'total': 0}

        corrections = [e for e in entries if e['original_prediction'] != e['corrected_label']]
        pos_corrections = sum(1 for e in corrections if e['corrected_label'] == 1)
        neg_corrections = sum(1 for e in corrections if e['corrected_label'] == 0)

        return {
            'total': len(entries),
            'corrections': len(corrections),
            'correction_rate': len(corrections) / len(entries) if entries else 0,
            'pos_corrections': pos_corrections,
            'neg_corrections': neg_corrections
        }


class OnlineLearner:
    """Incremental sentiment model that improves from user feedback.

    Uses HashingVectorizer (stateless) + SGDClassifier (supports partial_fit)
    so the model can be updated one batch at a time without full retraining.
    """

    def __init__(self, model_path: str = None):
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
        if model_path is None:
            model_path = OUTPUT_DIR / 'online_model.pkl'
        self.model_path = Path(model_path).resolve()
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
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                'model': self.model,
                'history': self.update_history,
                'total_feedback': self.total_feedback,
                'is_initialized': self.is_initialized
            }, str(self.model_path))
        except (OSError, PermissionError) as e:
            print(f"⚠️ Warning: Could not save model: {e}")

    def load(self) -> bool:
        """Load previously saved model. Returns True if loaded."""
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.update_history = data['history']
                self.total_feedback = data['total_feedback']
                self.is_initialized = data['is_initialized']
                return True
            except (OSError, PermissionError) as e:
                print(f"⚠️ Warning: Could not load model: {e}")
                return False
        return False
