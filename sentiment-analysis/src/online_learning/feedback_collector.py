"""Feedback collection and persistence for the online learning loop."""

import json
import os
from pathlib import Path
from datetime import datetime

# Get absolute paths
BASE_DIR = Path(__file__).parent.parent.parent
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
