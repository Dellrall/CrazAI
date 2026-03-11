"""Model evaluation and comparison utilities."""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)


def evaluate(y_true, y_pred) -> dict:
    """Calculate standard classification metrics.

    Returns dict with accuracy, precision, recall, f1_score,
    classification report, and confusion matrix.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'report': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


class ModelEvaluator:
    """Compare and evaluate multiple sentiment models."""

    def __init__(self):
        self.results: dict[str, dict] = {}

    def add_result(self, model_name: str, metrics: dict):
        """Store evaluation results for a model."""
        self.results[model_name] = metrics

    def comparison_table(self) -> pd.DataFrame:
        """Generate a comparison table of all models."""
        rows = []
        for name, metrics in self.results.items():
            rows.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1_score']:.4f}"
            })
        return pd.DataFrame(rows)

    def plot_comparison(self, save_path: str = 'outputs/model_comparison.png'):
        """Bar chart comparing model performance."""
        import matplotlib.pyplot as plt
        df = self.comparison_table()
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(df))
        width = 0.2

        for i, metric in enumerate(metric_cols):
            values = [float(v) for v in df[metric]]
            ax.bar([xi + i * width for xi in x], values, width, label=metric)

        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Sentiment Analysis — Model Comparison')
        ax.set_xticks([xi + 1.5 * width for xi in x])
        ax.set_xticklabels(df['Model'])
        ax.legend()
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Comparison chart saved to {save_path}")

    def plot_confusion_matrices(self, save_path: str = 'outputs/confusion_matrices.png'):
        """Plot confusion matrix for each model side-by-side."""
        import matplotlib.pyplot as plt
        n = len(self.results)
        if n == 0:
            print("No results to plot.")
            return

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (name, metrics) in zip(axes, self.results.items()):
            ConfusionMatrixDisplay(
                confusion_matrix=metrics['confusion_matrix'],
                display_labels=['Negative', 'Positive']
            ).plot(ax=ax, cmap='Blues')
            ax.set_title(f'{name}')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Confusion matrices saved to {save_path}")
