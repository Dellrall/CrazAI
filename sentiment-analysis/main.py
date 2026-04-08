"""
Sentiment Analysis — Main Pipeline Entry Point
================================================

Runs the complete NLP pipeline:
  Week 3-4:   Data loading, cleaning, preprocessing
  Week 5-6:   Naïve Bayes training & evaluation
  Week 7-8:   SVM training & evaluation
  Final:      Model comparison & report

Usage:
    # Full pipeline (Naïve Bayes + SVM)
    python main.py

    # Naïve Bayes only
    python main.py --nb-only
"""

import argparse
import os
import sys
import time
from functools import wraps
from pathlib import Path

# Get absolute paths
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / 'outputs'
DATA_DIR = BASE_DIR / 'data'

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from src.crawler.dataset_loader import DatasetLoader
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tokenizer import TextTokenizer
from src.evaluation.evaluator import ModelEvaluator


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] completed in {elapsed:.2f}s")
        return result
    return wrapper


def print_header(title: str, width: int = 60):
    """Print a formatted section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ─────────────────────────────────────────────
# Data pipeline
# ─────────────────────────────────────────────

@timer
def load_and_preprocess(sample_size: int = None) -> tuple:
    """Load IMDB, clean, and tokenize.

    Args:
        sample_size: Limit dataset to this many rows (None = all 50K).
                     Use e.g. 5000 for quick testing.

    Returns:
        (df, texts, labels) where texts are preprocessed joined strings.
    """
    # Load
    df = DatasetLoader.load_imdb(str(DATA_DIR / 'imdb'))
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"  Sampled {len(df)} rows for quick run")

    # Clean
    cleaner = TextCleaner()
    df['cleaned'] = df['text'].apply(cleaner.clean)

    # Tokenize
    tokenizer = TextTokenizer()
    df['tokens'] = df['cleaned'].apply(tokenizer.preprocess)
    df['processed'] = df['tokens'].apply(lambda t: ' '.join(t))

    texts = df['processed'].tolist()
    labels = df['label'].tolist()

    print(f"  Total samples: {len(df)}")
    pos = sum(labels)
    print(f"  Positive: {pos} ({pos/len(labels):.1%})  Negative: {len(labels)-pos} ({1-pos/len(labels):.1%})")

    return df, texts, labels


# ─────────────────────────────────────────────
# Week 5–6: Naïve Bayes
# ─────────────────────────────────────────────

@timer
def run_naive_bayes(texts: list, labels: list) -> dict:
    from src.models.naive_bayes import NaiveBayesModel
    print(f"  Training on {int(len(texts)*0.8):,} samples, testing on {int(len(texts)*0.2):,}")
    nb = NaiveBayesModel(max_features=10000)
    metrics = nb.train(texts, labels)
    return metrics, nb


# ─────────────────────────────────────────────
# Week 7–8: SVM
# ─────────────────────────────────────────────

@timer
def run_svm(texts: list, labels: list) -> dict:
    from src.models.svm_model import SVMModel
    print(f"  Training on {int(len(texts)*0.8):,} samples, testing on {int(len(texts)*0.2):,}")
    svm = SVMModel(max_features=10000)
    metrics = svm.train(texts, labels)
    return metrics, svm


# ─────────────────────────────────────────────
# Comparison
# ─────────────────────────────────────────────

def print_comparison(evaluator: ModelEvaluator):
    """Print and save the model comparison."""
    df = evaluator.comparison_table()
    print("\n" + df.to_string(index=False))

    evaluator.plot_comparison(str(OUTPUT_DIR / 'model_comparison.png'))
    evaluator.plot_confusion_matrices(str(OUTPUT_DIR / 'confusion_matrices.png'))

    # Save CSV
    df.to_csv(OUTPUT_DIR / 'results.csv', index=False)
    print("\n  Results saved to outputs/results.csv")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline')
    parser.add_argument('--nb-only', action='store_true',
                        help='Only run Naïve Bayes')
    parser.add_argument('--sample', type=int, default=None,
                        help='Limit dataset to N samples (e.g. --sample 5000)')
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir('outputs')
    evaluator = ModelEvaluator()

    # ── Step 1: Data ──────────────────────────────
    print_header("STEP 1: LOADING & PREPROCESSING DATA")
    df, texts, labels = load_and_preprocess(sample_size=args.sample)

    # ── Step 2: Naïve Bayes ───────────────────────
    print_header("STEP 2: NAÏVE BAYES (Week 5–6)")
    nb_metrics, nb_model = run_naive_bayes(texts, labels)
    evaluator.add_result('Naïve Bayes', nb_metrics)
    print(f"\n{nb_metrics['report']}")

    if args.nb_only:
        print_header("RESULTS (NB only)")
        print_comparison(evaluator)
        return

    # ── Step 3: SVM ───────────────────────────────
    print_header("STEP 3: SVM (Week 7–8)")
    svm_metrics, svm_model = run_svm(texts, labels)
    evaluator.add_result('SVM', svm_metrics)
    print(f"\n{svm_metrics['report']}")

    # ── Step 4: Comparison ────────────────────────
    print_header("FINAL COMPARISON")
    print_comparison(evaluator)

    print_header("DONE")
    print("  Charts saved to outputs/")
    print("  Next: streamlit run app_feedback.py")


if __name__ == '__main__':
    main()
