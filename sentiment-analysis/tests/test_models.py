"""Unit tests for all sentiment analysis models."""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tokenizer import TextTokenizer
from src.preprocessing.feature_extractor import FeatureExtractor
from src.evaluation.evaluator import evaluate


# ── Fixtures ─────────────────────────────────────────────────────────────────

SMALL_CORPUS = [
    "this movie was absolutely fantastic I loved every second of it",
    "terrible film waste of time money avoid at all costs",
    "great performances brilliant story highly recommend watching",
    "boring slow dull completely disappointing total letdown",
    "amazing cinematography emotional powerful masterpiece",
    "awful acting bad script horrible ending worst movie ever",
    "wonderful heartwarming feel good movie perfect family film",
    "dreadful nonsensical plot confusing characters not worth watching",
]
SMALL_LABELS = [1, 0, 1, 0, 1, 0, 1, 0]


# ── Preprocessing Tests ───────────────────────────────────────────────────────

class TestTextCleaner(unittest.TestCase):

    def setUp(self):
        self.cleaner = TextCleaner()

    def test_removes_html_tags(self):
        result = self.cleaner.clean('<p>Hello <b>world</b></p>')
        self.assertNotIn('<', result)
        self.assertNotIn('>', result)

    def test_removes_urls(self):
        result = self.cleaner.clean('Check http://example.com for more')
        self.assertNotIn('http', result)

    def test_removes_mentions_hashtags(self):
        result = self.cleaner.clean('@user loved it #greatmovie')
        self.assertNotIn('@', result)
        self.assertNotIn('#', result)

    def test_lowercases(self):
        result = self.cleaner.clean('GREAT Movie EVER')
        self.assertEqual(result, result.lower())

    def test_removes_numbers(self):
        result = self.cleaner.clean('rated 9/10 stars')
        self.assertNotIn('9', result)
        self.assertNotIn('10', result)

    def test_empty_string(self):
        result = self.cleaner.clean('')
        self.assertEqual(result, '')


class TestTextTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = TextTokenizer()

    def test_tokenize_returns_list(self):
        tokens = self.tokenizer.tokenize('hello world')
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_remove_stopwords(self):
        tokens = ['the', 'movie', 'is', 'great']
        result = self.tokenizer.remove_stopwords(tokens)
        self.assertNotIn('the', result)
        self.assertNotIn('is', result)
        self.assertIn('great', result)

    def test_lemmatize(self):
        tokens = ['running', 'movies', 'beautiful']
        result = self.tokenizer.lemmatize(tokens)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_preprocess_pipeline(self):
        tokens = self.tokenizer.preprocess('The movies were absolutely fantastic')
        self.assertIsInstance(tokens, list)
        lower_tokens = [t.lower() for t in tokens]
        # Stop words should be removed
        self.assertNotIn('the', lower_tokens)
        self.assertNotIn('were', lower_tokens)

    def test_preprocess_handles_negation_flip(self):
        positive_tokens = self.tokenizer.preprocess('This movie is not bad')
        negative_tokens = self.tokenizer.preprocess('This movie is not good')

        self.assertIn('bad_NEG', positive_tokens)
        self.assertIn('good', positive_tokens)

        self.assertIn('good_NEG', negative_tokens)
        self.assertIn('bad', negative_tokens)


class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = FeatureExtractor(max_features=50)
        self.corpus = [
            'the movie was great',
            'terrible film waste',
            'loved every minute'
        ]

    def test_tfidf_shape(self):
        matrix = self.extractor.extract_tfidf(self.corpus)
        self.assertEqual(matrix.shape[0], 3)
        self.assertLessEqual(matrix.shape[1], 50)

    def test_bow_shape(self):
        matrix = self.extractor.extract_bow(self.corpus)
        self.assertEqual(matrix.shape[0], 3)

    def test_tfidf_transform_same_shape(self):
        self.extractor.extract_tfidf(self.corpus)
        new_text = ['new movie review here']
        transformed = self.extractor.transform_tfidf(new_text)
        self.assertEqual(transformed.shape[0], 1)


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestNaiveBayes(unittest.TestCase):

    def setUp(self):
        from src.models.naive_bayes import NaiveBayesModel
        self.model = NaiveBayesModel(max_features=100)

    def test_train_returns_metrics(self):
        metrics = self.model.train(SMALL_CORPUS, SMALL_LABELS, test_size=0.25)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)

    def test_predict_returns_list(self):
        self.model.train(SMALL_CORPUS, SMALL_LABELS, test_size=0.25)
        preds = self.model.predict(['great movie loved it', 'terrible waste awful'])
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 2)
        self.assertTrue(all(p in [0, 1] for p in preds))

    def test_predict_proba_shape(self):
        self.model.train(SMALL_CORPUS, SMALL_LABELS, test_size=0.25)
        probs = self.model.predict_proba(['great film'])
        self.assertEqual(probs.shape, (1, 2))
        # Probabilities should sum to ~1
        self.assertAlmostEqual(probs[0].sum(), 1.0, places=5)


class TestSVM(unittest.TestCase):

    def setUp(self):
        from src.models.svm_model import SVMModel
        self.model = SVMModel(max_features=100)

    def test_train_returns_metrics(self):
        metrics = self.model.train(SMALL_CORPUS, SMALL_LABELS, test_size=0.25)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)

    def test_predict_returns_list(self):
        self.model.train(SMALL_CORPUS, SMALL_LABELS, test_size=0.25)
        preds = self.model.predict(['loved it great', 'worst film ever'])
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 2)
        self.assertTrue(all(p in [0, 1] for p in preds))

    def test_predict_proba_shape(self):
        self.model.train(SMALL_CORPUS, SMALL_LABELS, test_size=0.25)
        probs = self.model.predict_proba(['amazing film'])
        self.assertEqual(probs.shape[1], 2)


# ── Evaluator Tests ───────────────────────────────────────────────────────────

class TestEvaluator(unittest.TestCase):

    def test_perfect_predictions(self):
        y = [1, 0, 1, 0, 1]
        metrics = evaluate(y, y)
        self.assertAlmostEqual(metrics['accuracy'], 1.0)
        self.assertAlmostEqual(metrics['f1_score'], 1.0)

    def test_worst_predictions(self):
        y_true = [1, 1, 1, 0, 0]
        y_pred = [0, 0, 0, 1, 1]
        metrics = evaluate(y_true, y_pred)
        self.assertAlmostEqual(metrics['accuracy'], 0.0)

    def test_confusion_matrix_shape(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        metrics = evaluate(y_true, y_pred)
        self.assertEqual(metrics['confusion_matrix'].shape, (2, 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
