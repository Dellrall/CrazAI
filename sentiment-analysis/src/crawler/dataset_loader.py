"""Dataset loading utilities for sentiment analysis."""

import os
import tarfile
import urllib.request

import pandas as pd


class DatasetLoader:
    """Load sentiment datasets from reliable sources."""

    @staticmethod
    def load_imdb(data_dir: str = 'data/imdb') -> pd.DataFrame:
        """Load Stanford IMDB Large Movie Review Dataset.

        Downloads the dataset if not already present.
        Returns a DataFrame with 'text' and 'label' columns.
        Label: 1 = positive, 0 = negative.
        """
        archive_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')
        extracted_dir = os.path.join(data_dir, 'aclImdb')
        processed_path = os.path.join('data', 'processed', 'imdb_processed.csv')

        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
            if 'processed' not in df.columns and 'text' in df.columns:
                df['cleaned'] = df['text']
                df['processed'] = df['text']
            print(f"Loaded {len(df)} samples from cached IMDB dataset")
            return df

        if not os.path.exists(extracted_dir):
            os.makedirs(data_dir, exist_ok=True)
            print("Downloading IMDB dataset...")
            urllib.request.urlretrieve(
                'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                archive_path
            )
            print("Extracting...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(data_dir)
            print("Done.")

        reviews, sentiments = [], []
        for split in ['train', 'test']:
            for sentiment in ['pos', 'neg']:
                folder = os.path.join(extracted_dir, split, sentiment)
                if not os.path.exists(folder):
                    continue
                for filename in sorted(os.listdir(folder)):
                    filepath = os.path.join(folder, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        reviews.append(f.read())
                        sentiments.append(1 if sentiment == 'pos' else 0)

        df = pd.DataFrame({'text': reviews, 'label': sentiments})
        print(f"Loaded {len(df)} samples from IMDB dataset")
        return df

    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load a CSV dataset (crawled or downloaded).

        Expects columns: 'text' and 'label'.
        """
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples from {filepath}")
        return df
