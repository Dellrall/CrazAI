"""Dataset loading utilities for sentiment analysis."""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

# Get absolute paths based on script location
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
IMDB_DIR = DATA_DIR / 'imdb'
PROCESSED_DIR = DATA_DIR / 'processed'


class DatasetLoader:
    """Load sentiment datasets from reliable sources."""

    @staticmethod
    def load_imdb(data_dir: Optional[str] = None) -> pd.DataFrame:
        """Load Stanford IMDB Large Movie Review Dataset.

        Downloads the dataset if not already present.
        Returns a DataFrame with 'text' and 'label' columns.
        Label: 1 = positive, 0 = negative.
        
        Args:
            data_dir: Override default data directory (uses absolute path if provided)
        """
        if data_dir:
            data_dir = Path(data_dir).resolve()
        else:
            data_dir = IMDB_DIR
        
        # Ensure directories exist
        data_dir.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        archive_path = data_dir / 'aclImdb_v1.tar.gz'
        extracted_dir = data_dir / 'aclImdb'
        processed_path = PROCESSED_DIR / 'imdb_processed.csv'

        if processed_path.exists():
            df = pd.read_csv(processed_path)
            if 'processed' not in df.columns and 'text' in df.columns:
                df['cleaned'] = df['text']
                df['processed'] = df['text']
            print(f"Loaded {len(df)} samples from cached IMDB dataset")
            return df

        if not extracted_dir.exists():
            try:
                print("Downloading IMDB dataset...")
                urllib.request.urlretrieve(
                    'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
                    str(archive_path)
                )
                print("Extracting...")
                with tarfile.open(str(archive_path), 'r:gz') as tar:
                    tar.extractall(str(data_dir))
                print("Done.")
            except (OSError, PermissionError) as e:
                print(f"⚠️ Warning: Could not download/extract dataset: {e}")
                print("This may be normal on Streamlit Cloud. Pre-populate data/processed/imdb_processed.csv")
                return pd.DataFrame({'text': [], 'label': []})

        reviews, sentiments = [], []
        for split in ['train', 'test']:
            for sentiment in ['pos', 'neg']:
                folder = extracted_dir / split / sentiment
                if not folder.exists():
                    continue
                for filename in sorted(folder.iterdir()):
                    if filename.is_file():
                        with open(filename, 'r', encoding='utf-8') as f:
                            reviews.append(f.read())
                            sentiments.append(1 if sentiment == 'pos' else 0)

        if not reviews:
            print("⚠️ No reviews found - dataset may not have downloaded correctly")
            return pd.DataFrame({'text': [], 'label': []})
            
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
