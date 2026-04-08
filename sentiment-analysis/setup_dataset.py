#!/usr/bin/env python3
"""
Quick setup script to download and preprocess the IMDB dataset.
Run once before launching the Streamlit app.

Usage:
    python3 setup_dataset.py
"""

import os
import sys
from pathlib import Path

# Add parent to path
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

from src.crawler.dataset_loader import DatasetLoader
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tokenizer import TextTokenizer

def setup():
    """Download IMDB and preprocess all 50K reviews."""
    print("=" * 70)
    print("SENTIMENT ANALYSIS - DATASET SETUP")
    print("=" * 70)
    
    # Step 1: Load IMDB
    print("\n[1/3] Loading IMDB dataset (downloading if needed)...")
    loader = DatasetLoader()
    df = loader.load_imdb(str(BASE_DIR / 'data' / 'imdb'))
    
    if len(df) == 0:
        print("❌ Failed to load dataset")
        return False
    
    print(f"✓ Loaded {len(df):,} reviews")
    
    # Step 2: Check if already processed
    processed_path = BASE_DIR / 'data' / 'processed' / 'imdb_processed.csv'
    if 'processed' in df.columns:
        print("✓ Dataset already has 'processed' column")
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"✓ Saved to {processed_path}")
        return True
    
    # Step 3: Preprocess (clean + tokenize)
    print("\n[2/3] Preprocessing 50,000 reviews (2-3 minutes)...")
    print("  This includes: cleaning, tokenization, lemmatization")
    
    cleaner = TextCleaner()
    tokenizer = TextTokenizer()
    
    # Progress indicator
    total = len(df)
    checkpoint = total // 10
    
    def preprocess_with_progress(texts):
        cleaned = []
        for i, text in enumerate(texts):
            if i > 0 and i % checkpoint == 0:
                pct = (i / total) * 100
                print(f"  {pct:5.1f}% ({i:,}/{total:,})")
            cleaned.append(cleaner.clean(text))
        return cleaned
    
    df['cleaned'] = preprocess_with_progress(df['text'])
    
    def tokenize_with_progress(texts):
        tokenized = []
        for i, text in enumerate(texts):
            if i > 0 and i % checkpoint == 0:
                pct = (i / total) * 100
                print(f"  {pct:5.1f}% ({i:,}/{total:,})")
            tokenized.append(tokenizer.preprocess(text))
        return tokenized
    
    df['tokens'] = tokenize_with_progress(df['cleaned'])
    df['processed'] = df['tokens'].apply(lambda t: ' '.join(t))
    
    print(f"✓ Preprocessing complete")
    
    # Step 4: Save
    print("\n[3/3] Saving processed dataset...")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"✓ Saved {len(df):,} reviews to:\n  {processed_path}")
    
    print("\n" + "=" * 70)
    print("✅ Setup complete! Now run: streamlit run app.py")
    print("=" * 70)
    return True

if __name__ == '__main__':
    try:
        success = setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
