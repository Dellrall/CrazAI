# Streamlit Cloud Deployment Guide

## Issues Fixed

Your app fails on Streamlit Cloud / remote machines because:

1. **Hardcoded relative paths** — `'data/imdb'`, `'outputs/'`, `'data/feedback/'` work locally but fail on cloud
2. **Large dataset (~80MB)** — IMDB dataset download fails on ephemeral filesystem
3. **File persistence** — Trained models and feedback saved to disk are lost on redeploy
4. **Slow preprocessing** — Lemmatization takes 2s per text; 50K rows = 27+ hours
5. **No graceful fallback** — App crashes instead of handling missing files

## Local Development

### First Time Only

The app will **automatically preprocess small datasets** on first run, or you can pre-generate the processed data:

```bash
cd sentiment-analysis
python3 setup_dataset.py
```

This:
- ✅ Downloads IMDB (~80MB)
- ✅ Preprocesses all 50K reviews (2-3 minutes)
- ✅ Saves to `data/processed/imdb_processed.csv` (reused on every run)

### Then Launch

```bash
streamlit run app.py
```

**After first run, it loads instantly from cache.**

---

## Deploying on Streamlit Cloud

Streamlit Cloud has **NO persistent disk** — files are deleted after 24 hours. You must choose a strategy:

### ⭐ Option 1: Commit Processed CSV (Easiest)

Pre-generate locally and commit to repo:

```bash
# Local setup (one time)
python3 setup_dataset.py

# Commit to repo
git add data/processed/imdb_processed.csv
git commit -m "Add preprocessed IMDB dataset"
git push
```

**Pros:**
- ✅ App loads instantly on cloud
- ✅ No downloads needed

**Cons:**
- Large file (~400MB) — may hit GitHub limits
- Use **Git LFS** if needed:
  ```bash
  git lfs install
  git lfs track "data/processed/*.csv"
  git add .gitattributes data/processed/imdb_processed.csv
  git commit -m "Add IMDB dataset (LFS)"
  git push
  ```

### Option 2: GitHub Releases (Better for Size)

Host the CSV on a GitHub release instead of committing:

```bash
# Create release (requires GitHub CLI)
gh release create v1.0 \
  --title "IMDB Preprocessed Dataset" \
  data/processed/imdb_processed.csv

# The app will auto-download from:
# https://github.com/YOUR_USER/YOUR_REPO/releases/download/v1.0/imdb_processed.csv
```

Then update [src/crawler/dataset_loader.py](src/crawler/dataset_loader.py):

```python
# Add after line 20:
RELEASE_URL = "https://github.com/yourusername/yourrepo/releases/download/v1.0/imdb_processed.csv"

# In load_imdb(), after line 46:
if not processed_path.exists():
    try:
        print("Downloading dataset from GitHub release...")
        urllib.request.urlretrieve(RELEASE_URL, str(processed_path))
        print(f"Downloaded to {processed_path}")
        df = pd.read_csv(processed_path)
        return df
    except:
        print("Could not download from release")
```

### ⭐ Option 3: Hugging Face Datasets (Recommended)

Simplest and most reliable:

```bash
pip install datasets
```

Then in [src/crawler/dataset_loader.py](src/crawler/dataset_loader.py), add:

```python
@staticmethod
def load_imdb_hf() -> pd.DataFrame:
    """Load IMDB from Hugging Face (works everywhere)."""
    try:
        from datasets import load_dataset
        print("Loading IMDB from Hugging Face...")
        ds = load_dataset('imdb', split='train+test', trust_remote_code=True)
        df = ds.to_pandas()
        df['label'] = df['label'].astype(int)
        print(f"Loaded {len(df)} samples from HF")
        return df
    except Exception as e:
        print(f"HF load failed: {e}")
        return pd.DataFrame({'text': [], 'label': []})
```

Then in `load_imdb()`, add fallback:

```python
if not processed_path.exists() and not archive_exists:
    # Try Hugging Face first
    df = DatasetLoader.load_imdb_hf()
    if len(df) > 0:
        return df
```

**Pros:**
- ✅ Works on Streamlit Cloud without setup
- ✅ Automatic caching
- ✅ No repo size impact

**Cons:**
- Requires internet on first run (~30s)
- Dataset cached after first run

### Option 4: Use Demo Dataset

If you just want a quick working demo without full IMDB:

Create `data/processed/imdb_reduced.csv` with 1,000 random samples:

```python
# One-time: Create small demo
import pandas as pd
from src.crawler.dataset_loader import DatasetLoader

df = DatasetLoader.load_imdb()
df_demo = df.sample(n=1000, random_state=42)
df_demo.to_csv('data/processed/imdb_reduced.csv', index=False)
```

Then commit to repo. App auto-detects and uses it (fast loading, smaller scope).

---

## Deployment Checklist

- [ ] **Choose dataset strategy** (Option 1-4 above)
- [ ] **Local testing**: `streamlit run app.py` works
- [ ] **Setup complete**: Dataset loads instantly second run
- [ ] **Create `requirements.txt`**:
  ```bash
  pip freeze > requirements.txt
  ```
- [ ] **Create `.gitignore`** (don't commit ephemeral files):
  ```
  # Cache
  .streamlit/
  __pycache__/
  *.pyc
  
  # Large downloads (optional - redownloaded on each deploy)
  data/imdb/aclImdb_v1.tar.gz
  data/imdb/aclImdb/
  
  # Runtime files (will be recreated)
  outputs/
  data/feedback/
  ```
- [ ] **Push to GitHub**
- [ ] **Deploy on Streamlit Cloud**:
  - Go to https://share.streamlit.io
  - Click "New app"
  - Select your repo and `sentiment-analysis/app.py`
  - Click "Deploy"

## Online Scraper Status

The `ReviewCrawler` **will work** on Streamlit Cloud, but with considerations:

✅ **What Works:**
- Web scraping functionality
- Collecting reviews from sites
- Data stays in memory during session

⚠️ **What Fails:**
- Saving to persistent disk (ephemeral filesystem)
- Large-scale crawling (5-minute timeout)

**Best Practice for Cloud:** Cache crawled data in Streamlit session state:

```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def collect_reviews(urls):
    from src.crawler.review_crawler import ReviewCrawler
    crawler = ReviewCrawler("https://example.com")
    crawler.crawl_multiple_pages(urls)
    return pd.DataFrame(crawler.reviews)  # Returns, not saves
```

---

## Troubleshooting

### "Dataset failed to load"
→ Choose Option 1-3 above to provide dataset

### "Takes 2-3 minutes on every deploy"
→ Use Option 1 (commit CSV) to skip re-download

### "Out of memory"
→ Use Option 4 (demo dataset) or reduce `sample_size` in sidebar

### "Permission denied writing to outputs/"
→ Normal on Streamlit Cloud. Models/feedback stay in memory only (won't persist between deploys)

### "Lemmatization is slow"
→ Reduce dataset to 5K-10K samples instead of full 50K

---

## Files Modified

- ✅ `src/crawler/dataset_loader.py` — Absolute paths + auto-fallback
- ✅ `src/crawler/review_crawler.py` — Absolute paths
- ✅ `src/online_learning/online_learner.py` — Absolute paths
- ✅ `src/online_learning/feedback_collector.py` — Absolute paths  
- ✅ `src/models/bert_model.py` — Absolute paths
- ✅ `src/evaluation/evaluator.py` — Absolute paths
- ✅ `app.py` — Auto-preprocess + better errors
- ✅ `main.py` — Absolute paths
- ✅ `setup_dataset.py` — One-time preprocessor script
- ✅ `config.py` — Config template

---

## Questions?

- Streamlit Docs: https://docs.streamlit.io/deploy/streamlit-cloud
- See inline comments in updated files for more details
