# Streamlit Cloud Deployment Guide

## Issues Fixed

Your app fails on Streamlit Cloud / remote machines because:

1. **Hardcoded relative paths** — `'data/imdb'`, `'outputs/'`, `'data/feedback/'` work locally but fail on cloud
2. **Large dataset (~80MB)** — IMDB dataset download fails on ephemeral filesystem
3. **File persistence** — Trained models and feedback saved to disk are lost on redeploy
4. **No graceful fallback** — App crashes instead of handling missing files

## Solutions Implemented ✓

✅ **Absolute Path Handling**: All file paths now use `Path(__file__).parent` to work on any machine
✅ **Error Handling**: Downloads/saves fail gracefully on read-only filesystems (Streamlit Cloud)
✅ **Directory Creation**: Safe directory creation with error handling

## To Deploy on Streamlit Cloud

### Option 1: Pre-Download Dataset (Recommended)

Download the dataset locally and commit to your repo:

```bash
cd sentiment-analysis
python3 -c "from src.crawler.dataset_loader import DatasetLoader; DatasetLoader.load_imdb()"
git add data/processed/imdb_processed.csv
git commit -m "Add preprocessed IMDB dataset"
git push
```

**Pros:** App loads instantly on cloud
**Cons:** Commit is ~400MB (but GitHub allows it with Git LFS if needed)

### Option 2: Host Dataset on Cloud (Better for Git)

1. **Create a release** on GitHub and upload the processed CSV:
   ```bash
   # Create a release
   gh release create v1.0 data/processed/imdb_processed.csv
   ```

2. **Update `src/crawler/dataset_loader.py`** to download from the release:
   ```python
   RELEASE_URL = "https://github.com/yourusername/yourrepo/releases/download/v1.0/imdb_processed.csv"
   
   # Add to load_imdb():
   if not processed_path.exists():
       try:
           print("Downloading dataset from release...")
           urllib.request.urlretrieve(RELEASE_URL, str(processed_path))
       except:
           # Fallback...
   ```

### Option 3: Use Hugging Face Datasets (Best)

```python
from datasets import load_dataset

df = load_dataset('imdb', split='train+test', trust_remote_code=True).to_pandas()
df['label'] = df['label'].map({0: 0, 1: 1})
```

### Option 4: Use Demo Mode (Fallback)

If dataset fails to load, automatically use a small sample:

```python
# In app.py
@st.cache_resource
def load_and_train_models(sample_size=None):
    loader = DatasetLoader()
    df = loader.load_imdb()
    
    if len(df) == 0:  # Dataset failed - fallback to demo
        st.warning("⚠️ Running in demo mode with sample data")
        df = pd.DataFrame({
            'text': ['Good movie', 'Bad movie', 'Amazing film'],
            'label': [1, 0, 1]
        })
    # ... rest of training
```

## Online Scraper Status

The `ReviewCrawler` **will work** on Streamlit Cloud, but with caveats:

✅ **What Works:**
- Web scraping functionality
- Collecting reviews from sites
- CSV saving (if data folder is writable)

⚠️ **What May Fail:**
- Saving to persistent storage (disk is ephemeral)
- Large-scale crawling (timeouts after 5 min)

**Recommendation**: Cache crawled data in Streamlit session state instead of disk:

```python
import streamlit as st

@st.cache_resource
def collect_reviews(urls):
    crawler = ReviewCrawler("https://example.com")
    crawler.crawl_multiple_pages(urls)
    return pd.DataFrame(crawler.reviews)  # Cache, don't save to disk
```

## Deployment Checklist

- [ ] **Local testing**: Run `streamlit run app.py` locally and verify it works
- [ ] **Choose dataset strategy** (Option 1-4 above)
- [ ] **Update `.gitignore`** if keeping large files:
  ```
  data/imdb/aclImdb_v1.tar.gz  # Download on first run
  outputs/                      # Runtime files
  data/feedback/               # Runtime files
  ```
- [ ] **Create `requirements.txt`** with all dependencies:
  ```bash
  pip freeze > requirements.txt
  ```
- [ ] **Create `streamlit_config.toml`** (optional):
  ```toml
  [server]
  maxUploadSize = 200  # MB
  ```
- [ ] **Push to GitHub**
- [ ] **Deploy on Streamlit Cloud**:
  - Go to https://share.streamlit.io
  - Click "New app" → Select your repo
  - Choose `sentiment-analysis/app.py` as main file

## Troubleshooting

### "FileNotFoundError: data/imdb not found"
→ Dataset download failed. Use Option 1 or 2 (pre-download or use GitHub release)

### "Permission denied writing to outputs/"
→ Normal on Streamlit Cloud. Models/feedback stay in memory. Use session state caching.

### "Model takes 2-3 minutes to load"
→ First run trains models. Subsequent runs use cache. Reduce model size if targeting <30s:
```python
sample_size = 5000  # Train on 5K samples instead of 50K
```

### "Out of memory errors"
→ Streamlit Cloud has ~1GB RAM. Reduce dataset size or switch to Hugging Face Datasets.

## Files Modified

- ✅ `src/crawler/dataset_loader.py` — Absolute paths + error handling
- ✅ `src/crawler/review_crawler.py` — Absolute paths + graceful failures
- ✅ `src/online_learning/online_learner.py` — Absolute paths
- ✅ `src/online_learning/feedback_collector.py` — Absolute paths + error handling  
- ✅ `app.py` — Uses absolute paths for all file operations
- ✅ `config.py` — New: Centralized config for local/cloud

## Questions?

- Check Streamlit docs: https://docs.streamlit.io/deploy/streamlit-cloud
- Review modified files for inline comments
