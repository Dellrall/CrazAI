"""Configuration for local vs remote deployment."""

import os
import streamlit as st

# Get the directory of this script (works on local and remote)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FEEDBACK_DIR = os.path.join(DATA_DIR, 'feedback')

# Create directories if they don't exist (with error handling for read-only filesystems)
for dir_path in [DATA_DIR, OUTPUT_DIR, FEEDBACK_DIR]:
    try:
        os.makedirs(dir_path, exist_ok=True)
    except (OSError, PermissionError):
        pass  # On Streamlit Cloud, ignore permission errors

# Detect if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_SERVER_HEADLESS') == 'true'

# Use session state for cached data on Streamlit Cloud
@st.cache_resource
def get_cached_df():
    """Store dataframe in cache across reruns."""
    return {'df': None}

# For datasets too large for ephemeral storage, use a public URL
IMDB_PROCESSED_URL = "https://raw.githubusercontent.com/yourusername/yourrepo/main/data/processed/imdb_processed.csv"
# Or use Hugging Face datasets
USE_HF_DATASETS = True  # Recommended: use huggingface datasets library
