# 🎭 Sentiment Analysis — Streamlit App

NLP-powered sentiment analysis with **Naïve Bayes**, **SVM**, and **Online Learning** trained on the IMDB Large Movie Review Dataset (50K reviews).

**Features:**
- 🎯 Real-time sentiment classification
- 📊 Model comparison & metrics
- 🔄 Online learning with user feedback
- 📚 Full dataset exploration
- ⚡ Cached predictions for speed

---

## 🚀 Quick Start

### Prerequisites
```bash
python3.8+
pip
```

### Local Setup (2 steps)

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Download & preprocess dataset (one-time, ~3 minutes):**
```bash
python3 setup_dataset.py
```

This downloads IMDB (~80MB) and preprocesses all 50K reviews. Results are cached, so it only runs once.

**3. Launch the app:**
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. ✨

---

## 📁 Project Structure

```
sentiment-analysis/
├── app.py                          # Main Streamlit app
├── main.py                         # Batch pipeline (NB + SVM + BERT)
├── setup_dataset.py                # One-time dataset setup
├── requirements.txt                # Python dependencies
│
├── src/
│   ├── crawler/                    # Data loading & web scraping
│   │   ├── dataset_loader.py       # IMDB dataset loader
│   │   └── review_crawler.py       # Web scraper for reviews
│   ├── preprocessing/              # Text processing pipeline
│   │   ├── text_cleaner.py         # HTML, emoji, slang cleanup
│   │   ├── tokenizer.py            # Tokenization + lemmatization
│   │   ├── feature_extractor.py    # TF-IDF feature extraction
│   │   └── dictionary_normalizer.py# Slang normalization
│   ├── models/                     # Sentiment classifiers
│   │   ├── naive_bayes.py          # Naïve Bayes classifier
│   │   ├── svm_model.py            # SVM classifier
│   │   └── bert_model.py           # BERT fine-tuning
│   ├── online_learning/            # Incremental learning
│   │   ├── online_learner.py       # Online model updates
│   │   └── feedback_collector.py   # User feedback storage
│   ├── evaluation/                 # Model evaluation
│   │   └── evaluator.py            # Metrics & comparison
│   └── utils/
│       └── helpers.py              # Utility functions
│
├── data/
│   ├── imdb/                       # IMDB dataset (auto-downloaded)
│   ├── processed/
│   │   └── imdb_processed.csv      # Cache (generated once)
│   ├── raw/                        # For custom/crawled data
│   └── feedback/                   # User corrections
│
├── outputs/                        # Results & charts (runtime)
├── tests/                          # Unit tests
├── DEPLOYMENT.md                   # Cloud deployment guide
└── README.md                       # This file
```

---

## 📊 Usage

### Via Web App (Streamlit)
```bash
streamlit run app.py
```

**Tabs:**
- **🔍 Try It** — Enter text and get sentiment predictions
- **🔄 Feedback & Learning** — Correct predictions to improve the model
- **📊 Model Stats** — View accuracy, precision, recall, F1-score
- **📂 Dataset** — Explore the IMDB reviews
- **📖 How It Works** — Explain the pipeline

### Via CLI (Batch Processing)
```bash
# Train all models (NB + SVM + BERT)
python3 main.py

# Skip BERT (faster)
python3 main.py --skip-bert

# Naïve Bayes only
python3 main.py --nb-only
```

Output saved to `outputs/results.csv` and charts.

---

## ⚡ Performance

| Model | Accuracy | Speed |
|-------|----------|-------|
| Naïve Bayes | 86-88% | <1s per prediction |
| SVM | 86-88% | ~2-5s per prediction |
| BERT | 90%+ | ~5s per prediction |

**Dataset size impact:**
- 2,000 reviews → 1 min setup
- 5,000 reviews → 2 min setup
- 10,000 reviews → 5 min setup
- 50,000 reviews (full) → 20 min setup on first run

Subsequent runs use cache (instant).

---

## 🌐 Deploy to Streamlit Cloud

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

**Quick version:**
1. Commit dataset: `git add data/processed/imdb_processed.csv && git commit -m "Add dataset"`
2. Push to GitHub: `git push`
3. Go to https://share.streamlit.io → "New app"
4. Select your repo → choose `sentiment-analysis/app.py`
5. Click "Deploy"

**Note:** Large files? Use Option 2-3 in [DEPLOYMENT.md](DEPLOYMENT.md) to avoid hitting GitHub's limits.

---

## 🧪 Testing

```bash
python3 tests/test_models.py
```

Tests training, prediction, and evaluation on a small sample dataset.

---

## 🔧 Troubleshooting

### "App is slow / stuck after launch"
- First run preprocesses the dataset (~2-3 min). Subsequent runs are instant.
- If still stuck after 5 min: Kill the process and run `python3 setup_dataset.py` manually.

### "Out of memory"
- Reduce `sample_size` in the sidebar (2K-5K instead of full 50K)
- Or run on a machine with 8GB+ RAM

### "Permission denied writing to outputs/"
- This is normal on Streamlit Cloud (ephemeral filesystem).
- Models/feedback stay in memory during the session.
- Restart the app to reset.

### "Module import errors"
- Install dependencies: `pip install -r requirements.txt`
- For BERT: `pip install torch transformers`

---

## 📚 Model Details

### Naïve Bayes
- **Algorithm**: Multinomial Naïve Bayes
- **Features**: TF-IDF vectors (50K vocabulary)
- **Speed**: Fastest
- **Accuracy**: ~86-88%

### SVM
- **Algorithm**: LinearSVC
- **Features**: TF-IDF vectors (50K vocabulary)
- **Speed**: Moderate
- **Accuracy**: ~86-88%

### BERT
- **Model**: DistilBERT (fine-tuned)
- **Speed**: Slowest (~5s per prediction)
- **Accuracy**: ~90%+
- **Note**: GPU recommended for training

### Online Learner
- **Algorithm**: SGDClassifier with HashingVectorizer
- **Updates**: Real-time with user feedback
- **Memory**: ~100MB for full dataset

---

## 📖 Pipeline Explanation

```
Raw Text
   ↓
Text Cleaner (remove HTML, emojis, URLs)
   ↓
Tokenizer (split into words, lowercase)
   ↓
Lemmatizer (reduce to base form: "running" → "run")
   ↓
Feature Extractor (TF-IDF vectors)
   ↓
Classifier (NB / SVM / BERT)
   ↓
Sentiment: 0 (Negative) or 1 (Positive)
```

---

## 🤝 Contributing

Issues and PRs welcome! Please test locally before submitting:

```bash
pytest tests/test_models.py -v
```

---

## 📄 License

See [LICENSE](../LICENSE)

---

## 🔗 Resources

- **IMDB Dataset**: http://ai.stanford.edu/~amaas/data/sentiment/
- **Streamlit Docs**: https://docs.streamlit.io
- **Scikit-Learn**: https://scikit-learn.org
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **spaCy Lemmatizer**: https://spacy.io

---

**Questions?** Check [DEPLOYMENT.md](DEPLOYMENT.md) for cloud deployment or see inline code comments.
