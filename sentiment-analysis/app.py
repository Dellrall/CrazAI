"""
Sentiment Analysis — Streamlit Demo App
========================================
Demonstrates trained NB and SVM models with online learning feedback loop.

Run with:
    streamlit run app.py
"""

import os
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Get absolute paths
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / 'outputs'
FEEDBACK_DIR = BASE_DIR / 'data' / 'feedback'
DATA_DIR = BASE_DIR / 'data'

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

from src.crawler.dataset_loader import DatasetLoader
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tokenizer import TextTokenizer
from src.models.naive_bayes import NaiveBayesModel
from src.models.svm_model import SVMModel
from src.online_learning.online_learner import OnlineLearner
from src.online_learning.feedback_collector import FeedbackCollector

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment Analyser",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Check dataset availability (BEFORE cache)
# ─────────────────────────────────────────────

processed_csv = DATA_DIR / 'processed' / 'imdb_processed.csv'

def check_dataset_availability():
    """Check if dataset is ready before loading models."""
    if not processed_csv.exists():
        st.error("❌ **Dataset not found**")
        st.info(
            "The preprocessed IMDB dataset is missing. "
            "Run this **one-time setup** to download and preprocess:\n\n"
            "```bash\n"
            "python3 setup_dataset.py\n"
            "```\n\n"
            "This will:\n"
            "1. Download IMDB reviews (~80MB)\n"
            "2. Preprocess (clean, tokenize, lemmatize)\n"
            "3. Save to `data/processed/imdb_processed.csv`\n\n"
            "**Takes 2-3 minutes, runs only once.**"
        )
        return False
    return True

# Check on first load only
if 'dataset_checked' not in st.session_state:
    st.session_state.dataset_checked = check_dataset_availability()

if not st.session_state.dataset_checked:
    st.stop()

# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .big-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .subtitle { color: #94a3b8; font-size: 1.1rem; margin-top: -0.5rem; margin-bottom: 2rem; }

    .result-card {
        border-radius: 16px; padding: 1.5rem 2rem;
        text-align: center; font-size: 1.2rem; font-weight: 600; margin: 1rem 0;
    }
    .result-positive { background: linear-gradient(135deg,#d1fae5,#a7f3d0); color:#065f46; border:2px solid #34d399; }
    .result-negative { background: linear-gradient(135deg,#fee2e2,#fecaca); color:#7f1d1d; border:2px solid #f87171; }

    .guide-step {
        background: linear-gradient(135deg, #1e1b4b, #2e1065);
        border-left: 4px solid #8b5cf6;
        border-radius: 8px; padding: 1rem 1.5rem; margin: 0.75rem 0; color: #e2e8f0;
    }
    .guide-step h4 { color: #c4b5fd; margin: 0 0 0.4rem 0; }

    .uncertain-badge {
        background: #fef3c7; border: 1px solid #f59e0b;
        color: #92400e; border-radius: 8px; padding: 0.4rem 0.8rem;
        font-size: 0.85rem; font-weight: 600; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def preprocess_text(text: str, cleaner: TextCleaner, tokenizer: TextTokenizer) -> tuple[str, list[str]]:
    cleaned = cleaner.clean(text)
    tokens = tokenizer.preprocess(cleaned)
    return ' '.join(tokens), tokens

# ─────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_and_train_models(sample_size: int = None):
    """Load IMDB data and train NB + SVM + OnlineLearner. Cached after first run."""
    loader = DatasetLoader()
    df = loader.load_imdb(str(DATA_DIR / 'imdb'))
    
    if len(df) == 0:
        raise RuntimeError("Failed to load dataset")
    
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Dataset should already have 'processed' column from setup_dataset.py
    if 'processed' not in df.columns:
        # This shouldn't happen if check_dataset_availability() passed
        raise RuntimeError(
            "Dataset missing 'processed' column. "
            "Run: python3 setup_dataset.py"
        )

    texts = df['processed'].tolist()
    labels = df['label'].tolist()

    cleaner = TextCleaner()
    tokenizer = TextTokenizer()
    
    nb = NaiveBayesModel(max_features=50000)
    svm = SVMModel(max_features=50000)

    nb_metrics = nb.train(texts, labels, test_size=0.2)
    svm_metrics = svm.train(texts, labels, test_size=0.2)

    # Seed the online learner with the same training data
    online = OnlineLearner(str(OUTPUT_DIR / 'online_model.pkl'))
    if not online.load():
        online.initialize(texts, labels)

    return nb, svm, online, nb_metrics, svm_metrics, df, cleaner, tokenizer


@st.cache_resource(show_spinner=False)
def get_feedback_collector():
    return FeedbackCollector(str(FEEDBACK_DIR / 'corrections.jsonl'))


@st.cache_resource(show_spinner=False)
def load_csv_results():
    path = OUTPUT_DIR / 'results.csv'
    if path.exists():
        return pd.read_csv(path)
    return None

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    sample_size = st.selectbox(
        "Training sample size",
        [2000, 5000, 10000, 20000, None],
        index=4,
        format_func=lambda x: "Full dataset (50K) ← recommended" if x is None else f"{x:,} samples",
        help="More samples = better accuracy. Full 50K gives ~86-88%."
    )
    model_choice = st.radio(
        "Model to use",
        ["Naïve Bayes", "SVM", "Both (compare)"],
        index=2
    )
    st.divider()
    st.markdown("### 📋 About")
    st.markdown("""
    Demonstrates **NB**, **SVM**, and **Online Learning** trained on the **IMDB Large Movie Review Dataset** (50K reviews).

    **Pipeline:** Clean → Tokenize → TF-IDF → Classify

    **Online Learning:** The model improves live as you submit feedback corrections.
    """)

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────

st.markdown('<div class="big-title">🎭 Sentiment Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NLP-powered sentiment analysis · online learning · human-in-the-loop feedback</div>', unsafe_allow_html=True)

with st.spinner("Loading & training models on IMDB data... (first run takes ~2–3 min on full dataset)"):
    nb, svm, online, nb_metrics, svm_metrics, df, cleaner, tokenizer = load_and_train_models(sample_size)

collector = get_feedback_collector()

actual_train = nb_metrics['n_train']
actual_test = nb_metrics['n_test']
st.success(f"✅ Models ready — trained on {actual_train:,} samples, tested on {actual_test:,} · {collector.count()} feedback corrections collected")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab_demo, tab_feedback, tab_stats, tab_explore, tab_guide = st.tabs([
    "🔍 Try It", "🔄 Feedback & Learning", "📊 Model Stats", "📂 Dataset", "📖 How It Works"
])

# ══════════════════════════════════════════════
# TAB 1: Try It
# ══════════════════════════════════════════════

with tab_demo:
    st.subheader("Enter any text to analyse its sentiment")

    example_texts = [
        "Type your own text...",
        "This movie was absolutely fantastic. The acting was superb and the story kept me hooked!",
        "Terrible film. Complete waste of time and money. I want my 2 hours back.",
        "It was okay, nothing special but not bad either. Some parts were enjoyable.",
        "One of the best movies I have seen in years. A true masterpiece!",
        "Boring, slow, and predictable. I fell asleep halfway through.",
    ]

    selected = st.selectbox("Or pick an example:", example_texts)
    user_input = st.text_area(
        "Text to analyse:",
        value="" if selected == example_texts[0] else selected,
        height=120,
        placeholder="Paste a movie review, tweet, or any sentence here..."
    )

    analyse_btn = st.button("🔍 Analyse Sentiment", type="primary", width="stretch")

    if analyse_btn and user_input.strip():
        processed, tokens = preprocess_text(user_input, cleaner, tokenizer)
        st.divider()

        if model_choice == "Both (compare)":
            col_nb, col_svm = st.columns(2)
            for col, model, name in [(col_nb, nb, "Naïve Bayes"), (col_svm, svm, "SVM")]:
                with col:
                    st.markdown(f"#### {name}")
                    probs = model.predict_proba([processed])[0]
                    pred = int(np.argmax(probs))
                    conf = float(np.max(probs))
                    label = "Positive 😊" if pred == 1 else "Negative 😟"
                    css = "result-positive" if pred == 1 else "result-negative"
                    st.markdown(
                        f'<div class="result-card {css}">{label}<br>'
                        f'<span style="font-size:1.5rem">{conf:.1%} confident</span></div>',
                        unsafe_allow_html=True
                    )
                    if conf < 0.7:
                        st.markdown('<span class="uncertain-badge">⚠️ Low confidence — consider correcting in the Feedback tab</span>', unsafe_allow_html=True)
                    fig = go.Figure(go.Bar(
                        x=['Negative', 'Positive'], y=[probs[0], probs[1]],
                        marker_color=['#f87171', '#34d399'],
                        text=[f'{probs[0]:.1%}', f'{probs[1]:.1%}'], textposition='outside'
                    ))
                    fig.update_layout(
                        height=200, margin=dict(t=10, b=10, l=10, r=10),
                        yaxis=dict(range=[0, 1.1], showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            model = nb if model_choice == "Naïve Bayes" else svm
            probs = model.predict_proba([processed])[0]
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))
            label = "Positive 😊" if pred == 1 else "Negative 😟"
            css = "result-positive" if pred == 1 else "result-negative"
            st.markdown(
                f'<div class="result-card {css}">{label}<br>'
                f'<span style="font-size:2rem">{conf:.1%} confident</span></div>',
                unsafe_allow_html=True
            )
            if conf < 0.7:
                st.markdown('<span class="uncertain-badge">⚠️ Low confidence — correct it in the Feedback tab!</span>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Prediction", "Positive" if pred == 1 else "Negative")
            c2.metric("Confidence", f"{conf:.1%}")

        with st.expander("🔧 What the pipeline did"):
            st.write(f"**Original:** {user_input[:200]}")
            st.write(f"**After cleaning:** {cleaner.clean(user_input)[:200]}")
            st.write(f"**Tokens (first 20):** {tokens[:20]}")
            st.caption("The text was lowercased, stripped of HTML/URLs/special characters, tokenized, stop words removed, and lemmatized before being fed to the model.")

    elif analyse_btn:
        st.warning("Please enter some text first.")

# ══════════════════════════════════════════════
# TAB 2: Feedback & Online Learning
# ══════════════════════════════════════════════

with tab_feedback:
    st.subheader("🔄 Feedback & Continuous Learning")
    st.markdown("""
    If the model predicted incorrectly, provide the correct label here.
    The **Online Learner** (SGDClassifier with `partial_fit`) will update immediately —
    no full retraining required.
    """)

    # ── Feedback form ─────────────────────────────────────────────
    with st.form("feedback_form"):
        fb_text = st.text_area(
            "Text the model got wrong:",
            height=100,
            placeholder="Paste the text that was misclassified..."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            original_pred = st.selectbox("What did the model predict?", ["Negative", "Positive"])
        with col_b:
            correct_label = st.selectbox("What's the correct sentiment?", ["Negative", "Positive"])

        model_that_erred = st.radio("Which model got it wrong?", ["Naïve Bayes", "SVM", "Online Learner", "All"], horizontal=True)
        submitted = st.form_submit_button("✅ Submit Correction", type="primary", width="stretch")

    if submitted and fb_text.strip():
        orig_int = 1 if original_pred == "Positive" else 0
        correct_int = 1 if correct_label == "Positive" else 0

        # Save correction
        collector.record(
            text=fb_text,
            original_prediction=orig_int,
            corrected_label=correct_int,
            model_name=model_that_erred,
            confidence=0.0
        )

        # Immediately update the online learner
        processed, _ = preprocess_text(fb_text, cleaner, tokenizer)
        update_info = online.update([processed], [correct_int])

        st.success(f"✅ Correction recorded! Online model updated (total feedback: {update_info['total_feedback']} samples)")
        st.info("ℹ️ The Online Learner has been updated. Naïve Bayes and SVM are batch models — they update when you click 'Retrain Batch Models' below.")

    elif submitted:
        st.warning("Please enter the text that was misclassified.")

    # ── Online Learner live prediction ────────────────────────────
    st.divider()
    st.subheader("🤖 Try the Online Learner")
    st.caption("This model updates live from your feedback above. Compare it to NB/SVM in the Try It tab.")

    ol_input = st.text_input("Test the online learner:", placeholder="Type any text...")
    if ol_input.strip():
        processed_ol, _ = preprocess_text(ol_input, cleaner, tokenizer)
        pred_ol, conf_ol, uncertain = online.predict_with_confidence(processed_ol)
        label_ol = "Positive 😊" if pred_ol == 1 else "Negative 😟"
        css_ol = "result-positive" if pred_ol == 1 else "result-negative"
        st.markdown(
            f'<div class="result-card {css_ol}" style="padding:1rem">'
            f'Online Learner says: <b>{label_ol}</b> ({conf_ol:.1%})'
            f'{"<br><small>⚠️ Uncertain prediction</small>" if uncertain else ""}</div>',
            unsafe_allow_html=True
        )

    # ── Feedback stats ────────────────────────────────────────────
    st.divider()
    st.subheader("📈 Feedback Log")

    summary = collector.summary()
    if summary['total'] > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Corrections", summary['total'])
        c2.metric("Disagreements", summary['corrections'])
        c3.metric("Correction Rate", f"{summary['correction_rate']:.1%}")
        c4.metric("Online Updates", len([h for h in online.update_history if h['type'] == 'feedback_update']))

        # Show feedback history
        entries = collector.load_all()
        feed_df = pd.DataFrame(entries)
        feed_df['original_prediction'] = feed_df['original_prediction'].map({1: '😊 Positive', 0: '😟 Negative'})
        feed_df['corrected_label'] = feed_df['corrected_label'].map({1: '😊 Positive', 0: '😟 Negative'})
        feed_df['text'] = feed_df['text'].apply(lambda x: x[:80] + '...' if len(x) > 80 else x)
        st.dataframe(
            feed_df[['timestamp', 'model_name', 'original_prediction', 'corrected_label', 'text']],
            use_container_width=True, hide_index=True
        )

        # Batch retrain button
        st.divider()
        if st.button("🔁 Update NB model on feedback data", width="stretch"):
            all_feedback = collector.load_all()
            if all_feedback:
                processed_all = [preprocess_text(e['text'], cleaner, tokenizer)[0] for e in all_feedback]
                labels_all = [e['corrected_label'] for e in all_feedback]
                # Use partial_fit on NB — keeps the existing TF-IDF vectorizer,
                # only updates the classifier weights. Fast and safe.
                nb.update(processed_all, labels_all)
                # Also sync the online learner with all feedback
                online.update(processed_all, labels_all)
                st.success(f"✅ Naïve Bayes updated with {len(all_feedback)} feedback samples via partial_fit!")
                st.info("ℹ️ SVM is a batch model and cannot be updated incrementally — it keeps its original IMDB training.")
            else:
                st.warning("No feedback collected yet.")
    else:
        st.info("No feedback submitted yet. Use the form above to correct a wrong prediction — the model will learn from it!")

# ══════════════════════════════════════════════
# TAB 3: Model Stats
# ══════════════════════════════════════════════

with tab_stats:
    st.subheader("📊 Model Performance on Test Set")

    col1, col2 = st.columns(2)
    for col, metrics, name in [(col1, nb_metrics, "Naïve Bayes"), (col2, svm_metrics, "SVM")]:
        with col:
            st.markdown(f"#### {name}")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            m2.metric("F1 Score", f"{metrics['f1_score']:.2%}")
            m3, m4 = st.columns(2)
            m3.metric("Precision", f"{metrics['precision']:.2%}")
            m4.metric("Recall", f"{metrics['recall']:.2%}")
            m5, m6 = st.columns(2)
            m5.metric("Train size", f"{metrics['n_train']:,}")
            m6.metric("Test size", f"{metrics['n_test']:,}")

    st.divider()
    st.subheader("Side-by-Side Comparison")
    metrics_df = pd.DataFrame([
        {"Model": "Naïve Bayes", "Metric": m, "Score": nb_metrics[m.lower().replace(" ", "_")]}
        for m in ["Accuracy", "Precision", "Recall", "F1 Score"]
    ] + [
        {"Model": "SVM", "Metric": m, "Score": svm_metrics[m.lower().replace(" ", "_")]}
        for m in ["Accuracy", "Precision", "Recall", "F1 Score"]
    ])
    fig = px.bar(
        metrics_df, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_sequence=["#6366f1", "#ec4899"], text_auto=".3f"
    )
    fig.update_layout(yaxis=dict(range=[0.8, 1.0]), height=350,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    saved = load_csv_results()
    if saved is not None:
        st.divider()
        st.subheader("📄 Full Pipeline Results (`outputs/results.csv`)")
        st.dataframe(saved, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4: Dataset Explorer
# ══════════════════════════════════════════════

with tab_explore:
    st.subheader("📂 Dataset Overview — IMDB Movie Reviews")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{len(df):,}")
    pos = int(df['label'].sum())
    neg = len(df) - pos
    c2.metric("Positive Reviews", f"{pos:,}", f"{pos/len(df):.1%}")
    c3.metric("Negative Reviews", f"{neg:,}", f"{neg/len(df):.1%}")

    fig_pie = px.pie(
        names=["Positive 😊", "Negative 😟"], values=[pos, neg],
        color_discrete_sequence=["#34d399", "#f87171"], title="Class Distribution"
    )
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

    df['word_count'] = df['processed'].apply(lambda x: len(x.split()))
    fig_hist = px.histogram(
        df, x='word_count', color=df['label'].map({1: 'Positive', 0: 'Negative'}),
        nbins=50, title="Word Count Distribution (after preprocessing)",
        color_discrete_map={"Positive": "#34d399", "Negative": "#f87171"},
        labels={'word_count': 'Word Count'}
    )
    fig_hist.update_layout(height=280, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Sample Reviews")
    n_samples = st.slider("Show N samples", 5, 50, 10)
    sample_df = df[['text', 'label']].sample(n=n_samples, random_state=42).copy()
    sample_df['label'] = sample_df['label'].map({1: '😊 Positive', 0: '😟 Negative'})
    sample_df['text'] = sample_df['text'].apply(lambda x: x[:150] + '...' if len(x) > 150 else x)
    st.dataframe(sample_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 5: How It Works (User Guide)
# ══════════════════════════════════════════════

with tab_guide:
    st.subheader("📖 How This App Works")

    st.markdown("### 🧠 The Full NLP Pipeline")
    st.markdown("""
    Every piece of text goes through **5 stages** before the model makes a prediction:
    """)

    steps = [
        ("1️⃣ Text Cleaning", "Removes HTML tags, URLs, @mentions, #hashtags, numbers and special characters. `'<b>Great</b> movie! 10/10'` → `'great movie'`"),
        ("2️⃣ Tokenisation", "Splits the text into individual words (tokens). `'great movie'` → `['great', 'movie']`"),
        ("3️⃣ Stop Word Removal", "Removes common words that carry no sentiment signal (the, is, a, was…). `['the', 'movie', 'is', 'great']` → `['movie', 'great']`"),
        ("4️⃣ Lemmatisation", "Reduces words to their base form so 'movies' and 'movie' are treated the same. `'running'` → `'run'`, `'movies'` → `'movie'`"),
        ("5️⃣ TF-IDF Vectorisation", "Converts the cleaned token list into a numerical vector — higher weight for rare, distinctive words than for common ones."),
    ]
    for title, desc in steps:
        st.markdown(f"""
        <div class="guide-step">
        <h4>{title}</h4>
        {desc}
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🤖 The Three Models")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("#### Naïve Bayes")
        st.markdown("""
        - Uses **probability** — how likely is this word to appear in a positive vs negative review?
        - Very fast, works well with sparse text data
        - Assumes words are independent (the "naïve" part)
        - Typical accuracy: **~86%**
        """)
    with col_m2:
        st.markdown("#### SVM")
        st.markdown("""
        - Finds a **decision boundary** that best separates positive and negative review vectors
        - More accurate than NB, especially on edge cases
        - Slower to train but very robust
        - Typical accuracy: **~88%**
        """)
    with col_m3:
        st.markdown("#### Online Learner")
        st.markdown("""
        - Uses **SGD (gradient descent)** to update weights incrementally
        - Never needs to retrain from scratch — learns from each correction instantly via `partial_fit`
        - Adapts to new vocabulary (slang, domain-specific terms) over time
        - Gets better the more feedback you give it ♻️
        """)

    st.divider()
    st.markdown("### 🔄 Online Learning (Continuous Improvement)")
    st.markdown("""
    Online learning is a method where the model **doesn't retrain from scratch** — instead it updates
    its internal weights using each new piece of labeled data via `partial_fit`:

    ```
    New Text → Predict → User sees result → User corrects if wrong
         ↓
    Correction stored in: data/feedback/corrections.jsonl
         ↓
    Online model updates: model.partial_fit([new_text], [correct_label])
         ↓
    Model is now slightly better at handling similar texts
    ```

    The key insight: **train/test split**. The models were trained on **80% of the IMDB data**
    and tested on the remaining **20%** — which they never saw during training. This gives
    an honest measure of how well the model will generalise to new unseen text.
    """)

    st.info("💡 **Tip:** If the model is wrong or shows low confidence (< 70%), go to the **Feedback & Learning** tab to correct it. The Online Learner updates immediately!")

    st.divider()
    st.markdown("### 🏗️ Project Structure")
    st.code("""
sentiment-analysis/
├── src/
│   ├── crawler/            Dataset loading & web scraping
│   ├── preprocessing/      Cleaning, tokenisation, TF-IDF
│   ├── models/             Naïve Bayes, SVM, BERT
│   ├── online_learning/    OnlineLearner, FeedbackCollector
│   └── evaluation/         Metrics, charts, confusion matrices
├── data/
│   └── feedback/           User corrections (JSONL, gitignored)
├── outputs/                Charts, results.csv (committed)
├── main.py                 CLI pipeline runner
└── app.py                  This Streamlit app
    """, language="")
