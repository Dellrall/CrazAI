"""
Sentiment Analysis — Streamlit Demo App
========================================
Demonstrates the trained NB and SVM models interactively.

Run with:
    streamlit run app.py
"""

import os
import sys
import time

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.crawler.dataset_loader import DatasetLoader
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.tokenizer import TextTokenizer
from src.models.naive_bayes import NaiveBayesModel
from src.models.svm_model import SVMModel

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
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .big-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }
    .result-card {
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    .result-positive {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        color: #065f46;
        border: 2px solid #34d399;
    }
    .result-negative {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #7f1d1d;
        border: 2px solid #f87171;
    }
    .metric-label { color: #64748b; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_and_train_models(sample_size: int = 10000):
    """Load IMDB data and train NB + SVM models. Cached after first run."""
    loader = DatasetLoader()
    df = loader.load_imdb('data/imdb')
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    cleaner = TextCleaner()
    tokenizer = TextTokenizer()
    df['cleaned'] = df['text'].apply(cleaner.clean)
    df['tokens']  = df['cleaned'].apply(tokenizer.preprocess)
    df['processed'] = df['tokens'].apply(lambda t: ' '.join(t))

    texts  = df['processed'].tolist()
    labels = df['label'].tolist()

    nb  = NaiveBayesModel(max_features=10000)
    svm = SVMModel(max_features=10000)

    nb_metrics  = nb.train(texts, labels, test_size=0.2)
    svm_metrics = svm.train(texts, labels, test_size=0.2)

    return nb, svm, nb_metrics, svm_metrics, df, cleaner, tokenizer


@st.cache_resource(show_spinner=False)
def load_csv_results():
    path = 'outputs/results.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    sample_size = st.selectbox(
        "Training sample size",
        [2000, 5000, 10000, 20000],
        index=2,
        help="More samples = better accuracy but slower startup"
    )
    model_choice = st.radio(
        "Model to use",
        ["Naïve Bayes", "SVM", "Both (compare)"],
        index=2
    )
    st.divider()
    st.markdown("### 📋 About")
    st.markdown("""
    This app demonstrates two sentiment analysis models trained on the **IMDB Large Movie Review Dataset** (50K reviews).

    **Models:**
    - **Naïve Bayes** — fast probabilistic baseline
    - **SVM** — linear support vector machine

    **Pipeline:** Clean → Tokenize → TF-IDF → Classify
    """)

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────

st.markdown('<div class="big-title">🎭 Sentiment Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NLP-powered sentiment analysis using Naïve Bayes & SVM</div>', unsafe_allow_html=True)

with st.spinner("Loading & training models on IMDB data... (first run takes ~30 seconds)"):
    nb, svm, nb_metrics, svm_metrics, df, cleaner, tokenizer = load_and_train_models(sample_size)

st.success(f"✅ Models ready — trained on {int(sample_size * 0.8):,} samples, tested on {int(sample_size * 0.2):,}")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab_demo, tab_stats, tab_explore = st.tabs(["🔍 Try It", "📊 Model Stats", "📂 Dataset"])

# ────────── Tab 1: Try It ──────────────────────────────────────────────────────

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

    analyse_btn = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)

    if analyse_btn and user_input.strip():
        # Preprocess
        cleaned = cleaner.clean(user_input)
        tokens  = tokenizer.preprocess(cleaned)
        processed = ' '.join(tokens)

        st.divider()

        if model_choice == "Both (compare)":
            col_nb, col_svm = st.columns(2)

            for col, model, name in [(col_nb, nb, "Naïve Bayes"), (col_svm, svm, "SVM")]:
                with col:
                    st.markdown(f"#### {name}")
                    probs = model.predict_proba([processed])[0]
                    pred  = int(np.argmax(probs))
                    conf  = float(np.max(probs))
                    label = "Positive 😊" if pred == 1 else "Negative 😟"
                    css_class = "result-positive" if pred == 1 else "result-negative"
                    st.markdown(
                        f'<div class="result-card {css_class}">{label}<br>'
                        f'<span style="font-size:1.5rem">{conf:.1%} confident</span></div>',
                        unsafe_allow_html=True
                    )
                    # Probability bar
                    fig = go.Figure(go.Bar(
                        x=['Negative', 'Positive'],
                        y=[probs[0], probs[1]],
                        marker_color=['#f87171', '#34d399'],
                        text=[f'{probs[0]:.1%}', f'{probs[1]:.1%}'],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        height=200, margin=dict(t=10, b=10, l=10, r=10),
                        yaxis=dict(range=[0, 1], showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            model = nb if model_choice == "Naïve Bayes" else svm
            probs = model.predict_proba([processed])[0]
            pred  = int(np.argmax(probs))
            conf  = float(np.max(probs))
            label = "Positive 😊" if pred == 1 else "Negative 😟"
            css_class = "result-positive" if pred == 1 else "result-negative"

            st.markdown(
                f'<div class="result-card {css_class}">{label}<br>'
                f'<span style="font-size:2rem">{conf:.1%} confident</span></div>',
                unsafe_allow_html=True
            )
            col1, col2 = st.columns(2)
            col1.metric("Prediction", "Positive" if pred == 1 else "Negative")
            col2.metric("Confidence", f"{conf:.1%}")

        with st.expander("🔧 What the pipeline did"):
            st.write(f"**Original:** {user_input[:200]}")
            st.write(f"**After cleaning:** {cleaned[:200]}")
            st.write(f"**Tokens:** {tokens[:20]}")

    elif analyse_btn:
        st.warning("Please enter some text first.")

# ────────── Tab 2: Model Stats ─────────────────────────────────────────────────

with tab_stats:
    st.subheader("📊 Model Performance on Test Set")

    col1, col2 = st.columns(2)

    for col, metrics, name in [(col1, nb_metrics, "Naïve Bayes"), (col2, svm_metrics, "SVM")]:
        with col:
            st.markdown(f"#### {name}")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy",  f"{metrics['accuracy']:.2%}")
            m2.metric("F1 Score",  f"{metrics['f1_score']:.2%}")
            m3, m4 = st.columns(2)
            m3.metric("Precision", f"{metrics['precision']:.2%}")
            m4.metric("Recall",    f"{metrics['recall']:.2%}")
            m5, m6 = st.columns(2)
            m5.metric("Train size", f"{metrics['n_train']:,}")
            m6.metric("Test size",  f"{metrics['n_test']:,}")

    st.divider()

    # Comparison bar chart
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
        color_discrete_sequence=["#6366f1", "#ec4899"],
        text_auto=".3f"
    )
    fig.update_layout(
        yaxis=dict(range=[0.8, 1.0]),
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # CSV results if available
    saved = load_csv_results()
    if saved is not None:
        st.divider()
        st.subheader("📄 Saved Full Pipeline Results (`outputs/results.csv`)")
        st.dataframe(saved, use_container_width=True)

# ────────── Tab 3: Dataset ─────────────────────────────────────────────────────

with tab_explore:
    st.subheader("📂 Dataset Overview — IMDB Movie Reviews")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{len(df):,}")
    pos = df['label'].sum()
    c2.metric("Positive Reviews", f"{pos:,}", f"{pos/len(df):.1%}")
    neg = len(df) - pos
    c3.metric("Negative Reviews", f"{neg:,}", f"{neg/len(df):.1%}")

    # Class balance pie
    fig_pie = px.pie(
        names=["Positive 😊", "Negative 😟"],
        values=[pos, neg],
        color_discrete_sequence=["#34d399", "#f87171"],
        title="Class Distribution"
    )
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Word count distribution
    df['word_count'] = df['processed'].apply(lambda x: len(x.split()))
    fig_hist = px.histogram(
        df, x='word_count', color=df['label'].map({1: 'Positive', 0: 'Negative'}),
        nbins=50, title="Word Count Distribution (after preprocessing)",
        color_discrete_map={"Positive": "#34d399", "Negative": "#f87171"},
        labels={'word_count': 'Word Count', 'color': 'Sentiment'}
    )
    fig_hist.update_layout(height=280, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist, use_container_width=True)

    # Sample reviews
    st.subheader("Sample Reviews")
    n_samples = st.slider("Show N samples", 5, 50, 10)
    sample_df = df[['text', 'label']].sample(n=n_samples, random_state=42).copy()
    sample_df['label'] = sample_df['label'].map({1: '😊 Positive', 0: '😟 Negative'})
    sample_df['text'] = sample_df['text'].apply(lambda x: x[:150] + '...' if len(x) > 150 else x)
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
