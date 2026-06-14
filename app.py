"""
app.py — Streamlit Web App
AI-Powered Document Summarization System
TEYZIX CORE Internship | Task AI-INT-1 | June Batch 2026
"""

import streamlit as st
import string
from collections import Counter
import math
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# ── NLTK Data ─────────────────────────────────────────────
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AI Document Summarizer — TEYZIX CORE",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stTextArea textarea { font-size: 14px; }
    .metric-card {
        background: #1E2530;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #2D3748;
    }
    .keyword-tag {
        display: inline-block;
        background: #1D4ED8;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 3px;
        font-size: 13px;
    }
    .header-banner {
        background: linear-gradient(135deg, #0F6E56, #1D4ED8);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# NLP FUNCTIONS
# ══════════════════════════════════════════════════════════

def preprocess(text):
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered = [t for t in tokens if t not in string.punctuation
                and t not in stop_words and not t.isdigit()]
    return sentences, filtered


def word_freq(filtered_tokens):
    return Counter(filtered_tokens)


def tfidf_scores(sentences):
    stop_words = set(stopwords.words("english"))
    N = len(sentences)
    bags = []
    for s in sentences:
        tokens = word_tokenize(s.lower())
        bag = Counter([t for t in tokens if t not in string.punctuation
                       and t not in stop_words and not t.isdigit()])
        bags.append(bag)
    df = Counter()
    for bag in bags:
        for w in bag:
            df[w] += 1
    idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}
    scores = {}
    for i, (s, bag) in enumerate(zip(sentences, bags)):
        total = sum(bag.values()) or 1
        scores[i] = sum((c / total) * idf.get(w, 0) for w, c in bag.items())
    return scores


def freq_scores(sentences, freq_map):
    max_f = max(freq_map.values(), default=1)
    norm = {w: f / max_f for w, f in freq_map.items()}
    stop_words = set(stopwords.words("english"))
    scores = {}
    for i, s in enumerate(sentences):
        tokens = word_tokenize(s.lower())
        scores[i] = sum(norm.get(t, 0) for t in tokens
                        if t not in string.punctuation and t not in stop_words)
    return scores


def summarize(text, n, method):
    sentences, filtered = preprocess(text)
    freq = word_freq(filtered)
    n = max(1, min(n, len(sentences)))

    if method == "TF-IDF":
        scores = tfidf_scores(sentences)
    elif method == "Frequency":
        scores = freq_scores(sentences, freq)
    else:
        def norm(d):
            mx = max(d.values(), default=1) or 1
            return {k: v / mx for k, v in d.items()}
        f = norm(freq_scores(sentences, freq))
        t = norm(tfidf_scores(sentences))
        scores = {i: (f.get(i, 0) + t.get(i, 0)) / 2 for i in range(len(sentences))}

    top = sorted(sorted(scores, key=scores.get, reverse=True)[:n])
    summary = " ".join(sentences[i] for i in top)
    return summary, freq, scores, sentences


# ══════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="header-banner">
    <h1 style="color:white; margin:0;">🤖 AI Document Summarization System</h1>
    <p style="color:#A0AEC0; margin:5px 0 0 0;">TEYZIX CORE Internship | Task AI-INT-1 | June Batch 2026</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/TEYZIX-CORE-green?style=for-the-badge")
    st.markdown("### ⚙️ Settings")

    method = st.selectbox(
        "Summarization Method",
        ["TF-IDF", "Frequency", "Combined"],
        help="TF-IDF = best accuracy | Frequency = fast | Combined = balanced"
    )

    num_sentences = st.slider(
        "Number of sentences in summary",
        min_value=1, max_value=10, value=3
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This AI system uses **NLP** to extract the most important sentences from any document.
    
    **Methods:**
    - 🧠 TF-IDF Scoring
    - 📊 Frequency Scoring  
    - ⚡ Combined Method
    
    **Built with:** Python + NLTK
    """)
    st.markdown("---")
    st.markdown("**👨‍💻 Saqlain**  \nAI Intern — TEYZIX CORE")

# Main tabs
tab1, tab2 = st.tabs(["📝 Text Input", "📄 File Upload"])

with tab1:
    text_input = st.text_area(
        "Paste your document here:",
        height=200,
        placeholder="Paste any long article, report, or document here and click Summarize..."
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        summarize_btn = st.button("🚀 Summarize", type="primary", use_container_width=True)

    if summarize_btn:
        if not text_input.strip():
            st.error("⚠️ Please paste some text first!")
        elif len(text_input.split()) < 30:
            st.warning("⚠️ Text is too short. Please paste a longer document.")
        else:
            with st.spinner("🧠 Analyzing document..."):
                summary, freq, scores, sentences = summarize(text_input, num_sentences, method)

            # Metrics
            orig_words = len(text_input.split())
            summ_words = len(summary.split())
            compression = round(summ_words / orig_words * 100, 1)

            st.markdown("### 📊 Analytics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original Words", orig_words)
            c2.metric("Summary Words", summ_words)
            c3.metric("Compression", f"{compression}%")
            c4.metric("Method Used", method)

            # Summary
            st.markdown("### ✅ Summary")
            st.success(summary)

            # Keywords
            st.markdown("### 🔑 Top Keywords")
            top_kw = freq.most_common(10)
            kw_html = " ".join(
                f'<span class="keyword-tag">{w} ({c})</span>'
                for w, c in top_kw
            )
            st.markdown(kw_html, unsafe_allow_html=True)

            # Original vs Summary
            st.markdown("### 🔍 Original vs Summary")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original Text**")
                st.text_area("", text_input, height=200, disabled=True, label_visibility="collapsed")
            with col_b:
                st.markdown("**Summary**")
                st.text_area("", summary, height=200, disabled=True, label_visibility="collapsed")

            # Download
            st.markdown("### 💾 Export Summary")
            export_text = f"""AI DOCUMENT SUMMARIZATION SYSTEM — TEYZIX CORE
Task AI-INT-1 | Method: {method}
Original Words: {orig_words} | Summary Words: {summ_words} | Compression: {compression}%

SUMMARY:
{summary}

TOP KEYWORDS:
{chr(10).join(f'{w}: {c}' for w, c in top_kw)}

ORIGINAL TEXT:
{text_input}
"""
            st.download_button(
                "⬇️ Download Summary (.txt)",
                export_text,
                file_name="summary_output.txt",
                mime="text/plain"
            )

with tab2:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded:
        text_file = uploaded.read().decode("utf-8", errors="ignore")
        st.success(f"✅ File loaded: {uploaded.name} ({len(text_file.split())} words)")
        if st.button("🚀 Summarize File", type="primary"):
            with st.spinner("🧠 Analyzing document..."):
                summary, freq, scores, sentences = summarize(text_file, num_sentences, method)

            orig_words = len(text_file.split())
            summ_words = len(summary.split())
            compression = round(summ_words / orig_words * 100, 1)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original Words", orig_words)
            c2.metric("Summary Words", summ_words)
            c3.metric("Compression", f"{compression}%")
            c4.metric("Method", method)

            st.markdown("### ✅ Summary")
            st.success(summary)

            top_kw = freq.most_common(10)
            st.markdown("### 🔑 Top Keywords")
            kw_html = " ".join(
                f'<span class="keyword-tag">{w} ({c})</span>'
                for w, c in top_kw
            )
            st.markdown(kw_html, unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download Summary (.txt)",
                summary,
                file_name="summary_output.txt"
            )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4A5568;'>Built by Saqlain | TEYZIX CORE AI Internship | June 2026 | Task AI-INT-1</p>",
    unsafe_allow_html=True
)
