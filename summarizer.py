"""
AI-Powered Document Summarization System
Task ID: AI-INT-1
Domain: Artificial Intelligence / NLP
Author: TEYZIX CORE Internship - June Batch 2026
"""

import re
import math
import string
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# ─── Ensure NLTK data is available ────────────────────────────────────────────
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> dict:
    """
    Full NLP preprocessing pipeline.
    Returns a dict with cleaned tokens, sentences, and stopword-filtered tokens.
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    # Sentence segmentation (keep original sentences for output)
    sentences = sent_tokenize(text)

    # Lowercasing
    text_lower = text.lower()

    # Tokenization
    tokens = word_tokenize(text_lower)

    # Remove punctuation & digits
    tokens_clean = [t for t in tokens if t not in string.punctuation and not t.isdigit()]

    # Stopword removal
    stop_words = set(stopwords.words("english"))
    tokens_filtered = [t for t in tokens_clean if t not in stop_words]

    return {
        "sentences": sentences,
        "tokens": tokens_clean,
        "filtered_tokens": tokens_filtered,
        "word_count": len(tokens_clean),
        "sentence_count": len(sentences),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ANALYTICS MODULE
# ══════════════════════════════════════════════════════════════════════════════

def word_frequency_analysis(filtered_tokens: list, top_n: int = 10) -> dict:
    """Returns word frequency count and top-N most frequent keywords."""
    freq = Counter(filtered_tokens)
    top_keywords = freq.most_common(top_n)
    return {"frequency_map": dict(freq), "top_keywords": top_keywords}


def compute_tf_idf(sentences: list, filtered_tokens: list) -> dict:
    """
    Lightweight single-document TF-IDF scoring.
    Treats each sentence as a 'document' for IDF computation.
    """
    stop_words = set(stopwords.words("english"))
    N = len(sentences)

    # Build per-sentence token bags
    sent_bags = []
    for sent in sentences:
        tokens = word_tokenize(sent.lower())
        bag = [t for t in tokens if t not in string.punctuation
               and t not in stop_words and not t.isdigit()]
        sent_bags.append(Counter(bag))

    # IDF: how many sentences contain each word
    df = Counter()
    for bag in sent_bags:
        for word in bag:
            df[word] += 1

    idf = {word: math.log((N + 1) / (count + 1)) + 1 for word, count in df.items()}

    # TF-IDF score per sentence
    sent_scores = {}
    for i, (sent, bag) in enumerate(zip(sentences, sent_bags)):
        total = sum(bag.values()) or 1
        score = sum((count / total) * idf.get(word, 0) for word, count in bag.items())
        sent_scores[i] = score

    return sent_scores


def frequency_based_scoring(sentences: list, freq_map: dict) -> dict:
    """Score each sentence by summing normalised word frequencies."""
    max_freq = max(freq_map.values(), default=1)
    norm_freq = {w: f / max_freq for w, f in freq_map.items()}

    stop_words = set(stopwords.words("english"))
    sent_scores = {}
    for i, sent in enumerate(sentences):
        tokens = word_tokenize(sent.lower())
        score = sum(norm_freq.get(t, 0) for t in tokens
                    if t not in string.punctuation and t not in stop_words)
        sent_scores[i] = score

    return sent_scores


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EXTRACTIVE SUMMARIZATION
# ══════════════════════════════════════════════════════════════════════════════

def extractive_summarize(text: str, num_sentences: int = 3,
                          method: str = "tfidf") -> dict:
    """
    Core extractive summarization.

    Parameters
    ----------
    text         : raw input text
    num_sentences: how many sentences to include in the summary
    method       : 'tfidf' | 'frequency' | 'combined'

    Returns a result dict with summary, scores, and analytics.
    """
    processed = preprocess_text(text)
    sentences = processed["sentences"]
    filtered_tokens = processed["filtered_tokens"]

    # Clamp summary length
    num_sentences = max(1, min(num_sentences, len(sentences)))

    # Analytics
    analytics = word_frequency_analysis(filtered_tokens)

    # Sentence scoring
    if method == "frequency":
        scores = frequency_based_scoring(sentences, analytics["frequency_map"])
    elif method == "tfidf":
        scores = compute_tf_idf(sentences, filtered_tokens)
    else:  # combined
        freq_scores = frequency_based_scoring(sentences, analytics["frequency_map"])
        tfidf_scores = compute_tf_idf(sentences, filtered_tokens)
        # Normalise and average
        def normalise(d):
            mx = max(d.values(), default=1) or 1
            return {k: v / mx for k, v in d.items()}
        f_norm = normalise(freq_scores)
        t_norm = normalise(tfidf_scores)
        scores = {i: (f_norm.get(i, 0) + t_norm.get(i, 0)) / 2
                  for i in range(len(sentences))}

    # Rank sentences by score, then restore original order
    top_indices = sorted(
        sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    )
    summary = " ".join(sentences[i] for i in top_indices)

    return {
        "original_text": text,
        "summary": summary,
        "original_word_count": processed["word_count"],
        "summary_word_count": len(word_tokenize(summary)),
        "original_sentence_count": processed["sentence_count"],
        "summary_sentence_count": num_sentences,
        "compression_ratio": round(
            len(word_tokenize(summary)) / max(processed["word_count"], 1) * 100, 1
        ),
        "sentence_scores": scores,
        "top_keywords": analytics["top_keywords"],
        "method": method,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  DATA INPUT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def load_from_txt(filepath: str) -> str:
    """Load text from a .txt file."""
    if not filepath.endswith(".txt"):
        raise ValueError("Only .txt files are supported by this function.")
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"File '{filepath}' is empty.")
    return content


def load_from_pdf(filepath: str) -> str:
    """Load text from a PDF file using PyPDF2."""
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is not installed. Run: pip install PyPDF2")

    text_parts = []
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

    full_text = "\n".join(text_parts).strip()
    if not full_text:
        raise ValueError("Could not extract text from the PDF (possibly scanned/image-based).")
    return full_text


def load_text(source: str, source_type: str = "auto") -> str:
    """
    Universal loader.
    source_type: 'txt' | 'pdf' | 'direct' | 'auto'
    """
    if source_type == "direct":
        return source.strip()

    if source_type == "auto":
        if source.endswith(".txt"):
            source_type = "txt"
        elif source.endswith(".pdf"):
            source_type = "pdf"
        else:
            # Treat as raw text
            return source.strip()

    if source_type == "txt":
        return load_from_txt(source)
    elif source_type == "pdf":
        return load_from_pdf(source)
    else:
        raise ValueError(f"Unsupported source_type: '{source_type}'")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FILE HANDLING / OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def export_summary(result: dict, output_path: str, fmt: str = "txt") -> None:
    """
    Export summary to a .txt file.
    fmt: 'txt' only (PDF export requires additional libraries).
    """
    if fmt == "txt":
        lines = [
            "=" * 60,
            "  AI-POWERED DOCUMENT SUMMARIZATION SYSTEM",
            "  TEYZIX CORE Internship | Task AI-INT-1",
            "=" * 60,
            "",
            f"Method Used     : {result['method'].upper()}",
            f"Original Words  : {result['original_word_count']}",
            f"Summary Words   : {result['summary_word_count']}",
            f"Compression     : {result['compression_ratio']}%",
            f"Sentences Used  : {result['summary_sentence_count']} / {result['original_sentence_count']}",
            "",
            "─" * 60,
            "SUMMARY",
            "─" * 60,
            result["summary"],
            "",
            "─" * 60,
            "TOP KEYWORDS",
            "─" * 60,
        ]
        for word, freq in result["top_keywords"]:
            lines.append(f"  {word:<20} freq: {freq}")

        lines += [
            "",
            "─" * 60,
            "ORIGINAL TEXT",
            "─" * 60,
            result["original_text"],
            "",
            "=" * 60,
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[✓] Summary exported → {output_path}")
    else:
        raise ValueError("Only 'txt' format is supported for export.")
