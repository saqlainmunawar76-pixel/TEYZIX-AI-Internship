AI-Powered Document Summarization System
==========================================
Task ID  : AI-INT-1
Domain   : Artificial Intelligence / NLP
Batch    : TEYZIX CORE Internship – June 2026
Deadline : 19 June 2026

────────────────────────────────────────────
DESCRIPTION
────────────────────────────────────────────
This project implements an Extractive Text Summarization system using
Python and NLTK. It accepts input from plain-text files, PDF files, or
direct user input, preprocesses the text using standard NLP techniques,
scores sentences using TF-IDF or frequency-based methods, and generates
a concise summary along with keyword analytics.

────────────────────────────────────────────
PROJECT STRUCTURE
────────────────────────────────────────────
AI-Document-Summarizer/
├── main.py              ← Entry point (CLI + Interactive mode)
├── summarizer.py        ← Core NLP & summarization logic
├── requirements.txt     ← Python dependencies
├── sample_docs/
│   ├── healthcare_ai.txt
│   └── climate_change.txt
├── outputs/             ← Generated summary files saved here
└── README.txt           ← This file

────────────────────────────────────────────
REQUIREMENTS
────────────────────────────────────────────
Python 3.8+
  pip install -r requirements.txt

────────────────────────────────────────────
HOW TO RUN
────────────────────────────────────────────

1. Interactive Mode (recommended for beginners):
   python main.py
   → Follow the on-screen menu to paste text or load a file.

2. Command-Line Mode:

   Summarize a text file:
     python main.py --file sample_docs/healthcare_ai.txt

   Summarize with custom settings:
     python main.py --file sample_docs/climate_change.txt --sentences 4 --method combined --output outputs/my_summary.txt

   Summarize raw text:
     python main.py --text "Your long text here..." --sentences 2

   Arguments:
     --file       Path to .txt or .pdf file
     --text       Raw text string
     --sentences  Number of sentences in output (default: 3)
     --method     tfidf | frequency | combined  (default: tfidf)
     --output     Output file path (optional)

────────────────────────────────────────────
FEATURES
────────────────────────────────────────────
✔ Accepts .txt files, .pdf files, and direct text input
✔ Full NLP preprocessing (lowercase, tokenize, stopword removal, segmentation)
✔ TF-IDF sentence scoring
✔ Frequency-based sentence scoring
✔ Combined scoring (TF-IDF + Frequency)
✔ Adjustable summary length
✔ Word frequency analysis and top keyword extraction
✔ Sentence importance scoring
✔ Summary export to .txt file
✔ Clean error handling throughout

────────────────────────────────────────────
MODULES OVERVIEW
────────────────────────────────────────────
summarizer.py
  preprocess_text()          – NLP pipeline
  word_frequency_analysis()  – Keyword extraction
  compute_tf_idf()           – TF-IDF scoring
  frequency_based_scoring()  – Frequency scoring
  extractive_summarize()     – Main summarization function
  load_text()                – Universal file loader
  export_summary()           – File export

main.py
  interactive_mode()  – Menu-driven interface
  cli_mode()          – Argument-based interface

────────────────────────────────────────────
EVALUATION COVERAGE
────────────────────────────────────────────
NLP Preprocessing  25% ✔  (preprocess_text in summarizer.py)
Summarization Logic 25% ✔  (TF-IDF + frequency + combined)
Code Structure     20% ✔  (modular, documented functions)
Output Quality     15% ✔  (compression ratio shown, keywords listed)
Error Handling     10% ✔  (try/except on all I/O and NLP operations)
Documentation       5% ✔  (docstrings + this README)

────────────────────────────────────────────
AUTHOR
────────────────────────────────────────────
Saqlain | TEYZIX CORE Internship | June Batch 2026
