🤖 AI-Powered Document Summarization System
TEYZIX CORE Internship Program — June Batch 2026

Task ID: AI-INT-1 | Domain: Artificial Intelligence / NLP

Difficulty: Intermediate | Submitted by: Saqlain munwar

🌐 Live Demo: https://teyzix-ai-internship-kdgby3dah88jk9nkfynsku.streamlit.app

📌 About The Project

Organizations deal with large volumes of documents such as reports, emails, and articles. Manual summarization is time-consuming and inconsistent. This project solves that problem by building an AI-based extractive summarization system that automatically extracts the most important sentences from any document while preserving key information.


✨ Features


📄 Multiple Input Sources — Text files, PDF files, or direct text input
🔤 Full NLP Preprocessing — Lowercasing, tokenization, stopword removal, sentence segmentation
🧠 3 Summarization Methods:

TF-IDF Based Scoring
Frequency Based Scoring
Combined Method (TF-IDF + Frequency)



📊 Analytics Module — Word frequency analysis, keyword extraction, sentence importance scoring
⚙️ Adjustable Summary Length — Choose how many sentences you want
💾 Export Summary — Save output as .txt file
✅ Error Handling — Clean and robust error management throughout



🛠️ Technologies Used

TechnologyPurposePython 3.12Core programming languageNLTKNatural Language ProcessingPyPDF2PDF file readingTF-IDF AlgorithmSentence scoring & rankingFrequency AnalysisKeyword extraction


📁 Project Structure

AI-Document-Summarizer/
├── main.py              ← Entry point (CLI + Interactive mode)
├── summarizer.py        ← Core NLP & summarization logic
├── requirements.txt     ← Python dependencies
├── README.md            ← Project documentation
├── sample_docs/
│   ├── healthcare_ai.txt      ← Sample document 1
│   └── climate_change.txt     ← Sample document 2
└── outputs/
    └── summary_output.txt     ← Generated summaries

