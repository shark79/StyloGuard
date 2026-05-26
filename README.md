# 🛡️ StyloGuard — Stylometric Authorship Verification

**StyloGuard** is an AI-powered academic integrity tool that analyzes and verifies authorship by capturing unique writing style fingerprints. Built with Streamlit and fine-tuned deep learning models, it helps educators detect stylistic inconsistencies across student submissions.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Problem Statement

Traditional plagiarism checkers miss AI-generated or paraphrased content. StyloGuard adds a complementary layer of verification — comparing **how** something is written, not just what is written, using 10 stylometric features unique to each writer.

---

## ✨ Features

| Feature | Description |
|---|---|
| ✍️ Direct Analysis | Extract 10 stylometric features from any essay instantly |
| 🔁 Comparative Verification | Compare a submission against a student's reference essays |
| 🧠 Deep Learning Model | Fine-tuned triplet-loss model for stylometric embedding similarity |
| 🗃️ SQL-backed Storage | Automated pipeline linking stylometric indexes to Student IDs |
| 📊 Visual Insights | Side-by-side feature comparison charts |

---

## 🧮 Stylometric Features Analyzed

- Average sentence length
- Type-token ratio (vocabulary richness)
- Punctuation frequency
- Function word usage
- Average word length
- Passive voice ratio
- Sentence complexity
- Paragraph structure
- Lexical diversity
- POS tag distributions

---

## 🏗️ Architecture

```
Student Essay Input
       │
       ▼
Feature Extractor (10 stylometric features)
       │
       ├──▶ Direct Analysis Mode → Feature Report
       │
       └──▶ Comparison Mode → Fine-tuned Triplet Model → Similarity Score
                                       │
                               SQL Pipeline (Student ID → Style Index)
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/shark79/StyloGuard.git
cd StyloGuard
pip install -r requirements.txt
streamlit run StyloGuard.py
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| Streamlit | Web app UI |
| PyTorch | Fine-tuned triplet-loss model |
| SQLite | Stylometric index storage |
| scikit-learn | Feature engineering |
| spaCy / NLTK | NLP preprocessing |

---

## 📌 Use Cases

- University essay verification workflows
- Writing style tracking across a semester
- Detecting AI-ghostwritten submissions

---

## 👤 Author

**Shashank Jamkhandi** — AI Engineer  
[LinkedIn](https://www.linkedin.com/in/sjam) | [GitHub](https://github.com/shark79)
