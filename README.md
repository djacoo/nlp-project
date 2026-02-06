<p align="center">
  <h1 align="center">NLP Document Similarity Matcher</h1>
  <p align="center">
    Document similarity matching on the Reuters corpus using TF-IDF vectorization and cosine similarity.
    <br />
    <em>NLP Course — General Assignment, Option C</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/NLTK-Reuters%20Corpus-154f3c" alt="NLTK">
  <img src="https://img.shields.io/badge/scikit--learn-TF--IDF-f7931e?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  <img src="https://img.shields.io/badge/tests-3%2F3%20passing-brightgreen" alt="Tests Passing">
</p>

---

## Overview

This project implements a complete document similarity matching system as described in **Option C** of the NLP course general assignment. Given a user-provided document and a match percentile, the system identifies and returns all documents in the Reuters corpus whose cosine similarity (computed over TF-IDF vectors) exceeds the specified percentile threshold.

The entire pipeline operates on raw, unprocessed text — no stopword elimination is applied, as required by the assignment specification.

### Workflow

1. The Reuters corpus (~10,788 news articles) is loaded via NLTK
2. TF-IDF vectors are computed for every document in the corpus
3. The user provides a query document and a percentile threshold (0–100)
4. Cosine similarity is calculated between the query and each corpus document
5. Documents scoring above the percentile threshold are returned, sorted by descending similarity

## Project Structure

```
nlp-project/
├── src/
│   ├── __init__.py              # Package init and version
│   ├── main.py                  # Entry point and CLI interface
│   ├── corpus_loader.py         # Reuters corpus download and loading
│   └── document_matcher.py      # TF-IDF vectorization and cosine similarity
├── tests/
│   ├── __init__.py
│   ├── test_document_matcher.py # Unit tests for DocumentMatcher
│   ├── test_integration.py      # Integration tests on full corpus
│   └── test_sample.txt          # Sample document for file input testing
├── docs/
│   ├── ALGORITHM_EXPLAINED.md   # Detailed walkthrough of the algorithm
│   └── TESTING.md               # Test methodology and results
├── assignment/
│   └── assignment.pdf           # Original assignment specification
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip

### Install

```bash
# Clone the repository
git clone https://github.com/djacoo/nlp-project.git
cd nlp-project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The Reuters corpus (~2 MB) is downloaded automatically on first run via NLTK.

### Run

```bash
python src/main.py
```

The program will prompt for:
1. **Input method** — paste text directly or provide a path to a `.txt` file
2. **Match percentile** — a value between 0 and 100

### Example

```
======================================================================
INITIALIZING DOCUMENT SIMILARITY MATCHER
======================================================================

[1/4] Loading Reuters corpus...
Loaded 10788 documents from Reuters corpus

[2/4] Computing TF-IDF vectors...
TF-IDF matrix shape: (10788, 30916)
  - 10788 documents
  - 30916 unique terms in vocabulary

[3/4] Getting user input...

How would you provide the document?
1. Enter text directly
2. Provide a file path

Enter (1 or 2): 2
Enter the file path: tests/test_sample.txt

[4/4] Finding similar documents...

Similarity distribution: 4743/10788 documents share terms with query
Percentile threshold (70.0th): 0.0119

======================================================================
Documents matching above 70.0th percentile
Found 3237 matching documents
======================================================================

  1. training/144         | Similarity: 0.3247
  2. test/18911            | Similarity: 0.3195
  3. training/3734         | Similarity: 0.2978
  ...
```

## Tests

```bash
# Unit tests
python -m unittest tests.test_document_matcher -v

# Integration tests
python tests/test_integration.py
```

See [`docs/TESTING.md`](docs/TESTING.md) for full test methodology and results.

---

<p align="center"><em>Jacopo Parretti - VR536104 — NLP Project - MsC in Artificial Intelligence - 2025-2026</em></p>
