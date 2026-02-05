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

### How it works

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

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/djacoo/nlp-project.git
cd nlp-project

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

The Reuters corpus (~2 MB) is downloaded automatically on first run via NLTK.

### Running the Program

```bash
python src/main.py
```

The program will prompt for:
1. **Input method** — paste text directly or provide a path to a `.txt` file
2. **Match percentile** — a value between 0 and 100

### Example Session

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

## Running Tests

```bash
# Unit tests
python -m unittest tests.test_document_matcher -v

# Integration tests (requires Reuters corpus download)
python tests/test_integration.py
```

See [`docs/TESTING.md`](docs/TESTING.md) for full test methodology and results.

## Technical Details

### Dependencies

| Library | Purpose |
|---------|---------|
| **NLTK** (>=3.8.1) | Access to the Reuters newswire corpus |
| **scikit-learn** (>=1.3.0) | `TfidfVectorizer` and `cosine_similarity` |
| **NumPy** (>=1.24.0) | Percentile computation and array operations |

### Algorithm Summary

The matching pipeline follows these steps:

1. **Corpus loading** — Raw text of all 10,788 Reuters documents is retrieved through `nltk.corpus.reuters`
2. **TF-IDF vectorization** — `TfidfVectorizer` (with default settings, no stopword removal) builds the vocabulary and transforms each document into a sparse TF-IDF vector
3. **Query transformation** — The user's document is transformed using the same fitted vectorizer, ensuring a shared vocabulary space
4. **Cosine similarity** — `sklearn.metrics.pairwise.cosine_similarity` computes the similarity between the query vector and every corpus vector
5. **Percentile filtering** — `numpy.percentile` determines the threshold value; only documents meeting or exceeding this threshold are retained
6. **Sorting** — Results are ordered by similarity score in descending order

For a detailed explanation with worked examples, see [`docs/ALGORITHM_EXPLAINED.md`](docs/ALGORITHM_EXPLAINED.md).

### Design Decisions

- **No stopword removal** — The assignment explicitly requires that no stopword elimination phase is applied. All words contribute to the TF-IDF vectors.
- **Sparse matrix representation** — The TF-IDF matrix is stored in compressed sparse row (CSR) format, keeping memory usage manageable despite the large vocabulary (~30,000 terms).
- **Percentile-based thresholding** — Rather than requiring the user to guess an absolute similarity cutoff, the percentile approach adapts to the actual score distribution for any given query.

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/ALGORITHM_EXPLAINED.md`](docs/ALGORITHM_EXPLAINED.md) | Step-by-step explanation of TF-IDF, cosine similarity, and the full matching pipeline with worked examples |
| [`docs/TESTING.md`](docs/TESTING.md) | Test environment, methodology, unit and integration test results, edge cases, and performance notes |
| [`assignment/assignment.pdf`](assignment/assignment.pdf) | Original assignment specification |

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

<p align="center"><em>Jacopo Parretti — NLP Course, 2026</em></p>
