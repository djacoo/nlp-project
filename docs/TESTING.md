# Testing Documentation

## NLP Document Similarity Matcher — Option C

This document describes the testing methodology, environment, and results for the document similarity matching system.

---

## Test Environment

| Component | Version / Details |
|-----------|-------------------|
| Operating System | macOS Darwin 25.2.0 |
| Python | 3.12 |
| NLTK | 3.9.2 |
| scikit-learn | 1.8.0 |
| NumPy | (bundled with scikit-learn) |
| Corpus | Reuters — 10,788 documents, 30,916 unique terms |

---

## Unit Tests

**Location:** `tests/test_document_matcher.py`

The unit tests operate on a small, self-contained corpus of four documents. This makes them fast to run and independent of the Reuters download. Each test targets a specific aspect of the `DocumentMatcher` class.

### test_fit_corpus

Verifies that the TF-IDF vectorizer is correctly initialized and trained on the corpus.

**Assertions:**
- The vectorizer object is not `None` after fitting
- The corpus vectors matrix is not `None`
- The stored corpus length matches the input (4 documents)

**Result:** PASS — produced a 4×22 TF-IDF matrix (4 documents, 22 unique terms).

### test_find_similar_documents

Tests the similarity matching functionality with the query *"Machine learning and artificial intelligence"*.

**Assertions:**
- The return type is a list of `(doc_id, score)` tuples
- All similarity scores fall within the valid range [0.0, 1.0]
- At least one result is returned

**Result:** PASS — returned valid matches with scores in range.

### test_percentile_filtering

Checks that the percentile threshold correctly controls the number of results. The same query is run at the 50th and 90th percentile.

**Assertions:**
- The 50th percentile returns at least as many documents as the 90th percentile

**Result:** PASS — confirmed the expected inverse relationship between percentile and result count.

**Run command:**
```bash
python -m unittest tests.test_document_matcher -v
```

---

## Integration Tests

**Location:** `tests/test_integration.py`

The integration tests exercise the full pipeline against the complete Reuters corpus (10,788 documents). They verify that all components work together correctly at scale.

### Test 1 — Corpus Loading

Loads the Reuters corpus via `CorpusLoader` and checks that the expected number of documents is returned.

**Result:** PASS — loaded 10,788 documents with matching document IDs.

### Test 2 — TF-IDF Computation

Fits the `DocumentMatcher` on the full corpus and verifies the resulting matrix dimensions.

**Result:** PASS — TF-IDF matrix shape: 10,788 × 30,916.

### Test 3 — Financial Query

Query: *"Oil prices surged today as OPEC announced production cuts. Market analysts predict volatility."*
Percentile: 70th

**Result:** PASS
- Found 3,237 matching documents (approximately 30% of the corpus)
- Top match: `training/144` with similarity 0.3247
- Threshold value: 0.0119

### Test 4 — Percentile Filtering

The same financial query is tested at different percentile levels to verify the filtering mechanism.

| Percentile | Threshold | Documents Returned | % of Corpus |
|------------|-----------|-------------------|-------------|
| 50th | 0.0000 | 10,788 | 100% |
| 90th | 0.0348 | 1,079 | 10.0% |

The 50th percentile threshold of 0.0 is expected behavior: due to the sparse nature of TF-IDF vectors, the majority of documents in the corpus have zero similarity to any given short query. The median similarity is therefore 0.0, and the filter includes all documents. Meaningful filtering begins around the 55th–60th percentile for most queries.

**Result:** PASS — confirmed that a lower percentile always yields at least as many results as a higher one.

### Test 5 — Cross-Topic Query

Query: *"The government announced new legislation regarding environmental regulations."*
Percentile: 70th

**Result:** PASS
- Found 3,237 matching documents
- Top match: `training/10129` with similarity 0.2175
- The top match differs from the financial query, confirming that the system distinguishes between topics

### Test 6 — Boundary: 0th Percentile

Verifies that a 0th percentile threshold returns the entire corpus (no filtering).

**Result:** PASS — returned all 10,788 documents.

### Test 7 — Boundary: 100th Percentile

Verifies that a 100th percentile threshold returns only the single best-matching document.

**Result:** PASS — returned 1 document (threshold equal to the maximum similarity score of 0.3247).

### Test 8 — Result Ordering

Checks that the returned results are sorted in descending order of similarity score.

**Result:** PASS — each score in the result list is greater than or equal to the next.

### Test 9 — File Input

Loads a query from `tests/test_sample.txt` (a short text about oil prices) and verifies that the system processes file-based input correctly.

**Result:** PASS — found 4,315 matches at the 60th percentile.

**Run command:**
```bash
python tests/test_integration.py
```

---

## Edge Cases

The following edge cases were tested manually to ensure robust behavior.

### Empty or out-of-vocabulary query

If the user provides a query whose terms do not appear anywhere in the corpus vocabulary (or an empty string), the resulting TF-IDF vector contains only zeros. In this case, the system returns an empty result set with a warning message rather than returning all documents with a meaningless similarity of 0.0.

### No shared vocabulary

A query composed entirely of words absent from the Reuters corpus (for instance, modern terms like "cryptocurrency" or "blockchain", which do not appear in a 1987 news archive) is handled identically to the empty query case above — the system detects the zero vector and returns early.

### Identical document

When the query is identical to a document in the corpus, cosine similarity returns exactly 1.0, as expected (the angle between identical vectors is 0°).

### Very short queries

Queries consisting of only one or two words are processed normally. The TF-IDF vector will have very few non-zero entries, and similarity scores will tend to be lower, but the computation remains correct.

### Very long queries

Queries of several thousand words are handled without issue. The sparse matrix representation ensures that memory and computation scale with the number of non-zero terms rather than the total vocabulary size.

---

## Test Summary

| Category | Passed | Failed |
|----------|--------|--------|
| Unit Tests | 3 | 0 |
| Integration Tests | 9 | 0 |
| Edge Cases | 5 | 0 |
| **Total** | **17** | **0** |

---

## Performance

The following timings were observed during integration testing on a standard laptop (macOS, Apple Silicon):

| Operation | Duration |
|-----------|----------|
| Corpus loading (first run, includes download) | 2–3 seconds |
| Corpus loading (subsequent runs) | < 1 second |
| TF-IDF vectorization (full corpus) | 2–3 seconds |
| Single query matching | < 0.1 seconds |
| Peak memory usage | ~50–100 MB |

These figures are well within acceptable limits for a corpus of this size. The sparse matrix format used by scikit-learn keeps memory consumption modest despite the large vocabulary (~30,000 terms).

---

## Reproducing the Tests

```bash
# Unit tests (no corpus download required)
python -m unittest tests.test_document_matcher -v

# Integration tests (downloads Reuters corpus on first run)
python tests/test_integration.py

# Manual testing
python src/main.py
```

---
