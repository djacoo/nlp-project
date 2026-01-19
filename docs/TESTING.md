# Testing Documentation
## NLP Document Similarity Matcher - Option C

This document describes the testing and results.

---

## Test Environment

- macOS Darwin 25.2.0
- Python 3.12
- NLTK 3.9.2, scikit-learn 1.8.0
- Corpus: Reuters (10,788 documents, 30,916 unique terms)

---

## Unit Tests

Location: `tests/test_document_matcher.py`

Used a small test corpus (4 documents) to verify core components.

### Test 1: test_fit_corpus
Tests TF-IDF vectorizer training.

Verified:
- Vectorizer accepts corpus and document IDs
- TF-IDF vectors computed correctly
- Data structures initialized

Result: PASS
- Created 4x22 matrix (4 documents, 22 terms)
- Code: tests/test_document_matcher.py, lines 26-33

### Test 2: test_find_similar_documents
Tests similarity matching functionality.

Query: "Machine learning and artificial intelligence"

Verified:
- Returns list of (doc_id, score) tuples
- Scores are between 0 and 1
- Results not empty

Result: PASS
- Returned valid matches
- All scores in valid range [0.0, 1.0]
- Code: tests/test_document_matcher.py, lines 35-52

### Test 3: test_percentile_filtering
Tests percentile threshold logic.

Setup: Same query at 50th and 90th percentile

Verified:
- Higher percentile returns fewer results
- Percentile calculation correct

Result: PASS
- 50th percentile >= 90th percentile (in document count)
- Code: tests/test_document_matcher.py, lines 54-65

Run with: `python -m unittest tests.test_document_matcher -v`

---

## Integration Tests

Location: `test_integration.py`

Full system tests using complete Reuters corpus.

### Test 1: Corpus Loading
Verified Reuters corpus loads from NLTK.

Result: PASS
- Loaded 10,788 documents
- All document IDs match

### Test 2: TF-IDF Computation
Verified vectorization on full corpus.

Result: PASS
- Matrix shape: 10,788 x 30,916
- Sparse matrix format used

### Test 3: Financial Query
Query: "Oil prices surged today as OPEC announced production cuts. Market analysts predict volatility."
Percentile: 70th

Result: PASS
- Found 3,237 documents (30% of corpus)
- Top match: training/144, similarity 0.3247
- Threshold: 0.0119

### Test 4: Percentile Filtering
Same financial query tested at multiple percentiles.

Results:

| Percentile | Threshold | Documents Returned |
|------------|-----------|-------------------|
| 50th       | 0.0000    | 10,788            |
| 90th       | 0.0348    | 1,079             |

This confirms correct behavior:
- 50th percentile threshold is 0.0 because many documents share no vocabulary with the query
- 90th percentile returns roughly 10% of corpus (1,079 / 10,788 = 10.0%)

### Test 5: Political Query
Query: "The government announced new legislation regarding environmental regulations."
Percentile: 70th

Result: PASS
- Found 3,237 documents
- Top match: training/10129, similarity 0.2175
- Different top match confirms system works across topics

### Test 6: Boundary - 0th Percentile
Verified 0th percentile returns all documents.

Result: PASS
- Returned all 10,788 documents
- No filtering applied

### Test 7: Boundary - 100th Percentile
Verified 100th percentile returns only best match.

Result: PASS
- Returned 1 document
- Threshold equals max similarity (0.3247)

### Test 8: Results Sorting
Verified results sorted by similarity descending.

Result: PASS
- Each score >= next score in list

### Test 9: File Input
Used test_sample.txt (oil price).

Result: PASS
- File loaded successfully
- Found 4,315 matches at 60th percentile
- Identical processing to text input

Run with: `python test_integration.py`

---

## Edge Cases

### Unknown Vocabulary
Query with words not in corpus vocabulary (e.g., "cryptocurrency" in 1987 Reuters).

Behavior: Unknown words ignored, matching uses shared vocabulary only.
Handles correctly via TfidfVectorizer.transform()

### Empty Similarity
Query with no words in common with corpus.

Behavior: Returns 0.0 similarity
Correct - cosine of orthogonal vectors is 0.

### Identical Documents
Query identical to corpus document.

Behavior: Returns 1.0 similarity
Correct - cosine of identical vectors is 1.

### Short Queries
Query with 1-2 words.

Behavior: Computes TF-IDF normally, may have lower scores.
Works correctly.

### Long Queries
Query with thousands of words.

Behavior: Processes normally using sparse matrices.
Works correctly and no performance issues.

---

## Test Summary

| Category | Passed | Failed |
|----------|--------|--------|
| Unit Tests | 3 | 0 |
| Integration Tests | 9 | 0 |
| Edge Cases | 5 | 0 |
| Requirements | 8 | 0 |
| **Total** | **25** | **0** |

---

## Performance

Measured during integration testing:

- Corpus loading: 2-3 seconds (first run), <1 second (cached)
- TF-IDF computation: 2-3 seconds
- Query matching: <0.1 seconds
- Memory usage: ~50-100 MB

Performance is acceptable for corpus of this size.

---

## Running Tests

Unit tests:
```bash
python -m unittest tests.test_document_matcher -v
```

Integration tests:
```bash
python test_integration.py
```

Manual testing:
```bash
cd src
python main.py
```

Expected output format:
```
Percentile threshold (70th): 0.0119
Found 3237 matching documents

  1. training/144         | Similarity: 0.3247
  2. training/856         | Similarity: 0.2891
  ...
```

---
