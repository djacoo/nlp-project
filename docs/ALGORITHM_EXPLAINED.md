# Algorithm Explanation

## How Document Similarity Matching Works

This document provides a explanation of the techniques and the pipeline used in this project. The goal is: given a query document provided by the user, find the most similar documents in the Reuters corpus.

The approach relies on two well-known methods from information retrieval:

1. **TF-IDF** (Term Frequency–Inverse Document Frequency) to convert raw text into numerical vectors
2. **Cosine similarity** to measure how close two document vectors are to each other

A percentile-based filter is then applied so that only the most relevant matches are returned.

---

## Part 1: TF-IDF — From Words to Numbers

### Why simple word counting falls short

A naive approach to representing documents numerically would be to count how many times each word appears. Consider two documents:

- Doc A: *"The cat sat on the mat"*
- Doc B: *"The dog sat on the rug"*

Under simple word counts, "the" would be flagged as the most significant word — it appears twice in each document. But "the" carries almost no information about what these documents are actually about. The meaningful words are "cat", "dog", "mat", and "rug".

TF-IDF addresses this directly.

### Definition

**TF-IDF** is the product of two quantities:

```
TF-IDF(term, document) = TF(term, document) × IDF(term)
```

where:

- **TF (Term Frequency)** measures how often a term appears within a single document:

  ```
  TF(term, doc) = (number of times term appears in doc) / (total number of terms in doc)
  ```

- **IDF (Inverse Document Frequency)** measures how rare or common a term is across the entire corpus:

  ```
  IDF(term) = log(N / df(term))
  ```

  where *N* is the total number of documents and *df(term)* is the number of documents that contain the term.

A term that appears frequently in one document (high TF) but rarely across the corpus (high IDF) is likely a strong indicator of that document's topic. Terms that appear in nearly every document (like "the", "is", "and") receive very low IDF values.

### Example for context

Consider a small corpus of three documents:

| | Document text |
|---|---|
| Doc 1 | *"cats are cute animals"* |
| Doc 2 | *"dogs are cute animals"* |
| Doc 3 | *"fish swim in water"* |

**TF-IDF of "cute" in Doc 1:**

1. TF = 1/4 = 0.25 (one occurrence out of four words)
2. IDF = log(3/2) = log(1.5) ≈ 0.176 (appears in 2 out of 3 documents)
3. TF-IDF = 0.25 × 0.176 = **0.044**

**TF-IDF of "fish" in Doc 3:**

1. TF = 1/4 = 0.25 (same within-document frequency)
2. IDF = log(3/1) = log(3) ≈ 0.477 (appears in only 1 document — much rarer)
3. TF-IDF = 0.25 × 0.477 = **0.119**

"fish" receives a higher TF-IDF score than "cute" despite having the same raw frequency, because it is more distinctive — it separates out Doc 3 from the rest of the corpus.

### Documents as vectors

Once TF-IDF scores are computed for every term in every document, each document can be represented as a vector in a high-dimensional space (one dimension per unique term in the vocabulary):

```
Vocabulary: [animals, are, cats, cute, dogs, fish, in, swim, water]

Doc 1 = [0.044, 0.044, 0.119, 0.044, 0.000, 0.000, 0.000, 0.000, 0.000]
Doc 2 = [0.044, 0.044, 0.000, 0.044, 0.119, 0.000, 0.000, 0.000, 0.000]
Doc 3 = [0.000, 0.000, 0.000, 0.000, 0.000, 0.119, 0.119, 0.119, 0.119]
```

*(Values are simplified; scikit-learn applies L2 normalization automatically)*

A few infos:

- Doc 1 and Doc 2 share several non-zero entries (animals, are, cute), indicating overlap in topics.
- Doc 3 has non-zero entries in entirely different dimensions, indicating it covers a different topic.
- Most entries are zero — this is typical for large vocabularies and is handled efficiently through sparse matrix storage.

---

## Part 2: Cosine Similarity — Measuring the Angle Between Documents

### Why we don't use Euclidean distance?

Euclidean distance is a common choice in this context, but it has a prlblem in this context: it is sensitive to vector magnitude. A long document and a short document about the same topic would appear distant simply because the longer one has larger raw values.

Cosine similarity avoids this problem by measuring the *angle* between two vectors rather than the distance between their endpoints.

### Definition

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

where:

- **A · B** is the dot product of the two vectors
- **||A||** and **||B||** are their respective magnitudes (L2 norms)

The result is always between 0 and 1 (for non-negative TF-IDF vectors):

| Value | Interpretation |
|-------|----------------|
| 1.0 | Identical documents (vectors point in the same direction) |
| 0.0 | No similarity (vectors are orthogonal — no shared terms) |
| 0.0–1.0 | Partial overlap in vocabulary |

### Geometric interpretation

```
     Doc 2
      ↗
     /
    / θ
   /
  /________→ Doc 1
```

Cosine similarity is cos(θ). When two document vectors point in almost the same direction (small θ), the cosine is close to 1. When they are perpendicular (θ = 90°), the cosine is 0.

### Example

Using same Doc 1 and Doc 2:

```
Doc 1 = [0.044, 0.044, 0.119, 0.044, 0.000, ...]
Doc 2 = [0.044, 0.044, 0.000, 0.044, 0.119, ...]
```

**Step 1 — Dot product:**
```
A · B = (0.044 × 0.044) + (0.044 × 0.044) + (0.119 × 0.000) + (0.044 × 0.044) + (0.000 × 0.119)
      = 0.001936 + 0.001936 + 0 + 0.001936 + 0
      = 0.005808
```

**Step 2 — Magnitudes:**
```
||Doc 1|| = sqrt(0.044² + 0.044² + 0.119² + 0.044²) = sqrt(0.019969) ≈ 0.1413
||Doc 2|| = sqrt(0.044² + 0.044² + 0.044² + 0.119²) = sqrt(0.019969) ≈ 0.1413
```

**Step 3 — Cosine similarity:**
```
similarity = 0.005808 / (0.1413 × 0.1413) = 0.005808 / 0.01997 ≈ 0.291
```

A score of 0.291 is moderate similarity: the two documents share vocabulary related to animals ("are", "cute", "animals") but differ in their subject ("cats" vs. "dogs").

---

## Part 3: The Full Pipeline

How these two techniques are combined into a complete document matching system.

### Inputs

- The **Reuters corpus** (10,788 news articles from 1987, accessed via NLTK)
- A **query document** provided by the user
- A **percentile threshold** between 0 and 100

### Step-by-step process

**Step 1 — Load the corpus**

```python
from nltk.corpus import reuters
corpus = [reuters.raw(doc_id) for doc_id in reuters.fileids()]
```

This pulls the raw text of all ~10,788 documents. No preprocessing (tokenization, stopword removal, stemming) is applied.

**Step 2 — Build TF-IDF vectors for the corpus**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
corpus_vectors = vectorizer.fit_transform(corpus)
```

`fit_transform` performs two operations: it learns the vocabulary and IDF values from the corpus (fit), then transforms every document into a TF-IDF vector (transform). The result is a matrix of shape (10,788 × ~30,916), where each row is a document and each column is a term in the vocabulary.

**Step 3 — Transform the query document**

```python
query_vector = vectorizer.transform([user_document])
```

The query is transformed using the *same* vocabulary and IDF values learned in Step 2. Any words in the query that do not appear in the corpus vocabulary are ignored.

**Step 4 — Compute cosine similarity**

```python
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(query_vector, corpus_vectors)[0]
```

This gives an array of 10,788 similarity scores, one for each document in the corpus.

**Step 5 — Apply the percentile filter**

```python
import numpy as np

threshold = np.percentile(similarities, percentile)
matching_indices = np.where(similarities >= threshold)[0]
```

`np.percentile(similarities, 70)`, for instance, finds the value below which 70% of all scores. Only documents with a similarity at or above that value.

**Step 6 — Sort and return**

```python
results = [(doc_ids[idx], similarities[idx]) for idx in matching_indices]
results.sort(key=lambda x: x[1], reverse=True)
```

The matched documents are sorted from highest to lowest similarity.

---

## Part 4: Complete Example

Consider a small corpus:

| | Document |
|---|---|
| Doc 1 | *"The stock market rose today"* |
| Doc 2 | *"Oil prices increased significantly"* |
| Doc 3 | *"The weather is nice today"* |

**Query:** *"Stock prices rose"*
**Percentile:** 50 (return the top half)

### Execution

**1. Vocabulary construction**

The vectorizer finds 12 unique terms:
```
{increased, is, market, nice, oil, prices, rose, significantly, stock, the, today, weather}
```

**2. TF-IDF vectors** (simplified)

```
            market  oil  prices  rose  stock  the  today  ...
Doc 1:      0.40    0    0       0.30  0.40   0.15 0.30   ...
Doc 2:      0       0.40 0.30    0     0      0    0      ...
Doc 3:      0       0    0       0     0      0.15 0.30   ...
Query:      0       0    0.40    0.35  0.50   0    0      ...
```

**3. Cosine similarity scores**

```
similarity(Query, Doc 1) ≈ 0.72   (shared terms: stock, rose)
similarity(Query, Doc 2) ≈ 0.41   (shared term: prices)
similarity(Query, Doc 3) ≈ 0.08   (minimal overlap)
```

**4. Percentile filtering at 50th percentile**

The sorted scores are [0.08, 0.41, 0.72]. The median (50th percentile) is 0.41. Documents with similarity >= 0.41:

**5. Results**

```
1. Doc 1  "The stock market rose today"       | Similarity: 0.72
2. Doc 2  "Oil prices increased significantly" | Similarity: 0.41
```

Doc 3 is excluded — its content (weather) has no topical connection to the query.

---
