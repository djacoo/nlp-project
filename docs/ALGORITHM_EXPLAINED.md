# Algorithm Explanation
## How Document Similarity Matching Works

## The Explanation

**What we're trying to do**: Given a document from the user, find similar documents in the Reuters corpus.

**How we do it**:
1. Convert all documents (corpus + query) into numbers using TF-IDF
2. Measure similarity between the query and each corpus document using cosine similarity
3. Filter by percentile to only show the most similar ones

**Why this approach**:
- TF-IDF is a classic way to represent text numerically while capturing which words are important
- Cosine similarity works well for comparing documents regardless of their length

---

## Part 1: Understanding TF-IDF

### The Problem with Simple Word Counting

If you just count words, you run into issues. Consider these two documents:

- Doc A: "The cat sat on the mat"
- Doc B: "The dog sat on the rug"

A simple word count would say "the" is the most important word (appears twice in each). But "the" doesn't tell us anything about what the documents are about. The meaningful words are "cat", "dog", "mat", "rug".

TF-IDF solves this.

### What is TF-IDF?

**TF-IDF** = **Term Frequency** × **Inverse Document Frequency**

It's basically: *how often does this word appear in this document* × *how rare is this word overall*

**The Formula:**
```
TF-IDF(word, document) = TF(word, doc) × IDF(word)

where:
  TF(word, doc) = (times word appears in doc) / (total words in doc)
  IDF(word) = log(total documents / documents containing word)
```

### Breaking it Down with an Example

Let's use a simple corpus to make a simple example:

**Corpus:**
- Doc1: "cats are cute animals"
- Doc2: "dogs are cute animals"
- Doc3: "fish swim in water"

**Calculate TF-IDF for "cute" in Doc1:**

**Step 1: Calculate TF (Term Frequency)**
- "cute" appears 1 time in Doc1
- Doc1 has 4 words
- TF = 1/4 = 0.25

**Step 2: Calculate IDF (Inverse Document Frequency)**
- Total documents = 3
- Documents containing "cute" = 2 (Doc1 and Doc2)
- IDF = log(3/2) = log(1.5) ≈ 0.176

**Step 3: Calculate TF-IDF**
- TF-IDF = 0.25 × 0.176 = **0.044**

**Now compare with "fish" in Doc3:**
- TF("fish", Doc3) = 1/4 = 0.25 (same frequency as "cute")
- IDF("fish") = log(3/1) = 0.477 (appears in only 1 doc, so higher IDF!)
- TF-IDF("fish", Doc3) = 0.25 × 0.477 = **0.119**

**Key insight**: "fish" has a higher TF-IDF than "cute" because it's more unique/distinctive. Words that appear in many documents (like "the", "is", "are") get low IDF scores and thus low TF-IDF scores, even if they appear frequently.

### Documents as Vectors

After calculating TF-IDF for every word in every document, each document becomes a vector:

```
Vocabulary: [animals, are, cats, cute, dogs, fish, in, swim, water]

Doc1 = [0.044, 0.044, 0.119, 0.044, 0.000, 0.000, 0.000, 0.000, 0.000]
Doc2 = [0.044, 0.044, 0.000, 0.044, 0.119, 0.000, 0.000, 0.000, 0.000]
Doc3 = [0.000, 0.000, 0.000, 0.000, 0.000, 0.119, 0.119, 0.119, 0.119]
```

(These are simplified values - in reality sklearn does some additional normalization)

Notice:
- Doc1 and Doc2 share several non-zero values (animals, are, cute)
- Doc3 has completely different non-zero values
- This suggests Doc1 and Doc2 are more similar to each other than to Doc3

---

## Part 2: Understanding Cosine Similarity

### Why We Need It

Now that we have documents as vectors, we need a way to measure how similar two vectors are. We could use Euclidean distance, but that has a problem: longer documents would always seem more "different" just because they have more words.

Cosine similarity solves this by measuring the *angle* between vectors, not their magnitude.

### The Formula

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

where:
  A · B = dot product of A and B
  ||A|| = magnitude (length) of vector A
  ||B|| = magnitude (length) of vector B
```

### What This Means

Think of it geometrically:

```
     Doc2
      ↗
     /
    /  θ (angle)
   /
  /_________→ Doc1
```

Cosine similarity = cos(θ)

- If angle is 0° (vectors point same direction) → cos(0°) = 1 → documents are identical
- If angle is 90° (vectors perpendicular) → cos(90°) = 0 → documents are completely different
- If angle is 45° (somewhat similar) → cos(45°) ≈ 0.71 → documents are moderately similar

### Worked Example

Let's calculate similarity between Doc1 and Doc2 from earlier:

```
Doc1 = [0.044, 0.044, 0.119, 0.044, 0.000, 0.000, 0.000, 0.000, 0.000]
Doc2 = [0.044, 0.044, 0.000, 0.044, 0.119, 0.000, 0.000, 0.000, 0.000]
```

**Step 1: Dot Product (A · B)**
```
A · B = (0.044 × 0.044) + (0.044 × 0.044) + (0.119 × 0.000) + (0.044 × 0.044) + ...
      = 0.001936 + 0.001936 + 0 + 0.001936 + 0 + ...
      = 0.005808
```

**Step 2: Magnitude of Doc1**
```
||Doc1|| = sqrt(0.044² + 0.044² + 0.119² + 0.044² + 0² + ...)
         = sqrt(0.001936 + 0.001936 + 0.014161 + 0.001936)
         = sqrt(0.019969)
         = 0.1413
```

**Step 3: Magnitude of Doc2**
```
||Doc2|| = sqrt(0.044² + 0.044² + 0² + 0.044² + 0.119² + ...)
         = sqrt(0.019969)  (turns out to be same as Doc1)
         = 0.1413
```

**Step 4: Cosine Similarity**
```
similarity = 0.005808 / (0.1413 × 0.1413)
           = 0.005808 / 0.01997
           = 0.291
```

**Interpretation**: Similarity of 0.291 means Doc1 and Doc2 are somewhat similar (they share "are cute animals") but also different (one has "cats", the other has "dogs").

---

## Part 3: The Complete Algorithm

Here's how it works in my implementation:

### Input
- Reuters corpus (10,788 documents)
- User's query document
- Percentile threshold (0-100)

### Step-by-Step Process

**STEP 1: Load Corpus**
```python
from nltk.corpus import reuters
corpus = [reuters.raw(doc_id) for doc_id in reuters.fileids()]
```

We get about 10,788 news articles.

**STEP 2: Build TF-IDF Vectors for Entire Corpus**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
corpus_vectors = vectorizer.fit_transform(corpus)
```

This creates a matrix:
- Rows = documents (10,788)
- Columns = unique words in vocabulary (~35,000)
- Values = TF-IDF scores

The matrix is *sparse* (mostly zeros) because any given document only contains a small fraction of the total vocabulary.

**STEP 3: Transform Query Document**
```python
query_vector = vectorizer.transform([user_document])
```

Important: We use the SAME vectorizer (same vocabulary) that was fit on the corpus. If the query contains words not in the corpus vocabulary, they get ignored.

**STEP 4: Calculate Cosine Similarity**
```python
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(query_vector, corpus_vectors)[0]
```

This returns an array of 10,788 similarity scores, one for each corpus document.

**STEP 5: Apply Percentile Threshold**
```python
import numpy as np

threshold = np.percentile(similarities, percentile)
matching_indices = np.where(similarities >= threshold)[0]
```

Example: if percentile=70, this finds the similarity value where 70% of documents fall below it. Then we keep only documents at or above that threshold.

**STEP 6: Sort and Return Results**
```python
results = [(doc_ids[idx], similarities[idx]) for idx in matching_indices]
results.sort(key=lambda x: x[1], reverse=True)
```

Sort by similarity score, highest first.

---

## Part 4: Full Worked Example

Here's a complete example with a small corpus to show every step.

### Setup

**Corpus:**
1. Doc1: "The stock market rose today"
2. Doc2: "Oil prices increased significantly"
3. Doc3: "The weather is nice today"

**Query:** "Stock prices rose"

**Percentile:** 50 (return top 50%)

### Execution

**Step 1: Build Vocabulary**
```
{increased, is, market, nice, oil, prices, rose, significantly, stock, the, today, weather}
```
12 unique words total.

**Step 2: Create TF-IDF Vectors** (simplified values)

Here the non-zero values (for clarity):

```
               market  nice  oil  prices  rose  stock  the  today  weather  increased  is  sig
Doc1:          0.40    0     0    0       0.30  0.40   0.15 0.30   0        0         0   0
Doc2:          0       0     0.40 0.30    0     0      0    0      0        0.45      0   0.45
Doc3:          0       0.45  0    0       0     0      0.15 0.30   0.40     0         0.40 0
Query:         0       0     0    0.40    0.35  0.50   0    0      0        0         0   0
```

(In reality these would be normalized)

**Step 3: Calculate Cosine Similarities**

Manually calculating (tho sklearn does this):

```
similarity(Query, Doc1):
  - Both have: stock, rose
  - Dot product relatively high
  - Similarity ≈ 0.72

similarity(Query, Doc2):
  - Both have: prices
  - Dot product lower
  - Similarity ≈ 0.41

similarity(Query, Doc3):
  - No meaningful overlap
  - Similarity ≈ 0.08
```

**Step 4: Apply 50th Percentile**
```
Sorted similarities: [0.08, 0.41, 0.72]
50th percentile value = 0.41 (median)
Keep documents where similarity >= 0.41
```

**Step 5: Results**
```
1. Doc1 "stock market rose today"       | Similarity: 0.72
2. Doc2 "oil prices increased"          | Similarity: 0.41
```

**Why these results make sense:**
- Doc1 is most similar: shares "stock" and "rose" with query
- Doc2 is moderately similar: shares "prices" with query
- Doc3 is excluded: completely different topic (weather vs economics)

---

## Part 5: Why This Works (And When It Doesn't)

### Why It Works

**1. Captures Topic Similarity**
Documents about the same topic use similar vocabulary. A document about stock markets will have words like "stock", "market", "trading", "shares". TF-IDF captures this.

**2. Downweights Common Words**
Words like "the", "is", "and" appear everywhere. TF-IDF gives them low scores so they don't dominate the similarity calculation.

**3. Length-Independent**
Cosine similarity normalizes for document length. A 100-word article and a 1000-word article about the same topic can still have high similarity.

**4. Computationally Efficient**
Matrix operations are fast. Even with 10,000+ documents, we can compute all similarities in under a second.

### Limitations

**1. Bag of Words**
Word order doesn't matter. These are treated the same:
- "dog bites man"
- "man bites dog"

**2. No Semantic Understanding**
These are treated as completely different:
- "car"
- "automobile"

They don't share any letters so TF-IDF sees them as unrelated.

**3. Vocabulary Boundary**
If your query contains words not in the corpus, they're ignored. This can be a problem with proper nouns, new terminology, or typos.

**4. No Context**
The word "bank" could mean a financial bank or (for example) a river bank. TF-IDF treats all occurrences the same.

### When to Use This vs Alternatives

**Use TF-IDF + Cosine Similarity when:**
- You have a fixed corpus to search
- Documents use consistent terminology
- Speed is important
- You need explainable results

**Use alternatives (word embeddings, transformers) when:**
- You need semantic understanding (synonyms, context)
- You're working across languages
- You have GPU resources and time for more complex models
- Documents use varied terminology for the same concepts

---