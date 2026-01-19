# Project Plan
## NLP Document Similarity Matcher - Assignment Option C

---

## Assignment Choice

I chose **Assignment Option C** from the general assignments because it seemed the most straightforward to implement while still being technically interesting. The requirement is:

> Asks for a document to the user, and a match percentile, and returns the documents of the corpus that match the document above the percentile as similar by means of the cosine similarity of the tf-idf (the match has to be on complete docs, no application of the stopword elimination phase).

---

## What I Needed to Implement

### Core Requirements
1. Accept a document from the user
2. Accept a percentile value (0-100)
3. Use the Reuters corpus from NLTK
4. Calculate TF-IDF vectors
5. Use cosine similarity for matching
6. Return documents above the percentile threshold
7. **No stopword removal** (important!)

### Technical Approach
- **Language**: Python (easier than Java for NLP stuff)
- **Libraries**: NLTK for corpus, sklearn for TF-IDF and cosine similarity
- **Corpus**: Reuters (about 10,788 news documents)
- **Development**: Using gitflow workflow as required

---

## Project Structure

I organized the code into three main modules to keep things clean:

```
src/
├── corpus_loader.py      # Handles loading Reuters from NLTK
├── document_matcher.py   # The main algorithm (TF-IDF + cosine similarity)
└── main.py              # User interface and program flow
```

Plus tests, documentation, and project files.

---

## Implementation Phases

### Phase 1: Initial Setup
- Set up project structure
- Create git repo with gitflow (main, develop, feature branches)
- Set up requirements.txt with dependencies
- Basic README

### Phase 2: Core Implementation
- Implemented corpus_loader.py to load Reuters
- Implemented document_matcher.py with:
  - TF-IDF vectorization using sklearn
  - Cosine similarity calculation
  - Percentile filtering
- Implemented main.py with user interface
- Basic unit tests

### Phase 3: Documentation
- Added detailed comments to all code files
- Wrote ALGORITHM_EXPLAINED.md with step-by-step explanation
- Updated README with theory and usage instructions
- Added this project plan

---

## Key Implementation Details

### TF-IDF Vectorization

Used sklearn's `TfidfVectorizer` with default settings (no stopword removal):

```python
vectorizer = TfidfVectorizer()
corpus_vectors = vectorizer.fit_transform(corpus)
```

This creates a matrix where:
- Rows = documents (~10,788)
- Columns = vocabulary terms (~35,000)
- Values = TF-IDF scores

### Cosine Similarity

Used sklearn's `cosine_similarity`:

```python
similarities = cosine_similarity(query_vector, corpus_vectors)[0]
```

Returns an array of similarity scores (0 to 1) for each corpus document.

### Percentile Filtering

Used numpy's `percentile` function:

```python
threshold = np.percentile(similarities, percentile)
matching_docs = documents where similarity >= threshold
```

This was initially confusing - if percentile=70, we return the top 30% of documents (those above the 70th percentile value).

---

## Challenges and Solutions

### Challenge 1: Understanding Percentiles

**Problem**: At first I thought percentile=70 meant "return documents with >70% similarity", but that's not what it means.

**Solution**: Percentile=70 means "find the value where 70% of data falls below it, and return everything above that value". So you're getting the top 30% of documents.

### Challenge 2: Vocabulary Consistency

**Problem**: Need to ensure query document uses the same vocabulary as corpus.

**Solution**: Use `fit_transform()` on corpus to learn vocabulary, then use `transform()` on query with the same vectorizer instance.

### Challenge 3: No Stopword Removal

**Problem**: Normally you'd remove stopwords, but assignment says not to.

**Solution**: Just use default TfidfVectorizer settings. Made sure to document this clearly in the code.

### Challenge 4: Memory Efficiency

**Problem**: 10,788 × 35,000 matrix would use a lot of RAM.

**Solution**: Sklearn uses sparse matrices by default, which only store non-zero values. Reduces memory from ~3GB to ~50MB.

---

## Testing Strategy

Created basic unit tests to verify:
- Corpus loading works
- TF-IDF vectorization works
- Similarity calculation works
- Percentile filtering works correctly

Not extensive testing, but enough to catch obvious bugs.

---

## Git Workflow

Following gitflow as required:

1. **main** branch: stable code
2. **develop** branch: development work
3. **feature/** branches: individual features

Workflow for each feature:
```bash
git checkout develop
git checkout -b feature/feature-name
# implement feature
git commit -m "Description"
git checkout develop
git merge feature/feature-name
```

---

## What I Learned

### Technical Lessons

1. **TF-IDF is elegant**: Simple idea (multiply term frequency by inverse document frequency) that works surprisingly well.

2. **Cosine similarity makes sense**: By measuring angle instead of distance, it handles document length nicely.

3. **Sparse matrices are important**: Can't do NLP at scale without efficient matrix storage.

4. **Sklearn is powerful**: Makes complex NLP operations simple (2-3 lines of code for TF-IDF).

### Assignment Compliance Lessons

1. **Read requirements carefully**: The "no stopword elimination" requirement is easy to miss but important.

2. **Percentile vs threshold**: These are different things and it's important to implement what's actually asked for.

3. **Complete documents**: Assignment specifically says to match on complete docs, so no preprocessing.

---

## Assignment Requirements Checklist

Making sure I've covered everything from the assignment:

- [x] Accepts document from user (via direct input or file)
- [x] Accepts percentile parameter (0-100, validated)
- [x] Uses Reuters corpus from NLTK (nltk.corpus.reuters)
- [x] Implements TF-IDF (sklearn.feature_extraction.text.TfidfVectorizer)
- [x] Calculates cosine similarity (sklearn.metrics.pairwise.cosine_similarity)
- [x] Returns documents above percentile threshold
- [x] No stopword elimination (using default vectorizer)
- [x] Matches on complete documents (no preprocessing)

---

## Future Improvements (Not Required)

If I were to extend this (not part of the assignment):

1. **Add stopword removal option**: Make it configurable
2. **Stemming/lemmatization**: Normalize words better
3. **Better UI**: Maybe a web interface instead of command line
4. **Caching**: Save TF-IDF vectors so you don't recompute every time
5. **Parallel processing**: Speed up similarity calculations
6. **Visualization**: Show similar documents in a 2D space using dimensionality reduction

But for the assignment, the current implementation is sufficient.

---

## Time Spent

Rough breakdown:

- **Setup and structure**: 1 hour
- **Core implementation**: 3-4 hours
- **Debugging and testing**: 2 hours
- **Documentation**: 3-4 hours
- **Total**: ~10-12 hours

Most time was spent on making sure I understood TF-IDF and cosine similarity properly, and on writing clear documentation.

---

## References Used

- Assignment PDF
- Manning, Raghavan & Schütze - *Introduction to Information Retrieval*
- Sklearn documentation (TfidfVectorizer, cosine_similarity)
- NLTK documentation (Reuters corpus)
- StackOverflow for specific implementation questions

---

## Conclusion

The project successfully implements Assignment Option C requirements. The code is clean, well-documented, and follows the specified constraints (no stopword removal, complete documents). The algorithm works as expected and the results make sense (documents about similar topics get high similarity scores).

Ready for submission and presentation.
