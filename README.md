# NLP Document Similarity Matcher

This is my implementation of Assignment Option C for the NLP course - a document similarity matching system using TF-IDF and cosine similarity.

## What This Does

The program finds documents in the Reuters corpus that are similar to a document you provide. It uses two main NLP techniques:

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**: Converts documents into numerical vectors where important words get higher weights
2. **Cosine Similarity**: Measures how similar two documents are by calculating the angle between their vector representations

You give it a document and a percentile threshold, and it returns all corpus documents above that similarity threshold.

## The Theory Behind It

### Why TF-IDF?

Simple word counting doesn't work well for document similarity because common words like "the" or "is" appear everywhere but don't tell us much about what a document is actually about. TF-IDF solves this by:

- **TF (Term Frequency)**: Measuring how often a word appears in a document
- **IDF (Inverse Document Frequency)**: Measuring how rare/unique a word is across all documents
- **TF-IDF = TF × IDF**: Combining both to highlight important, distinctive words

Example: The word "the" appears in almost every document → low IDF → low TF-IDF
The word "cryptocurrency" appears in few documents → high IDF → high TF-IDF

So documents about similar topics will have high TF-IDF values for the same distinctive words.

### Why Cosine Similarity?

Once we have documents as TF-IDF vectors, we need to measure how similar they are. Cosine similarity does this by measuring the angle between two vectors:

```
similarity = cos(angle between vectors)
```

- Similarity = 1 means identical documents (0° angle)
- Similarity = 0 means completely different (90° angle)

The big advantage: it's **independent of document length**. A short article and a long article about the same topic will still show high similarity.

### The Percentile Threshold

The percentile parameter lets you control how selective the results are:

- **Percentile 90**: Only return top 10% most similar docs (very selective)
- **Percentile 70**: Return top 30% (moderate)
- **Percentile 50**: Return top 50% (inclusive)

## Project Structure

```
nlp-project/
├── src/                         # Main source code
│   ├── corpus_loader.py         # Loads Reuters corpus from NLTK
│   ├── document_matcher.py      # TF-IDF and cosine similarity implementation
│   └── main.py                  # User interface and program flow
├── tests/                       # Unit tests
│   └── test_document_matcher.py
├── docs/                        # Documentation
│   ├── PROJECT_PLAN.md         # Development plan
│   └── ALGORITHM_EXPLAINED.md  # Detailed algorithm walkthrough
├── data/                        # For example documents
├── assignment/                  # Assignment PDF
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation & Setup

### Requirements

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd nlp-project
```

2. **Create a virtual environment** (recommended to avoid dependency conflicts)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

This installs:
- nltk (for Reuters corpus)
- scikit-learn (for TF-IDF and cosine similarity)
- numpy (for numerical operations)
- pandas (for data handling)

4. **Download Reuters corpus** (optional - it downloads automatically on first run)
```python
import nltk
nltk.download('reuters')
```

The Reuters corpus is about 2MB and contains 10,788 news articles from 1987.

## How to Run

### Basic Usage

```bash
cd src
python main.py
```

The program will walk you through the input process:

1. **Choose how to provide your document**:
   - Option 1: Type it directly (good for testing with short text)
   - Option 2: Provide a file path (better for real documents)

2. **Enter a percentile threshold** (0-100):
   - Higher = fewer, more similar results
   - Lower = more, less strict results

3. **View the results**: The program shows matching documents sorted by similarity score

4. **Optional**: Save results to a file

### Example Run

```
======================================================================
INITIALIZING DOCUMENT SIMILARITY MATCHER
======================================================================

[1/4] Loading Reuters corpus...
Loaded 10788 documents from Reuters corpus

[2/4] Computing TF-IDF vectors...
TF-IDF matrix shape: (10788, 35423)
  - 10788 documents
  - 35423 unique terms in vocabulary
✓ TF-IDF computation complete

[3/4] Getting user input...

======================================================================
Document Similarity Matcher - NLP Assignment Option C
======================================================================

How would you like to provide the document?
1. Enter text directly (good for short documents)
2. Provide a file path (good for longer documents)

Enter your choice (1 or 2): 1

Enter your document text (press Ctrl+D on Mac/Linux or Ctrl+Z on Windows when done):
The stock market showed strong gains today as investors responded positively
to economic data showing growth in manufacturing and employment.
^D

Enter match percentile (0-100): 80

✓ Document received (156 characters)
✓ Percentile threshold: 80.0

[4/4] Finding similar documents...

Percentile threshold (80th): 0.1234
✓ Search complete

======================================================================
Documents matching above 80.0th percentile
Found 2158 matching documents
======================================================================

  1. training/5674       | Similarity: 0.7821
  2. test/14826          | Similarity: 0.7654
  3. training/8923       | Similarity: 0.7432
  ...
```

## Running Tests

I've included some basic unit tests to make sure the core functionality works:

```bash
python -m unittest discover tests
```

Or run a specific test file:
```bash
python -m unittest tests.test_document_matcher
```

## How It Works (Algorithm Overview)

Here's what happens when you run the program:

### Step 1: Load and Prepare Corpus
```python
# Load all Reuters documents
corpus, doc_ids = loader.load_corpus()

# Compute TF-IDF vectors for entire corpus
matcher.fit_corpus(corpus, doc_ids)
```

This creates a matrix where:
- Rows = documents (10,788 docs)
- Columns = unique words (vocabulary ~35,000 words)
- Values = TF-IDF scores

### Step 2: Process Query Document
```python
# Transform user's document using same vocabulary
query_vector = vectorizer.transform([user_document])
```

Important: The query uses the **same vocabulary** learned from the corpus. New words are ignored.

### Step 3: Calculate Similarities
```python
# Compute cosine similarity between query and all corpus docs
similarities = cosine_similarity(query_vector, corpus_vectors)
```

This gives us a similarity score (0 to 1) for each corpus document.

### Step 4: Filter by Percentile
```python
# Find the threshold value for the percentile
threshold = np.percentile(similarities, percentile)

# Keep only documents above threshold
results = documents where similarity >= threshold
```

### Step 5: Sort and Display
```python
# Sort by similarity (highest first)
results.sort(reverse=True)
```

## Assignment Requirements Compliance

This implementation fulfills all requirements for Assignment Option C:

✅ **Accepts document from user** - Via direct input or file path
✅ **Accepts percentile parameter** - Value from 0-100
✅ **Uses Reuters corpus** - Loaded via NLTK
✅ **Implements TF-IDF** - Using sklearn's TfidfVectorizer
✅ **Calculates cosine similarity** - Using sklearn's cosine_similarity
✅ **Returns documents above percentile** - Filtered and sorted results
✅ **No stopword elimination** - Complete documents as specified

## Technical Notes

### Why No Stopword Removal?

The assignment specifically states "the match has to be on complete docs, no application of the stopword elimination phase." This is why I'm using:

```python
vectorizer = TfidfVectorizer()  # No stopwords parameter
```

In a real-world application, you'd typically remove stopwords ("the", "is", "and", etc.) to improve results.

### Performance

- **Loading corpus**: ~2 seconds
- **Computing TF-IDF**: ~3-5 seconds (one-time cost)
- **Finding similar docs**: <1 second per query

The TF-IDF computation only happens once. After that, you can run multiple queries quickly.

### Limitations

1. **Bag of words approach**: Word order doesn't matter ("dog bites man" = "man bites dog")
2. **No semantic understanding**: "car" and "automobile" are treated as completely different words
3. **Vocabulary limited to corpus**: Query words not in corpus vocabulary are ignored
4. **No handling of synonyms or context**: Purely statistical approach

For more advanced similarity, you'd use techniques like word embeddings or transformers.

## Development Workflow

This project follows gitflow:

- **main**: Stable, production-ready code
- **develop**: Development branch
- **feature/**: Feature branches (e.g., feature/academic-enhancement)

To add a new feature:
```bash
git checkout develop
git checkout -b feature/feature-name
# Make your changes
git add .
git commit -m "Description of changes"
git checkout develop
git merge feature/feature-name
```

## Files Explained

### Source Code

- **corpus_loader.py**: Handles loading the Reuters corpus from NLTK. Pretty straightforward - just downloads if needed and loads all the documents.

- **document_matcher.py**: This is where the main algorithm lives. Implements TF-IDF vectorization and cosine similarity matching.

- **main.py**: The user interface. Coordinates everything and handles input/output.

### Documentation

- **ALGORITHM_EXPLAINED.md**: Detailed step-by-step explanation of how TF-IDF and cosine similarity work, with examples.

- **PROJECT_PLAN.md**: Development plan and project overview.

## Troubleshooting

**Problem**: "ModuleNotFoundError: No module named 'nltk'"
**Solution**: Make sure you activated the virtual environment and ran `pip install -r requirements.txt`

**Problem**: "Resource reuters not found"
**Solution**: The program should download it automatically, but if not, run:
```python
import nltk
nltk.download('reuters')
```

**Problem**: "ValueError: Must call fit_corpus() before finding similar documents"
**Solution**: This shouldn't happen in normal usage, but it means you tried to search before loading the corpus.

## References

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
- Scikit-learn TF-IDF Documentation: https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
- NLTK Reuters Corpus: https://www.nltk.org/book/ch02.html

## Author

MSc Artificial Intelligence Student
NLP Course Assignment - Option C
