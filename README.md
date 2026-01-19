# NLP Document Similarity Matcher

## Assignment Overview
This project implements **Option C** from the NLP course general assignment: a document similarity matcher using TF-IDF and cosine similarity.

### What it does
- Accepts a document and a match percentile from the user
- Computes TF-IDF vectors for the document and the Reuters corpus
- Calculates cosine similarity between the input document and all corpus documents
- Returns documents with similarity above the specified percentile threshold
- No stopword elimination (as per assignment requirements)

## Project Structure
```
nlp-project/
├── src/                    # Source code
│   ├── __init__.py
│   ├── main.py            # Entry point
│   ├── corpus_loader.py   # Reuters corpus loader
│   └── document_matcher.py # TF-IDF and similarity matching
├── tests/                  # Unit tests
│   ├── __init__.py
│   └── test_document_matcher.py
├── data/                   # Data files (optional)
├── docs/                   # Documentation
├── assignment/            # Assignment PDF
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd nlp-project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (Reuters corpus):
The corpus will be downloaded automatically on first run, or you can download it manually:
```python
import nltk
nltk.download('reuters')
```

## Usage

### Running the application
```bash
cd src
python main.py
```

### Input options
The program will prompt you to:
1. Choose input method:
   - Enter text directly
   - Provide a file path to a document
2. Enter a match percentile (0-100)

### Example
```
Document Similarity Matcher - NLP Assignment Option C
======================================================================

How would you like to provide the document?
1. Enter text directly
2. Provide a file path

Enter your choice (1 or 2): 1

Enter your document text (press Ctrl+D or Ctrl+Z when done):
This is a test document about economic trends and market analysis.
^D

Enter match percentile (0-100): 70

Searching for similar documents...

======================================================================
Documents matching above 70th percentile
Found 15 matching documents
======================================================================

  1. test/14826            | Similarity: 0.8542
  2. test/14828            | Similarity: 0.7893
  ...
```

## Running Tests
```bash
python -m unittest discover tests
```

Or run specific test file:
```bash
python -m unittest tests.test_document_matcher
```

## Technical Details

### Libraries Used
- **NLTK**: Reuters corpus access
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **NumPy**: Numerical operations
- **Pandas**: Data handling (optional)

### Algorithm
1. Load Reuters corpus using NLTK
2. Create TF-IDF vectors for all corpus documents using `TfidfVectorizer`
3. Transform user's input document using the same vectorizer
4. Compute cosine similarity between input and all corpus documents
5. Calculate percentile threshold
6. Return documents with similarity >= threshold

### Key Features
- No stopword elimination (as per assignment specification)
- Complete document matching
- Percentile-based filtering
- Sorted results by similarity score

## Development

### Git Workflow (Gitflow)
This project follows the gitflow workflow:

- `main`: Production-ready code
- `develop`: Development branch
- `feature/*`: Feature branches
- `hotfix/*`: Hotfix branches
- `release/*`: Release branches

### Creating a new feature
```bash
git checkout develop
git checkout -b feature/feature-name
# Make changes
git add .
git commit -m "Description of changes"
git checkout develop
git merge feature/feature-name
```

