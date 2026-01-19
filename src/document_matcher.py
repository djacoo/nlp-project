"""
Document Matcher - Core algorithm for Assignment Option C

Main core algorithm. It uses TF-IDF and cosine similarity
to find similar documents.

Quick recap of the techniques:
- TF-IDF: converts text to numbers, gives higher weight to rare/important words
- Cosine Similarity: measures angle between document vectors (0=different, 1=same)

The formulas (for reference):
    TF-IDF(term, doc) = TF(term, doc) × IDF(term)
    where TF = term frequency, IDF = inverse document frequency

    Cosine Similarity = (A · B) / (||A|| × ||B||)
    the dot product divided by magnitudes
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class DocumentMatcher:
    """
    Main class for document similarity matching

    Workflow:
    1. fit_corpus() - compute TF-IDF vectors for the whole corpus
    2. find_similar_documents() - find docs similar to a query
    3. print_results() - display the results

    Storing the vectorizer and corpus_vectors so don't have to recompute TF-IDF every time.
    """

    def __init__(self):
        """
        Initialize with empty values

        Everything set when fit_corpus() is called.
        Following sklearn's fit/transform pattern.
        """
        self.vectorizer = None      # TfidfVectorizer object
        self.corpus_vectors = None  # TF-IDF matrix (stored as sparse matrix for efficiency)
        self.corpus = None          # original texts
        self.doc_ids = None         # document identifiers

    def fit_corpus(self, corpus: List[str], doc_ids: List[str]):
        """
        Compute TF-IDF vectors for the entire corpus

        This is the training phase:
        1. Build vocabulary from all documents
        2. Calculate IDF for each word (how rare/common it is)
        3. Transform each document into a TF-IDF vector

        Example: if we have docs ["cats are nice", "dogs are nice", "fish swim"],
        the vectorizer builds vocabulary {cats, are, nice, dogs, fish, swim} and
        calculates how important each word is across all documents.

        Args:
            corpus: list of document texts (raw, no preprocessing)
            doc_ids: corresponding document IDs

        Note: no stopword elimination!!
        """
        self.corpus = corpus
        self.doc_ids = doc_ids

        # initialize vectorizer with default settings
        # (no stopwords removal, standard tokenization)
        self.vectorizer = TfidfVectorizer()

        # fit_transform does two steps:
        # 1) fit - learn vocabulary and IDF values
        # 2) transform - convert docs to TF-IDF vectors
        print("Computing TF-IDF vectors for corpus...")
        self.corpus_vectors = self.vectorizer.fit_transform(corpus)

        # show results
        print(f"TF-IDF matrix shape: {self.corpus_vectors.shape}")
        print(f"  - {self.corpus_vectors.shape[0]} documents")
        print(f"  - {self.corpus_vectors.shape[1]} unique terms in vocabulary")

    def find_similar_documents(
        self,
        query_document: str,
        percentile: float
    ) -> List[Tuple[str, float]]:
        """
        Find documents similar to query above a percentile threshold

        This implements the main algorithm:
        1. Convert query to TF-IDF vector (using same vocabulary as corpus)
        2. Calculate cosine similarity between query and all corpus docs
        3. Use percentile to find threshold value
        4. Keep only docs above threshold
        5. Sort results by similarity

        Args:
            query_document: the user's input text
            percentile: threshold value (0-100)
                0 = return all docs
                50 = return top 50%
                90 = return top 10%

        Returns:
            list of (doc_id, similarity_score) tuples, sorted by score

        """
        # fitted corpus check
        if self.vectorizer is None or self.corpus_vectors is None:
            raise ValueError("Must call fit_corpus() before finding similar documents")

        # transform query using the same vocabulary
        # if query has new words, they just get ignored
        query_vector = self.vectorizer.transform([query_document])

        # calculate cosine similarity with all corpus documents
        # returns 2D array (only 1 query, so [0])
        similarities = cosine_similarity(query_vector, self.corpus_vectors)[0]

        # find the threshold value based on percentile
        # numpy's percentile finds the value where X% of data falls below it
        threshold = np.percentile(similarities, percentile)

        print(f"\nPercentile threshold ({percentile}th): {threshold:.4f}")

        # get indices of documents above threshold
        matching_indices = np.where(similarities >= threshold)[0]

        # build results with doc IDs and scores
        results = [
            (self.doc_ids[idx], similarities[idx])
            for idx in matching_indices
        ]

        # sort by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def print_results(self, results: List[Tuple[str, float]], percentile: float):
        """
        Print results

        """
        print(f"\n{'='*70}")
        print(f"Documents matching above {percentile}th percentile")
        print(f"Found {len(results)} matching documents")
        print(f"{'='*70}\n")

        # print each result
        for i, (doc_id, score) in enumerate(results, 1):
            print(f"{i:3d}. {doc_id:20s} | Similarity: {score:.4f}")
