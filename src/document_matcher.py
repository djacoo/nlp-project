"""
Document Matcher Module
Implements document similarity matching using TF-IDF and cosine similarity
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple


class DocumentMatcher:
    """
    Matches documents based on cosine similarity of TF-IDF vectors
    Assignment Option C implementation
    """

    def __init__(self):
        """Initialize the document matcher"""
        self.vectorizer = None
        self.corpus_vectors = None
        self.corpus = None
        self.doc_ids = None

    def fit_corpus(self, corpus: List[str], doc_ids: List[str]):
        """
        Fit the TF-IDF vectorizer on the corpus

        Args:
            corpus: List of document texts
            doc_ids: List of document identifiers
        """
        self.corpus = corpus
        self.doc_ids = doc_ids

        # Initialize TF-IDF vectorizer (no stopword removal as per assignment)
        self.vectorizer = TfidfVectorizer()

        # Fit and transform the corpus
        print("Computing TF-IDF vectors for corpus...")
        self.corpus_vectors = self.vectorizer.fit_transform(corpus)
        print(f"TF-IDF matrix shape: {self.corpus_vectors.shape}")

    def find_similar_documents(
        self,
        query_document: str,
        percentile: float
    ) -> List[Tuple[str, float]]:
        """
        Find documents similar to the query document above the percentile threshold

        Args:
            query_document: The input document text
            percentile: Match percentile (0-100)

        Returns:
            List of tuples (doc_id, similarity_score) for matching documents
        """
        if self.vectorizer is None or self.corpus_vectors is None:
            raise ValueError("Must call fit_corpus() before finding similar documents")

        # Transform the query document
        query_vector = self.vectorizer.transform([query_document])

        # Compute cosine similarity with all corpus documents
        similarities = cosine_similarity(query_vector, self.corpus_vectors)[0]

        # Calculate the threshold based on percentile
        threshold = np.percentile(similarities, percentile)

        # Find documents above the threshold
        matching_indices = np.where(similarities >= threshold)[0]

        # Create results list with doc_id and similarity score
        results = [
            (self.doc_ids[idx], similarities[idx])
            for idx in matching_indices
        ]

        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def print_results(self, results: List[Tuple[str, float]], percentile: float):
        """
        Print the matching results in a formatted way

        Args:
            results: List of (doc_id, similarity_score) tuples
            percentile: The percentile threshold used
        """
        print(f"\n{'='*70}")
        print(f"Documents matching above {percentile}th percentile")
        print(f"Found {len(results)} matching documents")
        print(f"{'='*70}\n")

        for i, (doc_id, score) in enumerate(results, 1):
            print(f"{i:3d}. {doc_id:20s} | Similarity: {score:.4f}")
