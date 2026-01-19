"""
Unit tests for DocumentMatcher

Basic tests to make sure the core functionality works.
Not comprehensive, but covers the main features.
"""

import unittest
from src.document_matcher import DocumentMatcher


class TestDocumentMatcher(unittest.TestCase):
    """Test cases for the DocumentMatcher class"""

    def setUp(self):
        """Set up test data"""
        self.matcher = DocumentMatcher()
        # using a small test corpus so tests run fast
        self.test_corpus = [
            "This is a document about machine learning and AI",
            "Python programming is great for data science",
            "Machine learning models need training data",
            "The weather today is sunny and warm"
        ]
        self.test_doc_ids = ["doc1", "doc2", "doc3", "doc4"]

    def test_fit_corpus(self):
        """Test that corpus fitting works"""
        self.matcher.fit_corpus(self.test_corpus, self.test_doc_ids)

        # check that vectorizer was created
        self.assertIsNotNone(self.matcher.vectorizer)
        self.assertIsNotNone(self.matcher.corpus_vectors)
        self.assertEqual(len(self.matcher.corpus), 4)

    def test_find_similar_documents(self):
        """Test finding similar documents"""
        self.matcher.fit_corpus(self.test_corpus, self.test_doc_ids)

        query = "Machine learning and artificial intelligence"
        results = self.matcher.find_similar_documents(query, percentile=50)

        # basic checks
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # check that results are (doc_id, score) tuples
        for doc_id, score in results:
            self.assertIsInstance(doc_id, str)
            self.assertIsInstance(score, float)
            # similarity scores should be between 0 and 1
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_percentile_filtering(self):
        """Test that percentile filtering works correctly"""
        self.matcher.fit_corpus(self.test_corpus, self.test_doc_ids)

        query = "Machine learning"

        # higher percentile = fewer results (more selective)
        results_50 = self.matcher.find_similar_documents(query, percentile=50)
        results_90 = self.matcher.find_similar_documents(query, percentile=90)

        # 90th percentile should give us fewer or equal results than 50th
        self.assertGreaterEqual(len(results_50), len(results_90))


if __name__ == '__main__':
    unittest.main()
