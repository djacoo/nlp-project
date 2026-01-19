"""
Unit tests for DocumentMatcher
"""

import unittest
from src.document_matcher import DocumentMatcher


class TestDocumentMatcher(unittest.TestCase):
    """Test cases for DocumentMatcher class"""

    def setUp(self):
        """Set up test fixtures"""
        self.matcher = DocumentMatcher()
        self.test_corpus = [
            "This is a document about machine learning and AI",
            "Python programming is great for data science",
            "Machine learning models need training data",
            "The weather today is sunny and warm"
        ]
        self.test_doc_ids = ["doc1", "doc2", "doc3", "doc4"]

    def test_fit_corpus(self):
        """Test corpus fitting"""
        self.matcher.fit_corpus(self.test_corpus, self.test_doc_ids)
        self.assertIsNotNone(self.matcher.vectorizer)
        self.assertIsNotNone(self.matcher.corpus_vectors)
        self.assertEqual(len(self.matcher.corpus), 4)

    def test_find_similar_documents(self):
        """Test finding similar documents"""
        self.matcher.fit_corpus(self.test_corpus, self.test_doc_ids)

        query = "Machine learning and artificial intelligence"
        results = self.matcher.find_similar_documents(query, percentile=50)

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # Check that results are tuples of (doc_id, score)
        for doc_id, score in results:
            self.assertIsInstance(doc_id, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_percentile_filtering(self):
        """Test that percentile filtering works correctly"""
        self.matcher.fit_corpus(self.test_corpus, self.test_doc_ids)

        query = "Machine learning"

        # Higher percentile should return fewer or equal results
        results_50 = self.matcher.find_similar_documents(query, percentile=50)
        results_90 = self.matcher.find_similar_documents(query, percentile=90)

        self.assertGreaterEqual(len(results_50), len(results_90))


if __name__ == '__main__':
    unittest.main()
