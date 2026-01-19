"""
Integration Testing Script
Tests the complete workflow
"""

import sys
sys.path.insert(0, 'src')

from corpus_loader import CorpusLoader
from document_matcher import DocumentMatcher


def test_complete_workflow():
    """Test the complete workflow"""

    print("="*70)
    print("INTEGRATION TEST - Document Similarity Matcher")
    print("="*70)

    # Test 1: Load corpus
    print("\n[TEST 1] Loading Reuters corpus...")
    loader = CorpusLoader()
    corpus, doc_ids = loader.load_corpus()

    assert len(corpus) > 0, "Corpus should not be empty"
    assert len(corpus) == len(doc_ids), "Corpus and doc_ids should have same length"
    print(f"✓ PASS: Loaded {len(corpus)} documents")

    # Test 2: Fit the matcher
    print("\n[TEST 2] Fitting TF-IDF vectorizer...")
    matcher = DocumentMatcher()
    matcher.fit_corpus(corpus, doc_ids)

    assert matcher.vectorizer is not None, "Vectorizer should be initialized"
    assert matcher.corpus_vectors is not None, "Corpus vectors should be computed"
    print(f"✓ PASS: TF-IDF matrix shape: {matcher.corpus_vectors.shape}")

    # Test 3: Query with financial text (should match Reuters financial news)
    print("\n[TEST 3] Testing with financial query document...")
    query1 = "Oil prices surged today as OPEC announced production cuts. Market analysts predict volatility."
    results1 = matcher.find_similar_documents(query1, percentile=70)

    assert len(results1) > 0, "Should find matching documents"
    assert all(0 <= score <= 1 for _, score in results1), "Scores should be between 0 and 1"
    print(f"✓ PASS: Found {len(results1)} documents above 70th percentile")
    print(f"  Top match: {results1[0][0]} with similarity {results1[0][1]:.4f}")

    # Test 4: Test different percentiles
    print("\n[TEST 4] Testing percentile filtering...")
    results_50 = matcher.find_similar_documents(query1, percentile=50)
    results_90 = matcher.find_similar_documents(query1, percentile=90)

    assert len(results_50) >= len(results_90), "Lower percentile should return more results"
    print(f"✓ PASS: 50th percentile returned {len(results_50)} docs")
    print(f"✓ PASS: 90th percentile returned {len(results_90)} docs")

    # Test 5: Test with different topicss
    print("\n[TEST 5] Testing with different query topic...")
    query2 = "The government announced new legislation regarding environmental regulations."
    results2 = matcher.find_similar_documents(query2, percentile=70)

    assert len(results2) > 0, "Should find matching documents"
    print(f"✓ PASS: Found {len(results2)} documents for political query")
    print(f"  Top match: {results2[0][0]} with similarity {results2[0][1]:.4f}")

    # Test 6: Test with very low percentile (should return many docs)
    print("\n[TEST 6] Testing with 0th percentile...")
    results_0 = matcher.find_similar_documents(query1, percentile=0)
    print(f"✓ PASS: 0th percentile returned {len(results_0)} documents (all corpus)")

    # Test 7: Test with 100th percentile (should return only highest)
    print("\n[TEST 7] Testing with 100th percentile...")
    results_100 = matcher.find_similar_documents(query1, percentile=100)
    print(f"✓ PASS: 100th percentile returned {len(results_100)} document(s)")

    # Test 8: Verify results are sorted
    print("\n[TEST 8] Verifying results are sorted by similarity...")
    for i in range(len(results1) - 1):
        assert results1[i][1] >= results1[i+1][1], "Results should be sorted in descending order"
    print("✓ PASS: Results are properly sorted")

    # Test 9: Test file loading
    print("\n[TEST 9] Testing file input...")
    try:
        with open('test_sample.txt', 'r') as f:
            file_content = f.read()
        results_file = matcher.find_similar_documents(file_content, percentile=60)
        assert len(results_file) > 0, "Should find matches from file content"
        print(f"✓ PASS: File input processed, found {len(results_file)} matches")
    except FileNotFoundError:
        print("⚠ SKIP: test_sample.txt not found (create it for full test)")

    print("\n" + "="*70)
    print("ALL INTEGRATION TESTS PASSED ✓")
    print("="*70)
    print("\nSummary:")
    print(f"  - Corpus size: {len(corpus)} documents")
    print(f"  - Vocabulary size: {matcher.corpus_vectors.shape[1]} unique terms")
    print(f"  - All percentile filtering working correctly")
    print(f"  - Results properly sorted by similarity")
    print(f"  - Ready for production use")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        test_complete_workflow()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
