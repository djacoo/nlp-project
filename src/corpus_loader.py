"""
Corpus Loader Module
Handles loading and preprocessing of the Reuters corpus
"""

import nltk
from typing import List, Tuple


class CorpusLoader:
    """Loads and manages the Reuters corpus"""

    def __init__(self):
        """Initialize the corpus loader"""
        self.corpus = None
        self.doc_ids = None

    def download_reuters(self):
        """Download the Reuters corpus if not already available"""
        try:
            nltk.data.find('corpora/reuters')
        except LookupError:
            print("Downloading Reuters corpus...")
            nltk.download('reuters')
            print("Download complete!")

    def load_corpus(self) -> Tuple[List[str], List[str]]:
        """
        Load the Reuters corpus

        Returns:
            Tuple of (documents, document_ids)
        """
        self.download_reuters()

        from nltk.corpus import reuters

        self.doc_ids = reuters.fileids()
        self.corpus = [reuters.raw(doc_id) for doc_id in self.doc_ids]

        print(f"Loaded {len(self.corpus)} documents from Reuters corpus")

        return self.corpus, self.doc_ids

    def get_document_by_id(self, doc_id: str) -> str:
        """
        Get a specific document by its ID

        Args:
            doc_id: The document ID

        Returns:
            The document text
        """
        from nltk.corpus import reuters
        return reuters.raw(doc_id)
