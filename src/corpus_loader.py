"""
Corpus Loader Module

This handles loading the Reuters corpus from NLTK.
Reuters is a collection of about 10,788 news articles from 1987 
It's got financial news which works well for testing similarity.

--no preprocessing
"""

import nltk
from typing import List, Tuple


class CorpusLoader:
    """
    Loads and manages the Reuters corpus

   Class usage:
    - Download Reuters if it's not already there
    - Load all the documents
    - Keep track of document IDs so they can be referenced later

    The corpus and doc_ids get set when load_corpus() is called
    """

    def __init__(self):
        """Initialize with empty values - actual loading happens in load_corpus()"""
        self.corpus = None      # will hold the document texts
        self.doc_ids = None     # will hold document identifiers

    def download_reuters(self):
        """
        Downloads Reuters corpus if needed

        NLTK stores data separately, so first time run it'll download ~2MB.
        After that it just checks if it's there and skips the download.

        Put this in a separate method so it's not cluttering up load_corpus()
        """
        try:
            # check if we already have it
            nltk.data.find('corpora/reuters')
        except LookupError:
            # nope, need to download
            print("Downloading Reuters corpus...")
            print("(This only happens once - approximately 2MB download)")
            nltk.download('reuters')
            print("Download complete!")

    def load_corpus(self) -> Tuple[List[str], List[str]]:
        """
        Main loading method - gets all Reuters documents

        What this does:
        1. Makes sure corpus is downloaded
        2. Gets all document IDs (filenames)
        3. Loads raw text for each document
        4. Returns both texts and IDs

        Returns:
            (documents, doc_ids) - two lists with the texts and their IDs

        Important: using reuters.raw() which gives the complete unprocessed text.
        """
        # first make sure we have the data
        self.download_reuters()

        # import reuters from NLTK
        from nltk.corpus import reuters

        # get all the document IDs - these look like "test/14826", "training/9865", etc.
        self.doc_ids = reuters.fileids()

        # load the actual text for each document
        # reuters.raw(doc_id) gives us the complete text without any processing
        # using list comprehension here to do it in one line
        self.corpus = [reuters.raw(doc_id) for doc_id in self.doc_ids]

        print(f"Loaded {len(self.corpus)} documents from Reuters corpus")

        return self.corpus, self.doc_ids

    def get_document_by_id(self, doc_id: str) -> str:
        """
        Get a specific document by its ID

        might be useful later
        if need to show the actual content of matched documents, or for debugging.

        Args:
            doc_id: document identifier (like "test/14826")

        Returns:
            the full text of that document
        """
        from nltk.corpus import reuters
        return reuters.raw(doc_id)
