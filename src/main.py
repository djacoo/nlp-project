"""
Main entry point for the Document Similarity Matcher
NLP Assignment - Option C
"""

import sys
from corpus_loader import CorpusLoader
from document_matcher import DocumentMatcher


def read_document_from_file(file_path: str) -> str:
    """
    Read document text from a file

    Args:
        file_path: Path to the document file

    Returns:
        Document text
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def get_user_input() -> tuple:
    """
    Get document and percentile from user

    Returns:
        Tuple of (document_text, percentile)
    """
    print("\n" + "="*70)
    print("Document Similarity Matcher - NLP Assignment Option C")
    print("="*70)

    # Get document input
    print("\nHow would you like to provide the document?")
    print("1. Enter text directly")
    print("2. Provide a file path")

    choice = input("\nEnter your choice (1 or 2): ").strip()

    if choice == "1":
        print("\nEnter your document text (press Ctrl+D or Ctrl+Z when done):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        document_text = "\n".join(lines)
    elif choice == "2":
        file_path = input("\nEnter the file path: ").strip()
        document_text = read_document_from_file(file_path)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # Get percentile
    while True:
        try:
            percentile = float(input("\nEnter match percentile (0-100): ").strip())
            if 0 <= percentile <= 100:
                break
            else:
                print("Please enter a value between 0 and 100")
        except ValueError:
            print("Please enter a valid number")

    return document_text, percentile


def main():
    """Main function"""
    # Load corpus
    loader = CorpusLoader()
    corpus, doc_ids = loader.load_corpus()

    # Initialize matcher
    matcher = DocumentMatcher()
    matcher.fit_corpus(corpus, doc_ids)

    # Get user input
    document_text, percentile = get_user_input()

    # Find similar documents
    print("\nSearching for similar documents...")
    results = matcher.find_similar_documents(document_text, percentile)

    # Display results
    matcher.print_results(results, percentile)

    # Option to save results
    save = input("\nWould you like to save results to a file? (y/n): ").strip().lower()
    if save == 'y':
        output_file = input("Enter output filename: ").strip()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Documents matching above {percentile}th percentile\n")
            f.write(f"Found {len(results)} matching documents\n\n")
            for i, (doc_id, score) in enumerate(results, 1):
                f.write(f"{i}. {doc_id} | Similarity: {score:.4f}\n")
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
