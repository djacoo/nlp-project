"""
Main Entry Point - Document Similarity Matcher
NLP Assignment - Option C

This is the main program that the user runs. It handles:
- Loading the Reuters corpus
- Getting input from the user
- Finding similar documents
- Displaying results

Usage: python main.py
"""

import sys
from corpus_loader import CorpusLoader
from document_matcher import DocumentMatcher


def read_document_from_file(file_path: str) -> str:
    """
    Reads a document from a file path

    Handles errors if file doesn't exist or can't be read.
    Supports any text file (.txt, .md, etc.)

    Args:
        file_path: path to the document

    Returns:
        the file contents as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        print("Please check the path and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def get_user_input() -> tuple:
    """
    Gets the document and percentile from input

    The assignment requires two inputs:
    1. A document (can be typed or from file)
    2. A percentile value (0-100) for filtering results

    Returns:
        (document_text, percentile) tuple
    """
    print("\n" + "="*70)
    print("Document Similarity Matcher - NLP Assignment Option C")
    print("="*70)

    # get the document
    print("\nHow would you provide the document?")
    print("1. Enter text directly")
    print("2. Provide a file path")

    choice = input("\nEnter (1 or 2): ").strip()

    if choice == "1":
        # option 1: type it in
        print("\nEnter your document text (press Ctrl+D on Linux or Ctrl+Z on Windows to finish):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            # user pressed Ctrl+D or Ctrl+Z
            pass
        document_text = "\n".join(lines)

    elif choice == "2":
        # option 2: read from file
        file_path = input("\nEnter the file path: ").strip()
        document_text = read_document_from_file(file_path)
        print(f"\nDocument loaded successfully ({len(document_text)} characters)")

    else:
        print("Invalid choice. Please run again and choose 1 or 2.")
        sys.exit(1)

    # get the percentile
    # keep asking until we get a valid number between 0 and 100
    while True:
        try:
            percentile = float(input("\nEnter match percentile (0-100): ").strip())

            if 0 <= percentile <= 100:
                break
            else:
                print("Please enter a value between 0 and 100")
                print("  - Higher percentile = fewer, more similar documents")
                print("  - Lower percentile = more, less similar documents")

        except ValueError:
            print("Please enter a valid number")

    return document_text, percentile


def main():
    """
    Main function that runs everything

    Steps:
    1. Load corpus and compute TF-IDF
    2. Get user input
    3. Find similar documents
    4. Display results
    5. Optionally save to file
    """

    print("\n" + "="*70)
    print("INITIALIZING DOCUMENT SIMILARITY MATCHER")
    print("="*70)

    # Step 1: Load corpus
    print("\n[1/4] Loading Reuters corpus...")
    loader = CorpusLoader()
    corpus, doc_ids = loader.load_corpus()
    print(f"✓ Loaded {len(corpus)} documents")

    # Step 2: Compute TF-IDF vectors
    print("\n[2/4] Computing TF-IDF vectors...")
    matcher = DocumentMatcher()
    matcher.fit_corpus(corpus, doc_ids)
    print("✓ TF-IDF computation complete")

    # Step 3: Get user input
    print("\n[3/4] Getting user input...")
    document_text, percentile = get_user_input()
    print(f"✓ Document received ({len(document_text)} characters)")
    print(f"✓ Percentile threshold: {percentile}")

    # Step 4: Find similar documents
    print("\n[4/4] Finding similar documents...")
    results = matcher.find_similar_documents(document_text, percentile)
    print(f"✓ Search complete")

    # Display results
    matcher.print_results(results, percentile)

    # Optional: save to file
    save = input("\nWould you like to save results to a file? (y/n): ").strip().lower()
    if save == 'y':
        output_file = input("Enter output filename (e.g., results.txt): ").strip()

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("Document Similarity Matching Results\n")
                f.write("NLP Assignment - Option C\n")
                f.write("="*70 + "\n\n")

                f.write(f"Percentile threshold: {percentile}\n")
                f.write(f"Total matching documents: {len(results)}\n\n")

                f.write("Matching Documents:\n")
                f.write("-"*70 + "\n")
                for i, (doc_id, score) in enumerate(results, 1):
                    f.write(f"{i:3d}. {doc_id:20s} | Similarity: {score:.4f}\n")

                f.write("\n" + "="*70 + "\n")

            print(f"✓ Results saved to {output_file}")

        except Exception as e:
            print(f"Error saving file: {e}")

    print("\n" + "="*70)
    print("Program completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
