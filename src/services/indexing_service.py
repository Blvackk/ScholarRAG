from src.rag.loader import load_all_pdfs
from src.rag.chunker import split_documents


def process_documents():
    """Load all PDFs and split them into chunks."""

    documents = load_all_pdfs()
    chunks = split_documents(documents)

    return chunks