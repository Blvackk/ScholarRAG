# src/services/indexing_service.py

from src.rag.loader import load_all_pdfs, load_pdf
from src.rag.chunker import split_documents


def process_documents():
    """
    Load, clean, and chunk all PDFs from the uploads directory.

    Used when creating the FAISS index from scratch.
    """

    documents = load_all_pdfs()

    if not documents:
        return []

    return split_documents(documents)


def process_single_pdf(pdf_path: str):
    """
    Load, clean, and chunk one uploaded PDF.

    Uses the same loading and cleaning pipeline as the
    multi-document indexing path.
    """

    documents = load_pdf(pdf_path)

    if not documents:
        raise ValueError(
            "No readable text was found in the PDF."
        )

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError(
            "The PDF could not be converted into text chunks."
        )

    return chunks