# src/services/indexing_service.py

from langchain_community.document_loaders import PyPDFLoader

from src.rag.loader import load_all_pdfs
from src.rag.chunker import split_documents


def process_documents():
    """
    Load and chunk all PDFs from the uploads directory.

    Used when creating the FAISS index from scratch.
    """
    documents = load_all_pdfs()

    if not documents:
        return []

    return split_documents(documents)


def process_single_pdf(pdf_path: str):
    """
    Load and chunk one uploaded PDF.

    Used when a new paper is uploaded from the UI.
    """
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    if not documents:
        raise ValueError("No readable text was found in the PDF.")

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("The PDF could not be converted into text chunks.")

    return chunks