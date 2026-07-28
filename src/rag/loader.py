#loader.py
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import PyPDFLoader

from config import Config
from src.utils.helpers import clean_text


def load_pdf(pdf_path):
    """Load and clean a single PDF."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    documents = []

    for page in pages:
        cleaned_text = clean_text(page.page_content)

        if len(cleaned_text) < 20:
            continue

        page.page_content = cleaned_text
        page.metadata["source"] = os.path.basename(pdf_path)

        documents.append(page)

    return documents


def load_all_pdfs():
    """Load all PDFs from the configured directory."""

    pdf_files = glob(Config.PDF_GLOB)

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside:\n{Config.PDF_GLOB}"
        )

    documents = []

    with ThreadPoolExecutor() as executor:
        for docs in executor.map(load_pdf, pdf_files):
            documents.extend(docs)

    return documents