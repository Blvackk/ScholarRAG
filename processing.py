# processing.py

import os
import re
from glob import glob
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import Config


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def process_documents():
    """Load all PDFs and split into chunks."""
    pdf_files = glob(Config.PDF_GLOB)

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside:\n{Config.PDF_GLOB}"
        )

    documents = []

    with ThreadPoolExecutor() as executor:
        for docs in executor.map(load_pdf, pdf_files):
            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ Loaded {len(documents)} pages")
    print(f"✅ Created {len(chunks)} chunks")

    return chunks


@lru_cache(maxsize=1)
def get_embeddings():
    """Load embedding model only once."""
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )


def load_or_create_index():
    """Load an existing FAISS index or create a new one."""

    embeddings = get_embeddings()

    faiss_file = os.path.join(Config.INDEX_DIR, "index.faiss")
    pkl_file = os.path.join(Config.INDEX_DIR, "index.pkl")

    # Load existing index only if both files exist
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        print("📂 Loading existing FAISS index...")

        return FAISS.load_local(
            Config.INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("📁 No existing index found.")
    print("⚡ Creating a new FAISS index...")

    chunks = process_documents()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs(Config.INDEX_DIR, exist_ok=True)

    vector_store.save_local(Config.INDEX_DIR)

    print("✅ FAISS index created successfully.")

    return vector_store