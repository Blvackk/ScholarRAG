# src/rag/vector_store.py

import os

from langchain_community.vectorstores import FAISS

from config import Config
from src.rag.embeddings import get_embeddings
from src.services.indexing_service import (
    process_documents,
    process_single_pdf,
)


def load_or_create_index():
    """
    Load the existing FAISS index.

    If no index exists, create one from PDFs currently
    available in the uploads directory.
    """

    embeddings = get_embeddings()

    faiss_file = os.path.join(
        Config.INDEX_DIR,
        "index.faiss"
    )

    pkl_file = os.path.join(
        Config.INDEX_DIR,
        "index.pkl"
    )

    if os.path.exists(faiss_file) and os.path.exists(pkl_file):

        print("📂 Loading existing FAISS index...")

        return FAISS.load_local(
            Config.INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("📁 No existing FAISS index found.")

    chunks = process_documents()

    if not chunks:
        return None

    print("⚡ Creating FAISS index...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs(
        Config.INDEX_DIR,
        exist_ok=True
    )

    vector_store.save_local(
        Config.INDEX_DIR
    )

    print("✅ FAISS index created.")

    return vector_store


def create_index_for_pdf(pdf_path: str):
    """
    Create a new FAISS index for one uploaded research paper.

    The newly uploaded PDF becomes the active paper.
    """

    print("📄 Processing uploaded paper...")

    chunks = process_single_pdf(pdf_path)

    print(
        f"✂️ Created {len(chunks)} chunks."
    )

    embeddings = get_embeddings()

    print("🧠 Generating embeddings...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs(
        Config.INDEX_DIR,
        exist_ok=True
    )

    vector_store.save_local(
        Config.INDEX_DIR
    )

    print("✅ Active paper indexed successfully.")

    return vector_store