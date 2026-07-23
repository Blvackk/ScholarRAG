#vector_store.py
import os

from langchain_community.vectorstores import FAISS

from config import Config
from src.services.indexing_service import process_documents
from src.rag.embeddings import get_embeddings


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