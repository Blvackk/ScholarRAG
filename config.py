# config.py

import os
from dataclasses import dataclass

# Absolute path to the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Config:
    """
    Global configuration for ScholarRAG.
    """

    # =========================
    # Project Paths
    # =========================
    ROOT_DIR = ROOT_DIR

    # Folder where uploaded PDFs are stored
    UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")

    # Load every PDF from uploads/
    PDF_GLOB = os.path.join(UPLOAD_DIR, "*.pdf")

    # Folder where the FAISS index is stored
    INDEX_DIR = os.path.join(ROOT_DIR, "faiss_index")

    # GGUF LLM model
    MODEL_PATH = os.path.join(
        ROOT_DIR,
        "models",
        "gemma-2-2b-it-Q4_K_M.gguf"
    )

    # =========================
    # RAG Settings
    # =========================
    CHUNK_SIZE = 2048
    CHUNK_OVERLAP = 512

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # =========================
    # LLM Settings
    # =========================
    LLM_THREADS = 8
    LLM_CTX = 4096