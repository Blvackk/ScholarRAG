# config.py

import os
from dataclasses import dataclass

# ==========================================================
# Project Root Directory
# ==========================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Config:
    """
    Global configuration for ScholarRAG.
    """

    # ======================================================
    # Project Paths
    # ======================================================

    ROOT_DIR = ROOT_DIR

    # Upload Directory
    UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")

    # All uploaded PDFs
    PDF_GLOB = os.path.join(UPLOAD_DIR, "*.pdf")

    # FAISS Vector Store
    INDEX_DIR = os.path.join(ROOT_DIR, "faiss_index")

    # Local GGUF Model
    MODEL_PATH = os.path.join(
        ROOT_DIR,
        "models",
        "gemma-2-2b-it-Q4_K_M.gguf"
    )

    # ======================================================
    # RAG Configuration
    # ======================================================

    CHUNK_SIZE = 2048
    CHUNK_OVERLAP = 512

    EMBEDDING_MODEL = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ======================================================
    # LLM Configuration
    # ======================================================

    LLM_THREADS = 8
    LLM_CTX = 4096

    # ======================================================
    # Upload Configuration (NEW)
    # ======================================================

    ALLOWED_EXTENSIONS = {".pdf"}

    MAX_UPLOAD_SIZE_MB = 100

    # ======================================================
    # Application Configuration (NEW)
    # ======================================================

    APP_NAME = "ScholarRAG"

    APP_VERSION = "1.1.0"