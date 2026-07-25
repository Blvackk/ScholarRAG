from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import Config


@lru_cache(maxsize=1)
def get_embeddings():
    """Load embedding model only once."""

    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )