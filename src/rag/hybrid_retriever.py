# src/rag/hybrid_retriever.py

import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import PrivateAttr


# ==========================================================
# Configuration
# ==========================================================

TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60


# ==========================================================
# Tokenization for BM25
# ==========================================================

def _tokenize(text: str):
    """
    Convert text into simple lowercase tokens for BM25.
    """

    if not text:
        return []

    return re.findall(
        r"[a-z0-9]+",
        text.lower()
    )


# ==========================================================
# Hybrid Retriever
# ==========================================================

class HybridRetriever(BaseRetriever):
    """
    ScholarRAG production hybrid retriever.

    Retrieval pipeline:

        Question
            |
            +---- FAISS semantic retrieval
            |
            +---- BM25 keyword retrieval
            |
            v
        Reciprocal Rank Fusion (RRF)
            |
            v
        Top-K documents
    """

    _vector_store: object = PrivateAttr()
    _documents: list = PrivateAttr()
    _bm25: object = PrivateAttr()

    _top_k: int = PrivateAttr()
    _candidate_k: int = PrivateAttr()
    _rrf_k: int = PrivateAttr()


    def __init__(
        self,
        vector_store,
        top_k=TOP_K,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        **kwargs
    ):
        super().__init__(**kwargs)

        if vector_store is None:
            raise ValueError(
                "HybridRetriever requires a valid FAISS vector store."
            )

        self._vector_store = vector_store

        self._top_k = top_k
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

        # --------------------------------------------------
        # Recover documents already stored inside FAISS
        # --------------------------------------------------

        self._documents = self._extract_documents()

        if not self._documents:
            raise ValueError(
                "No documents were found inside the FAISS vector store."
            )

        # --------------------------------------------------
        # Build BM25 over the SAME chunks as FAISS
        # --------------------------------------------------

        tokenized_corpus = [
            _tokenize(doc.page_content)
            for doc in self._documents
        ]

        self._bm25 = BM25Okapi(
            tokenized_corpus
        )

        print(
            f"Hybrid retriever ready "
            f"({len(self._documents)} chunks)."
        )


    # ======================================================
    # Extract documents from FAISS
    # ======================================================

    def _extract_documents(self):
        """
        Recover LangChain Documents from the FAISS docstore.

        This ensures BM25 and FAISS operate over the same
        chunks without maintaining a second copy elsewhere.
        """

        documents = []

        index_to_docstore_id = (
            self._vector_store.index_to_docstore_id
        )

        docstore = self._vector_store.docstore

        for index in sorted(
            index_to_docstore_id.keys()
        ):
            doc_id = index_to_docstore_id[index]

            document = docstore.search(
                doc_id
            )

            if isinstance(document, Document):
                documents.append(document)

        return documents


    # ======================================================
    # Document identity
    # ======================================================

    @staticmethod
    def _document_key(document):
        """
        Create a stable key used to identify the same chunk
        returned by FAISS and BM25.
        """

        metadata = document.metadata or {}

        return (
            metadata.get("source", ""),
            metadata.get(
                "page",
                metadata.get("page_number", "")
            ),
            document.page_content,
        )


    # ======================================================
    # FAISS retrieval
    # ======================================================

    def _faiss_search(self, query):
        """
        Semantic retrieval using the existing FAISS index.
        """

        k = min(
            self._candidate_k,
            len(self._documents)
        )

        return self._vector_store.similarity_search(
            query,
            k=k
        )


    # ======================================================
    # BM25 retrieval
    # ======================================================

    def _bm25_search(self, query):
        """
        Keyword retrieval using BM25.
        """

        query_tokens = _tokenize(query)

        scores = self._bm25.get_scores(
            query_tokens
        )

        k = min(
            self._candidate_k,
            len(self._documents)
        )

        ranked_indices = np.argsort(
            scores
        )[::-1][:k]

        return [
            self._documents[int(index)]
            for index in ranked_indices
        ]


    # ======================================================
    # Reciprocal Rank Fusion
    # ======================================================

    def _rrf_fusion(
        self,
        faiss_documents,
        bm25_documents
    ):
        """
        Combine FAISS and BM25 rankings using equal-weight
        Reciprocal Rank Fusion.

        score(d) =
            1 / (RRF_K + FAISS_rank)
            +
            1 / (RRF_K + BM25_rank)
        """

        scores = defaultdict(float)
        documents = {}

        # --------------------------------------------------
        # FAISS rankings
        # --------------------------------------------------

        for rank, document in enumerate(
            faiss_documents,
            start=1
        ):
            key = self._document_key(
                document
            )

            documents[key] = document

            scores[key] += (
                1.0 /
                (self._rrf_k + rank)
            )

        # --------------------------------------------------
        # BM25 rankings
        # --------------------------------------------------

        for rank, document in enumerate(
            bm25_documents,
            start=1
        ):
            key = self._document_key(
                document
            )

            documents[key] = document

            scores[key] += (
                1.0 /
                (self._rrf_k + rank)
            )

        ranked_keys = sorted(
            scores.keys(),
            key=lambda key: scores[key],
            reverse=True
        )

        return [
            documents[key]
            for key in ranked_keys[
                :self._top_k
            ]
        ]


    # ======================================================
    # LangChain retrieval interface
    # ======================================================

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None
    ):
        """
        Called by LangChain whenever RetrievalQA needs
        context for a question.
        """

        faiss_documents = (
            self._faiss_search(query)
        )

        bm25_documents = (
            self._bm25_search(query)
        )

        final_documents = (
            self._rrf_fusion(
                faiss_documents,
                bm25_documents
            )
        )

        return final_documents


# ==========================================================
# Factory
# ==========================================================

def create_hybrid_retriever(
    vector_store,
    top_k=TOP_K,
    candidate_k=CANDIDATE_K,
):
    """
    Create the production ScholarRAG hybrid retriever.
    """

    return HybridRetriever(
        vector_store=vector_store,
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=RRF_K,
    )