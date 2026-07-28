import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

from src.rag.loader import load_pdf
from src.rag.chunker import split_documents
from src.rag.embeddings import get_embeddings


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
PAPERS_DIR = BASE_DIR / "papers"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_PATH = RESULTS_DIR / "hybrid_regression_diagnostic.json"

TARGET_ID = "q10"

SEARCH_K = 15
RRF_K = 60


def normalize_text(text):

    if not text:
        return ""

    text = text.lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")
    text = text.replace("-", " ")

    text = re.sub(r"[^\w\s%.]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text):
    return normalize_text(text).split()


def evidence_in_text(text, evidence):

    return (
        normalize_text(evidence)
        in normalize_text(text)
    )


def document_key(document):

    return (
        document.metadata.get("source", ""),
        document.metadata.get("page", ""),
        normalize_text(document.page_content),
    )


def display_page(document):

    page = document.metadata.get("page")

    if page is None:
        return None

    try:
        return int(page) + 1

    except (TypeError, ValueError):
        return page


def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def load_papers():

    pdf_files = sorted(
        PAPERS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PAPERS_DIR}"
        )

    documents = []

    print("\n📄 Loading papers...")

    for pdf_path in pdf_files:

        print(f"   Loading: {pdf_path.name}")

        documents.extend(
            load_pdf(str(pdf_path))
        )

    print(
        f"✅ Loaded {len(documents)} pages."
    )

    return documents


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        corpus = [
            tokenize(document.page_content)
            for document in documents
        ]

        self.bm25 = BM25Okapi(corpus)

    def search(self, query, k):

        scores = self.bm25.get_scores(
            tokenize(query)
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        return [
            self.documents[int(index)]
            for index in ranked_indices[:k]
        ]


def reciprocal_rank_fusion(
    faiss_documents,
    bm25_documents,
):

    scores = {}
    documents = {}

    for rank, document in enumerate(
        faiss_documents,
        start=1,
    ):

        key = document_key(document)

        documents[key] = document

        scores[key] = (
            scores.get(key, 0.0)
            + 1 / (RRF_K + rank)
        )

    for rank, document in enumerate(
        bm25_documents,
        start=1,
    ):

        key = document_key(document)

        documents[key] = document

        scores[key] = (
            scores.get(key, 0.0)
            + 1 / (RRF_K + rank)
        )

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        (documents[key], scores[key])
        for key in ranked_keys
    ]


def find_evidence_rank(
    documents,
    evidence,
):

    for rank, document in enumerate(
        documents,
        start=1,
    ):

        if evidence_in_text(
            document.page_content,
            evidence,
        ):

            return rank, display_page(document)

    return None, None


def main():

    print(
        "\n🔬 Starting ScholarRAG "
        "Hybrid Regression Diagnostic"
    )

    dataset = load_dataset()

    item = next(
        question
        for question in dataset
        if question["id"] == TARGET_ID
    )

    query = item["question"]
    expected_evidence = item["expected_evidence"]

    print(f"\n🎯 Target: {TARGET_ID}")
    print(f"Question: {query}")

    print("\nExpected Evidence:")

    for evidence in expected_evidence:
        print(f"  • {evidence}")

    documents = load_papers()

    print("\n✂️ Creating chunks...")

    chunks = split_documents(documents)

    print(
        f"✅ Created {len(chunks)} chunks."
    )

    print("\n🧠 Loading embeddings...")

    embeddings = get_embeddings()

    print("✅ Embeddings ready.")

    print("\n🔎 Building FAISS...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    print("✅ FAISS ready.")

    print("\n🔤 Building BM25...")

    bm25 = BM25Retriever(chunks)

    print("✅ BM25 ready.")

    # --------------------------------------------------------
    # Retrieve 15 from both systems
    # --------------------------------------------------------

    faiss_documents = (
        vector_store.similarity_search(
            query,
            k=SEARCH_K,
        )
    )

    bm25_documents = bm25.search(
        query,
        SEARCH_K,
    )

    fused_results = reciprocal_rank_fusion(
        faiss_documents,
        bm25_documents,
    )

    fused_documents = [
        document
        for document, score
        in fused_results
    ]

    # --------------------------------------------------------
    # Evidence ranks
    # --------------------------------------------------------

    results = {}

    print("\n" + "=" * 88)
    print("Q10 EVIDENCE RANK DIAGNOSTIC")
    print("=" * 88)

    for evidence in expected_evidence:

        faiss_rank, faiss_page = (
            find_evidence_rank(
                faiss_documents,
                evidence,
            )
        )

        bm25_rank, bm25_page = (
            find_evidence_rank(
                bm25_documents,
                evidence,
            )
        )

        hybrid_rank, hybrid_page = (
            find_evidence_rank(
                fused_documents,
                evidence,
            )
        )

        results[evidence] = {
            "faiss_rank": faiss_rank,
            "faiss_page": faiss_page,
            "bm25_rank": bm25_rank,
            "bm25_page": bm25_page,
            "hybrid_rank": hybrid_rank,
            "hybrid_page": hybrid_page,
        }

        print(f"\nEvidence: {evidence}")

        print(
            f"FAISS  : "
            f"{'Rank ' + str(faiss_rank) if faiss_rank else 'Not found'}"
            f"{' | Page ' + str(faiss_page) if faiss_page else ''}"
        )

        print(
            f"BM25   : "
            f"{'Rank ' + str(bm25_rank) if bm25_rank else 'Not found'}"
            f"{' | Page ' + str(bm25_page) if bm25_page else ''}"
        )

        print(
            f"HYBRID : "
            f"{'Rank ' + str(hybrid_rank) if hybrid_rank else 'Not found'}"
            f"{' | Page ' + str(hybrid_page) if hybrid_page else ''}"
        )

    # --------------------------------------------------------
    # Show fused Top-10
    # --------------------------------------------------------

    print("\n" + "=" * 88)
    print("HYBRID TOP-10")
    print("=" * 88)

    for rank, (document, score) in enumerate(
        fused_results[:10],
        start=1,
    ):

        page = display_page(document)

        matched = [
            evidence
            for evidence in expected_evidence
            if evidence_in_text(
                document.page_content,
                evidence,
            )
        ]

        print(
            f"\nRank {rank}"
            f" | Page {page}"
            f" | RRF {score:.6f}"
        )

        print(
            f"Evidence: "
            f"{matched if matched else 'None'}"
        )

        preview = (
            document.page_content
            .replace("\n", " ")
            [:250]
        )

        print(
            f"Text: {preview}..."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "target": TARGET_ID,
        "question": query,
        "search_k": SEARCH_K,
        "rrf_k": RRF_K,
        "evidence": results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "\n💾 Results saved to:"
    )

    print(OUTPUT_PATH)

    print(
        "\n✅ Regression diagnostic complete."
    )


if __name__ == "__main__":
    main()