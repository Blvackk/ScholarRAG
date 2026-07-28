import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

from src.rag.loader import load_pdf
from src.rag.chunker import split_documents
from src.rag.embeddings import get_embeddings


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset.json"
PAPERS_DIR = BASE_DIR / "papers"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_PATH = RESULTS_DIR / "hybrid_retrieval_experiment.json"

TOP_K = 5

# Number of candidates collected from each retriever before fusion
CANDIDATE_K = 10

# Reciprocal Rank Fusion constant
RRF_K = 60


# ============================================================
# Expanded Queries
# ============================================================

EXPANDED_QUERIES = {
    "q04": (
        "What research methodology, study design, literature review "
        "method, data collection approach, experimental procedure, "
        "and research method were used in this study?"
    ),

    "q08": (
        "What mitigation strategies, interventions, adaptive autonomy "
        "methods, explainable AI interface designs, micro-interventions, "
        "and workload management techniques are proposed for reducing "
        "mental fatigue in human-in-the-loop AI systems?"
    ),
}


TARGET_IDS = ["q04", "q08"]


# ============================================================
# Text Helpers
# ============================================================

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


# ============================================================
# Dataset
# ============================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# Load PDFs
# ============================================================

def load_papers():

    pdf_files = sorted(
        PAPERS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            f"No evaluation PDFs found in:\n"
            f"{PAPERS_DIR}"
        )

    documents = []

    print("\n📄 Loading evaluation papers...")

    for pdf_path in pdf_files:

        print(
            f"   Loading: {pdf_path.name}"
        )

        docs = load_pdf(
            str(pdf_path)
        )

        documents.extend(docs)

    print(
        f"✅ Loaded {len(documents)} pages."
    )

    return documents


# ============================================================
# Document Identity
# ============================================================

def document_key(document):

    """
    Generate a stable key for identifying the same chunk
    across FAISS and BM25 results.
    """

    source = document.metadata.get(
        "source",
        ""
    )

    page = document.metadata.get(
        "page",
        ""
    )

    content = normalize_text(
        document.page_content
    )

    return (
        source,
        page,
        content,
    )


# ============================================================
# BM25 Retriever
# ============================================================

class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        tokenized_corpus = [
            tokenize(document.page_content)
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    def search(self, query, k=5):

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices[:k]:

            results.append(
                self.documents[
                    int(index)
                ]
            )

        return results


# ============================================================
# FAISS Search
# ============================================================

def faiss_search(
    vector_store,
    query,
    k,
):

    return vector_store.similarity_search(
        query,
        k=k,
    )


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    dense_documents,
    lexical_documents,
    top_k,
):

    """
    Fuse FAISS and BM25 rankings using Reciprocal Rank Fusion.

    Score:
        1 / (RRF_K + rank)

    A document retrieved highly by both systems receives
    contributions from both rankings.
    """

    scores = {}
    documents_by_key = {}

    # Dense / FAISS
    for rank, document in enumerate(
        dense_documents,
        start=1,
    ):

        key = document_key(document)

        documents_by_key[key] = document

        scores[key] = (
            scores.get(key, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    # Lexical / BM25
    for rank, document in enumerate(
        lexical_documents,
        start=1,
    ):

        key = document_key(document)

        documents_by_key[key] = document

        scores[key] = (
            scores.get(key, 0.0)
            + 1.0 / (RRF_K + rank)
        )

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    fused_documents = [
        documents_by_key[key]
        for key in ranked_keys[:top_k]
    ]

    return fused_documents


# ============================================================
# Hybrid Search
# ============================================================

def hybrid_search(
    vector_store,
    bm25_retriever,
    query,
):

    dense_documents = faiss_search(
        vector_store,
        query,
        CANDIDATE_K,
    )

    lexical_documents = (
        bm25_retriever.search(
            query,
            CANDIDATE_K,
        )
    )

    return reciprocal_rank_fusion(
        dense_documents,
        lexical_documents,
        TOP_K,
    )


# ============================================================
# Evaluate Retrieved Documents
# ============================================================

def evaluate_documents(
    documents,
    expected_evidence,
):

    matched = []
    missing = []

    evidence_ranks = {}

    for evidence in expected_evidence:

        found_rank = None

        for rank, document in enumerate(
            documents,
            start=1,
        ):

            if evidence_in_text(
                document.page_content,
                evidence,
            ):

                found_rank = rank
                break

        evidence_ranks[evidence] = (
            found_rank
        )

        if found_rank is None:
            missing.append(evidence)
        else:
            matched.append(evidence)

    total = len(expected_evidence)

    coverage = (
        len(matched) / total
        if total
        else 0.0
    )

    full_evidence = (
        total > 0
        and len(missing) == 0
    )

    pages = []

    for document in documents:

        page = document.metadata.get(
            "page"
        )

        if page is not None:

            try:
                page = int(page) + 1

            except (TypeError, ValueError):
                pass

        pages.append(page)

    return {
        "matched_evidence": matched,
        "missing_evidence": missing,
        "coverage": coverage,
        "full_evidence": full_evidence,
        "evidence_ranks": evidence_ranks,
        "retrieved_pages": pages,
    }


# ============================================================
# Run Configuration
# ============================================================

def run_configuration(
    vector_store,
    bm25_retriever,
    query,
    expected_evidence,
    method,
):

    if method == "faiss":

        documents = faiss_search(
            vector_store,
            query,
            TOP_K,
        )

    elif method == "hybrid":

        documents = hybrid_search(
            vector_store,
            bm25_retriever,
            query,
        )

    else:

        raise ValueError(
            f"Unknown retrieval method: "
            f"{method}"
        )

    result = evaluate_documents(
        documents,
        expected_evidence,
    )

    result["query"] = query
    result["method"] = method

    return result


# ============================================================
# Print Result
# ============================================================

def print_result(
    label,
    result,
):

    print(f"\n{label}")

    print(
        f"Coverage : "
        f"{result['coverage']:.3f}"
    )

    print(
        "Full     : "
        f"{'✅' if result['full_evidence'] else '❌'}"
    )

    print(
        f"Matched  : "
        f"{result['matched_evidence']}"
    )

    print(
        f"Missing  : "
        f"{result['missing_evidence']}"
    )

    print(
        f"Pages    : "
        f"{result['retrieved_pages']}"
    )

    print("Evidence ranks:")

    for evidence, rank in (
        result["evidence_ranks"].items()
    ):

        if rank is None:

            print(
                f"   ❌ {evidence}: "
                f"not in Top-{TOP_K}"
            )

        else:

            print(
                f"   ✅ {evidence}: "
                f"rank {rank}"
            )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n🧪 Starting ScholarRAG "
        "Hybrid Retrieval Experiment"
    )

    dataset = load_dataset()

    dataset_by_id = {
        item["id"]: item
        for item in dataset
    }

    # --------------------------------------------------------
    # Load PDFs
    # --------------------------------------------------------

    documents = load_papers()

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    print("\n✂️ Creating chunks...")

    chunks = split_documents(
        documents
    )

    print(
        f"✅ Created {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # Embeddings / FAISS
    # --------------------------------------------------------

    print("\n🧠 Loading embeddings...")

    embeddings = get_embeddings()

    print("✅ Embeddings ready.")

    print(
        "\n🔎 Building FAISS index..."
    )

    vector_store = (
        FAISS.from_documents(
            chunks,
            embeddings,
        )
    )

    print("✅ FAISS ready.")

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print(
        "\n🔤 Building BM25 index..."
    )

    bm25_retriever = BM25Retriever(
        chunks
    )

    print("✅ BM25 ready.")

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print("\n" + "=" * 96)
    print(
        "SCHOLARRAG HYBRID RETRIEVAL EXPERIMENT"
    )
    print("=" * 96)

    experiment_results = []

    for question_id in TARGET_IDS:

        item = dataset_by_id[
            question_id
        ]

        original_query = (
            item["question"]
        )

        expanded_query = (
            EXPANDED_QUERIES[
                question_id
            ]
        )

        expected_evidence = (
            item["expected_evidence"]
        )

        print(
            f"\n[{question_id}] "
            f"{original_query}"
        )

        print("\nExpected Evidence:")

        for evidence in expected_evidence:

            print(
                f"  • {evidence}"
            )

        # ====================================================
        # A. Original + FAISS
        # ====================================================

        faiss_original = (
            run_configuration(
                vector_store,
                bm25_retriever,
                original_query,
                expected_evidence,
                "faiss",
            )
        )

        # ====================================================
        # B. Expanded + FAISS
        # ====================================================

        faiss_expanded = (
            run_configuration(
                vector_store,
                bm25_retriever,
                expanded_query,
                expected_evidence,
                "faiss",
            )
        )

        # ====================================================
        # C. Original + Hybrid
        # ====================================================

        hybrid_original = (
            run_configuration(
                vector_store,
                bm25_retriever,
                original_query,
                expected_evidence,
                "hybrid",
            )
        )

        # ====================================================
        # D. Expanded + Hybrid
        # ====================================================

        hybrid_expanded = (
            run_configuration(
                vector_store,
                bm25_retriever,
                expanded_query,
                expected_evidence,
                "hybrid",
            )
        )

        print("\n" + "-" * 96)

        print_result(
            "A. ORIGINAL + FAISS",
            faiss_original,
        )

        print("\n" + "-" * 96)

        print_result(
            "B. EXPANDED + FAISS",
            faiss_expanded,
        )

        print("\n" + "-" * 96)

        print_result(
            "C. ORIGINAL + HYBRID",
            hybrid_original,
        )

        print("\n" + "-" * 96)

        print_result(
            "D. EXPANDED + HYBRID",
            hybrid_expanded,
        )

        experiment_results.append(
            {
                "id": question_id,
                "question": original_query,
                "expanded_query": (
                    expanded_query
                ),
                "expected_evidence": (
                    expected_evidence
                ),
                "faiss_original": (
                    faiss_original
                ),
                "faiss_expanded": (
                    faiss_expanded
                ),
                "hybrid_original": (
                    hybrid_original
                ),
                "hybrid_expanded": (
                    hybrid_expanded
                ),
            }
        )

    # ========================================================
    # Final Comparison
    # ========================================================

    print("\n" + "=" * 96)
    print("FINAL COMPARISON")
    print("=" * 96)

    header = (
        f"{'ID':<6}"
        f"{'FAISS-O':<14}"
        f"{'FAISS-E':<14}"
        f"{'HYBRID-O':<14}"
        f"{'HYBRID-E':<14}"
    )

    print("\n" + header)
    print("-" * 62)

    for result in experiment_results:

        fo = (
            result["faiss_original"][
                "coverage"
            ]
        )

        fe = (
            result["faiss_expanded"][
                "coverage"
            ]
        )

        ho = (
            result["hybrid_original"][
                "coverage"
            ]
        )

        he = (
            result["hybrid_expanded"][
                "coverage"
            ]
        )

        print(
            f"{result['id']:<6}"
            f"{fo:<14.3f}"
            f"{fe:<14.3f}"
            f"{ho:<14.3f}"
            f"{he:<14.3f}"
        )

    # ========================================================
    # Average Coverage
    # ========================================================

    methods = [
        "faiss_original",
        "faiss_expanded",
        "hybrid_original",
        "hybrid_expanded",
    ]

    averages = {}

    for method in methods:

        values = [
            result[method]["coverage"]
            for result
            in experiment_results
        ]

        averages[method] = (
            sum(values) / len(values)
        )

    print("\nAverage Coverage:")

    for method, value in averages.items():

        print(
            f"{method:<20} : "
            f"{value:.3f}"
        )

    best_method = max(
        averages,
        key=averages.get,
    )

    print(
        f"\n🏆 Best configuration: "
        f"{best_method}"
    )

    print(
        f"   Mean coverage: "
        f"{averages[best_method]:.3f}"
    )

    # ========================================================
    # Save Results
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": (
            "hybrid_retrieval"
        ),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "averages": averages,
        "best_method": best_method,
        "results": experiment_results,
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
        "\n✅ Hybrid retrieval "
        "experiment complete."
    )


if __name__ == "__main__":
    main()