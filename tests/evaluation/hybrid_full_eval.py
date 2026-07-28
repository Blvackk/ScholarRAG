#hybrid_full_eval.py

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

OUTPUT_PATH = RESULTS_DIR / "hybrid_full_eval.json"

TOP_K = 5

# Number of candidates retrieved by each retriever before fusion
CANDIDATE_K = 10

# Reciprocal Rank Fusion constant
RRF_K = 60


# ============================================================
# Text Helpers
# ============================================================

def normalize_text(text):
    """
    Normalize text so evidence matching is more robust.
    """

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
    """
    Basic tokenizer for BM25.
    """

    return normalize_text(text).split()


def evidence_in_text(text, evidence):
    """
    Check whether expected evidence exists inside a chunk.
    """

    return (
        normalize_text(evidence)
        in normalize_text(text)
    )


# ============================================================
# Dataset
# ============================================================

def load_dataset():
    """
    Load evaluation dataset.
    """

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# PDF Loading
# ============================================================

def load_papers():
    """
    Load every evaluation PDF.
    """

    pdf_files = sorted(
        PAPERS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No evaluation PDFs found in:\n{PAPERS_DIR}"
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
# Stable Document Identity
# ============================================================

def document_key(document):
    """
    Create a stable identity for the same chunk across
    FAISS and BM25 retrieval results.
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
                self.documents[int(index)]
            )

        return results


# ============================================================
# FAISS Retrieval
# ============================================================

def faiss_search(
    vector_store,
    query,
    k,
):
    """
    Dense similarity retrieval.
    """

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

    RRF score:
        1 / (RRF_K + rank)

    A chunk appearing in both rankings receives contributions
    from both retrievers.
    """

    scores = {}
    documents_by_key = {}

    # FAISS ranking
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

    # BM25 ranking
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

    return [
        documents_by_key[key]
        for key in ranked_keys[:top_k]
    ]


# ============================================================
# Hybrid Retrieval
# ============================================================

def hybrid_search(
    vector_store,
    bm25_retriever,
    query,
):
    """
    Retrieve candidates using FAISS and BM25,
    then combine them using RRF.
    """

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
# Evidence Evaluation
# ============================================================

def evaluate_documents(
    documents,
    expected_evidence,
):
    """
    Evaluate evidence retrieval quality.

    Metrics:
        Hit@K
        Full Evidence@K
        Evidence Coverage
        Evidence Reciprocal Rank
    """

    matched = []
    missing = []

    evidence_ranks = {}

    first_evidence_rank = None

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

        evidence_ranks[evidence] = found_rank

        if found_rank is None:
            missing.append(evidence)

        else:
            matched.append(evidence)

            if (
                first_evidence_rank is None
                or found_rank < first_evidence_rank
            ):
                first_evidence_rank = found_rank

    total_evidence = len(
        expected_evidence
    )

    coverage = (
        len(matched) / total_evidence
        if total_evidence
        else 0.0
    )

    hit = len(matched) > 0

    full = (
        total_evidence > 0
        and len(missing) == 0
    )

    reciprocal_rank = (
        1.0 / first_evidence_rank
        if first_evidence_rank
        else 0.0
    )

    pages = []

    for document in documents:

        page = document.metadata.get(
            "page"
        )

        if page is not None:

            try:
                # LangChain page metadata is normally zero-based
                page = int(page) + 1

            except (TypeError, ValueError):
                pass

        pages.append(page)

    return {
        "hit": hit,
        "full_evidence": full,
        "coverage": coverage,
        "reciprocal_rank": reciprocal_rank,
        "matched_evidence": matched,
        "missing_evidence": missing,
        "evidence_ranks": evidence_ranks,
        "retrieved_pages": pages,
    }


# ============================================================
# Aggregate Metrics
# ============================================================

def aggregate_metrics(results):
    """
    Calculate dataset-level retrieval metrics.
    """

    if not results:

        return {
            "questions": 0,
            "hit_at_k": 0.0,
            "full_evidence_at_k": 0.0,
            "mean_coverage": 0.0,
            "mrr": 0.0,
        }

    count = len(results)

    hit_at_k = sum(
        result["hit"]
        for result in results
    ) / count

    full_at_k = sum(
        result["full_evidence"]
        for result in results
    ) / count

    mean_coverage = sum(
        result["coverage"]
        for result in results
    ) / count

    mrr = sum(
        result["reciprocal_rank"]
        for result in results
    ) / count

    return {
        "questions": count,
        "hit_at_k": hit_at_k,
        "full_evidence_at_k": full_at_k,
        "mean_coverage": mean_coverage,
        "mrr": mrr,
    }


# ============================================================
# Per-Question Comparison
# ============================================================

def classify_change(
    faiss_result,
    hybrid_result,
):
    """
    Determine whether hybrid improved, regressed,
    or produced no change in evidence coverage.
    """

    delta = (
        hybrid_result["coverage"]
        - faiss_result["coverage"]
    )

    if delta > 0:
        return "improved"

    if delta < 0:
        return "regressed"

    return "unchanged"


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n🧪 Starting ScholarRAG "
        "Full Hybrid Retrieval Evaluation"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = load_dataset()

    answerable_questions = [
        item
        for item in dataset
        if (
            item.get("category")
            != "unanswerable"
            and item.get("expected_evidence")
        )
    ]

    print(
        f"📋 Loaded {len(dataset)} total questions."
    )

    print(
        f"🎯 Evaluating "
        f"{len(answerable_questions)} "
        f"answerable questions."
    )

    # --------------------------------------------------------
    # PDFs
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
    # Embeddings
    # --------------------------------------------------------

    print("\n🧠 Loading embeddings...")

    embeddings = get_embeddings()

    print("✅ Embeddings ready.")

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

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
    # Evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 100)

    print(
        "SCHOLARRAG FULL RETRIEVAL BENCHMARK"
    )

    print("=" * 100)

    faiss_results = []
    hybrid_results = []

    question_results = []

    for item in answerable_questions:

        question_id = item["id"]

        question = item["question"]

        expected_evidence = (
            item["expected_evidence"]
        )

        # ====================================================
        # FAISS
        # ====================================================

        faiss_documents = faiss_search(
            vector_store,
            question,
            TOP_K,
        )

        faiss_result = (
            evaluate_documents(
                faiss_documents,
                expected_evidence,
            )
        )

        # ====================================================
        # Hybrid
        # ====================================================

        hybrid_documents = hybrid_search(
            vector_store,
            bm25_retriever,
            question,
        )

        hybrid_result = (
            evaluate_documents(
                hybrid_documents,
                expected_evidence,
            )
        )

        faiss_results.append(
            faiss_result
        )

        hybrid_results.append(
            hybrid_result
        )

        # ====================================================
        # Delta
        # ====================================================

        coverage_delta = (
            hybrid_result["coverage"]
            - faiss_result["coverage"]
        )

        change = classify_change(
            faiss_result,
            hybrid_result,
        )

        question_results.append(
            {
                "id": question_id,
                "question": question,
                "category": item.get(
                    "category"
                ),
                "expected_evidence": (
                    expected_evidence
                ),
                "faiss": faiss_result,
                "hybrid": hybrid_result,
                "coverage_delta": (
                    coverage_delta
                ),
                "change": change,
            }
        )

        # ====================================================
        # Console Output
        # ====================================================

        print(
            f"\n[{question_id}] "
            f"{question}"
        )

        print(
            f"FAISS  → "
            f"Hit: "
            f"{'✅' if faiss_result['hit'] else '❌'} "
            f"| Full: "
            f"{'✅' if faiss_result['full_evidence'] else '❌'} "
            f"| Coverage: "
            f"{faiss_result['coverage']:.3f} "
            f"| RR: "
            f"{faiss_result['reciprocal_rank']:.3f}"
        )

        print(
            f"HYBRID → "
            f"Hit: "
            f"{'✅' if hybrid_result['hit'] else '❌'} "
            f"| Full: "
            f"{'✅' if hybrid_result['full_evidence'] else '❌'} "
            f"| Coverage: "
            f"{hybrid_result['coverage']:.3f} "
            f"| RR: "
            f"{hybrid_result['reciprocal_rank']:.3f}"
        )

        if change == "improved":

            print(
                f"📈 Improvement: "
                f"+{coverage_delta:.3f}"
            )

        elif change == "regressed":

            print(
                f"📉 Regression: "
                f"{coverage_delta:.3f}"
            )

        else:

            print(
                "➖ No coverage change"
            )

    # ========================================================
    # Aggregate Results
    # ========================================================

    faiss_metrics = aggregate_metrics(
        faiss_results
    )

    hybrid_metrics = aggregate_metrics(
        hybrid_results
    )

    # ========================================================
    # Count Improvements / Regressions
    # ========================================================

    improved = sum(
        result["change"] == "improved"
        for result in question_results
    )

    regressed = sum(
        result["change"] == "regressed"
        for result in question_results
    )

    unchanged = sum(
        result["change"] == "unchanged"
        for result in question_results
    )

    # ========================================================
    # Final Table
    # ========================================================

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print(
        f"\nTop K                 : {TOP_K}"
    )

    print(
        f"Candidate K           : {CANDIDATE_K}"
    )

    print(
        f"Answerable Questions  : "
        f"{len(answerable_questions)}"
    )

    print("\n" + "-" * 76)

    print(
        f"{'Metric':<28}"
        f"{'FAISS':<18}"
        f"{'HYBRID':<18}"
        f"{'Delta':<12}"
    )

    print("-" * 76)

    metrics_to_print = [
        (
            f"Evidence Hit@{TOP_K}",
            "hit_at_k",
        ),
        (
            f"Full Evidence@{TOP_K}",
            "full_evidence_at_k",
        ),
        (
            "Mean Evidence Coverage",
            "mean_coverage",
        ),
        (
            "Evidence MRR",
            "mrr",
        ),
    ]

    for label, key in metrics_to_print:

        faiss_value = (
            faiss_metrics[key]
        )

        hybrid_value = (
            hybrid_metrics[key]
        )

        delta = (
            hybrid_value
            - faiss_value
        )

        print(
            f"{label:<28}"
            f"{faiss_value:<18.3f}"
            f"{hybrid_value:<18.3f}"
            f"{delta:+.3f}"
        )

    # ========================================================
    # Per-Question Coverage
    # ========================================================

    print("\n" + "=" * 100)
    print("PER-QUESTION COVERAGE")
    print("=" * 100)

    print(
        f"\n{'ID':<7}"
        f"{'FAISS':<14}"
        f"{'HYBRID':<14}"
        f"{'DELTA':<14}"
        f"{'STATUS':<14}"
    )

    print("-" * 63)

    for result in question_results:

        faiss_coverage = (
            result["faiss"]["coverage"]
        )

        hybrid_coverage = (
            result["hybrid"]["coverage"]
        )

        delta = (
            result["coverage_delta"]
        )

        change = result["change"]

        if change == "improved":
            status = "📈 improved"

        elif change == "regressed":
            status = "📉 regressed"

        else:
            status = "➖ unchanged"

        print(
            f"{result['id']:<7}"
            f"{faiss_coverage:<14.3f}"
            f"{hybrid_coverage:<14.3f}"
            f"{delta:<+14.3f}"
            f"{status}"
        )

    # ========================================================
    # Regression Summary
    # ========================================================

    print("\n" + "=" * 100)
    print("CHANGE SUMMARY")
    print("=" * 100)

    print(
        f"Improved questions : {improved}"
    )

    print(
        f"Regressed questions: {regressed}"
    )

    print(
        f"Unchanged questions: {unchanged}"
    )

    # ========================================================
    # Recommendation
    # ========================================================

    coverage_gain = (
        hybrid_metrics["mean_coverage"]
        - faiss_metrics["mean_coverage"]
    )

    full_gain = (
        hybrid_metrics["full_evidence_at_k"]
        - faiss_metrics["full_evidence_at_k"]
    )

    hit_gain = (
        hybrid_metrics["hit_at_k"]
        - faiss_metrics["hit_at_k"]
    )

    if (
        coverage_gain > 0
        and full_gain >= 0
        and hit_gain >= 0
        and regressed == 0
    ):

        recommendation = (
            "HYBRID_RECOMMENDED"
        )

        recommendation_text = (
            "Hybrid retrieval improves overall evidence "
            "coverage without per-question coverage regressions."
        )

    elif (
        coverage_gain > 0
        and regressed > 0
    ):

        recommendation = (
            "HYBRID_PROMISING_WITH_REGRESSIONS"
        )

        recommendation_text = (
            "Hybrid retrieval improves aggregate coverage, "
            "but some questions regress. Investigate before "
            "production integration."
        )

    elif coverage_gain == 0:

        recommendation = (
            "NO_CLEAR_ADVANTAGE"
        )

        recommendation_text = (
            "Hybrid retrieval does not improve mean evidence "
            "coverage over FAISS."
        )

    else:

        recommendation = (
            "KEEP_FAISS_BASELINE"
        )

        recommendation_text = (
            "Hybrid retrieval performs worse than the "
            "FAISS baseline on mean evidence coverage."
        )

    print("\nDecision:")

    print(
        f"  {recommendation}"
    )

    print(
        f"  {recommendation_text}"
    )

    # ========================================================
    # Save JSON
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": (
            "full_hybrid_retrieval_evaluation"
        ),
        "configuration": {
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "answerable_questions": (
                len(answerable_questions)
            ),
            "chunks": len(chunks),
        },
        "faiss_metrics": (
            faiss_metrics
        ),
        "hybrid_metrics": (
            hybrid_metrics
        ),
        "metric_deltas": {
            "hit_at_k": (
                hybrid_metrics["hit_at_k"]
                - faiss_metrics["hit_at_k"]
            ),
            "full_evidence_at_k": (
                hybrid_metrics["full_evidence_at_k"]
                - faiss_metrics["full_evidence_at_k"]
            ),
            "mean_coverage": (
                hybrid_metrics["mean_coverage"]
                - faiss_metrics["mean_coverage"]
            ),
            "mrr": (
                hybrid_metrics["mrr"]
                - faiss_metrics["mrr"]
            ),
        },
        "change_summary": {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        },
        "recommendation": {
            "decision": recommendation,
            "reason": recommendation_text,
        },
        "questions": (
            question_results
        ),
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

    print(
        OUTPUT_PATH
    )

    print(
        "\n✅ Full hybrid retrieval "
        "evaluation complete."
    )


if __name__ == "__main__":
    main()