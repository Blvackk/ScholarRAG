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

OUTPUT_PATH = RESULTS_DIR / "preservation_experiment.json"

TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60


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
    return normalize_text(evidence) in normalize_text(text)


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
# PDF Loading
# ============================================================

def load_papers():

    pdf_files = sorted(
        PAPERS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in:\n{PAPERS_DIR}"
        )

    documents = []

    print("\n📄 Loading evaluation papers...")

    for pdf_path in pdf_files:

        print(f"   Loading: {pdf_path.name}")

        documents.extend(
            load_pdf(str(pdf_path))
        )

    print(
        f"✅ Loaded {len(documents)} pages."
    )

    return documents


# ============================================================
# Document Identity
# ============================================================

def document_key(document):

    return (
        document.metadata.get("source", ""),
        document.metadata.get("page", ""),
        normalize_text(document.page_content),
    )


# ============================================================
# BM25
# ============================================================

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


# ============================================================
# RRF
# ============================================================

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
            + 1.0 / (RRF_K + rank)
        )

    for rank, document in enumerate(
        bm25_documents,
        start=1,
    ):

        key = document_key(document)

        documents[key] = document

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
        documents[key]
        for key in ranked_keys
    ]


# ============================================================
# Preservation Strategy
# ============================================================

def preserve_faiss_top1(
    faiss_documents,
    fused_documents,
):
    """
    Preserve the highest-ranked FAISS result, then fill
    the remaining TOP_K - 1 positions using RRF results.

    Duplicate chunks are removed.
    """

    if not faiss_documents:
        return fused_documents[:TOP_K]

    preserved = faiss_documents[0]

    final_documents = [preserved]

    preserved_key = document_key(
        preserved
    )

    for document in fused_documents:

        if document_key(document) == preserved_key:
            continue

        final_documents.append(document)

        if len(final_documents) == TOP_K:
            break

    return final_documents


# ============================================================
# Evaluation
# ============================================================

def evaluate_documents(
    documents,
    expected_evidence,
):

    matched = []
    missing = []
    evidence_ranks = {}

    first_rank = None

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

            if (
                first_rank is None
                or found_rank < first_rank
            ):
                first_rank = found_rank

    total = len(expected_evidence)

    coverage = (
        len(matched) / total
        if total
        else 0.0
    )

    hit = len(matched) > 0

    full = (
        total > 0
        and len(missing) == 0
    )

    rr = (
        1.0 / first_rank
        if first_rank
        else 0.0
    )

    return {
        "hit": hit,
        "full": full,
        "coverage": coverage,
        "rr": rr,
        "matched": matched,
        "missing": missing,
        "evidence_ranks": evidence_ranks,
    }


# ============================================================
# Aggregate Metrics
# ============================================================

def aggregate(results):

    count = len(results)

    if count == 0:
        return {
            "hit_at_5": 0.0,
            "full_at_5": 0.0,
            "coverage": 0.0,
            "mrr": 0.0,
        }

    return {
        "hit_at_5": sum(
            result["hit"]
            for result in results
        ) / count,

        "full_at_5": sum(
            result["full"]
            for result in results
        ) / count,

        "coverage": sum(
            result["coverage"]
            for result in results
        ) / count,

        "mrr": sum(
            result["rr"]
            for result in results
        ) / count,
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n🧪 Starting ScholarRAG "
        "Candidate Preservation Experiment"
    )

    dataset = load_dataset()

    answerable = [
        item
        for item in dataset
        if (
            item.get("category")
            != "unanswerable"
            and item.get("expected_evidence")
        )
    ]

    print(
        f"📋 Loaded {len(dataset)} questions."
    )

    print(
        f"🎯 Evaluating {len(answerable)} "
        f"answerable questions."
    )

    # --------------------------------------------------------
    # Corpus
    # --------------------------------------------------------

    documents = load_papers()

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

    print("\n🔎 Building FAISS index...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings,
    )

    print("✅ FAISS ready.")

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print("\n🔤 Building BM25 index...")

    bm25 = BM25Retriever(chunks)

    print("✅ BM25 ready.")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    faiss_results = []
    rrf_results = []
    preserved_results = []

    per_question = []

    print("\n" + "=" * 100)

    print(
        "SCHOLARRAG CANDIDATE "
        "PRESERVATION EXPERIMENT"
    )

    print("=" * 100)

    for item in answerable:

        question = item["question"]

        expected = item[
            "expected_evidence"
        ]

        # ====================================================
        # Candidate retrieval
        # ====================================================

        faiss_candidates = (
            vector_store.similarity_search(
                question,
                k=CANDIDATE_K,
            )
        )

        bm25_candidates = bm25.search(
            question,
            CANDIDATE_K,
        )

        # ====================================================
        # Strategy A: FAISS Top-5
        # ====================================================

        faiss_top5 = (
            faiss_candidates[:TOP_K]
        )

        faiss_eval = evaluate_documents(
            faiss_top5,
            expected,
        )

        # ====================================================
        # Strategy B: Equal RRF Top-5
        # ====================================================

        fused = reciprocal_rank_fusion(
            faiss_candidates,
            bm25_candidates,
        )

        rrf_top5 = fused[:TOP_K]

        rrf_eval = evaluate_documents(
            rrf_top5,
            expected,
        )

        # ====================================================
        # Strategy C: FAISS #1 + RRF Top-4
        # ====================================================

        preserved_top5 = preserve_faiss_top1(
            faiss_candidates,
            fused,
        )

        preserved_eval = evaluate_documents(
            preserved_top5,
            expected,
        )

        faiss_results.append(
            faiss_eval
        )

        rrf_results.append(
            rrf_eval
        )

        preserved_results.append(
            preserved_eval
        )

        per_question.append(
            {
                "id": item["id"],
                "question": question,
                "expected_evidence": expected,
                "faiss": faiss_eval,
                "rrf": rrf_eval,
                "preserved": preserved_eval,
            }
        )

        print(
            f"\n[{item['id']}] "
            f"{question}"
        )

        print(
            f"FAISS     → "
            f"{faiss_eval['coverage']:.3f} "
            f"| Full: "
            f"{'✅' if faiss_eval['full'] else '❌'}"
        )

        print(
            f"RRF       → "
            f"{rrf_eval['coverage']:.3f} "
            f"| Full: "
            f"{'✅' if rrf_eval['full'] else '❌'}"
        )

        print(
            f"PRESERVED → "
            f"{preserved_eval['coverage']:.3f} "
            f"| Full: "
            f"{'✅' if preserved_eval['full'] else '❌'}"
        )

        if item["id"] in {
            "q04",
            "q08",
            "q10",
        }:

            print(
                "  Critical question"
            )

            if preserved_eval["missing"]:

                print(
                    "  Missing after preservation: "
                    f"{preserved_eval['missing']}"
                )

    # ========================================================
    # Aggregate
    # ========================================================

    faiss_metrics = aggregate(
        faiss_results
    )

    rrf_metrics = aggregate(
        rrf_results
    )

    preserved_metrics = aggregate(
        preserved_results
    )

    # ========================================================
    # Final Comparison
    # ========================================================

    print("\n" + "=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)

    print(
        f"\n{'Metric':<28}"
        f"{'FAISS':<18}"
        f"{'RRF':<18}"
        f"{'PRESERVED':<18}"
    )

    print("-" * 82)

    rows = [
        (
            "Evidence Hit@5",
            "hit_at_5",
        ),
        (
            "Full Evidence@5",
            "full_at_5",
        ),
        (
            "Mean Coverage",
            "coverage",
        ),
        (
            "MRR",
            "mrr",
        ),
    ]

    for label, key in rows:

        print(
            f"{label:<28}"
            f"{faiss_metrics[key]:<18.3f}"
            f"{rrf_metrics[key]:<18.3f}"
            f"{preserved_metrics[key]:<18.3f}"
        )

    # ========================================================
    # Critical Question Comparison
    # ========================================================

    print("\n" + "=" * 100)
    print("CRITICAL QUESTIONS")
    print("=" * 100)

    print(
        f"\n{'ID':<8}"
        f"{'FAISS':<16}"
        f"{'RRF':<16}"
        f"{'PRESERVED':<16}"
    )

    print("-" * 56)

    for result in per_question:

        if result["id"] not in {
            "q04",
            "q08",
            "q10",
        }:
            continue

        print(
            f"{result['id']:<8}"
            f"{result['faiss']['coverage']:<16.3f}"
            f"{result['rrf']['coverage']:<16.3f}"
            f"{result['preserved']['coverage']:<16.3f}"
        )

    # ========================================================
    # Regression Check vs Equal RRF
    # ========================================================

    improvements = 0
    regressions = 0
    unchanged = 0

    for result in per_question:

        rrf_coverage = (
            result["rrf"]["coverage"]
        )

        preserved_coverage = (
            result["preserved"]["coverage"]
        )

        if preserved_coverage > rrf_coverage:
            improvements += 1

        elif preserved_coverage < rrf_coverage:
            regressions += 1

        else:
            unchanged += 1

    print("\n" + "=" * 100)
    print("PRESERVATION VS EQUAL RRF")
    print("=" * 100)

    print(
        f"\nImproved   : {improvements}"
    )

    print(
        f"Regressed  : {regressions}"
    )

    print(
        f"Unchanged  : {unchanged}"
    )

    # ========================================================
    # Decision
    # ========================================================

    if (
        preserved_metrics["coverage"]
        > rrf_metrics["coverage"]
        and regressions == 0
    ):

        decision = (
            "PRESERVATION_RECOMMENDED"
        )

        reason = (
            "Preserving FAISS rank #1 improves "
            "coverage without regressions."
        )

    elif (
        preserved_metrics["coverage"]
        == rrf_metrics["coverage"]
        and regressions == 0
    ):

        decision = (
            "NO_MEANINGFUL_ADVANTAGE"
        )

        reason = (
            "Preservation does not improve the "
            "current equal-RRF benchmark."
        )

    else:

        decision = (
            "KEEP_EQUAL_RRF"
        )

        reason = (
            "Candidate preservation introduces "
            "regressions or performs worse than "
            "equal RRF."
        )

    print("\nDecision:")

    print(f"  {decision}")
    print(f"  {reason}")

    # ========================================================
    # Save Results
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": (
            "candidate_preservation"
        ),
        "configuration": {
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "preservation": (
                "FAISS rank 1 + "
                "RRF remaining slots"
            ),
            "questions": len(answerable),
            "chunks": len(chunks),
        },
        "metrics": {
            "faiss": faiss_metrics,
            "equal_rrf": rrf_metrics,
            "preserved": preserved_metrics,
        },
        "comparison_vs_rrf": {
            "improved": improvements,
            "regressed": regressions,
            "unchanged": unchanged,
        },
        "decision": {
            "result": decision,
            "reason": reason,
        },
        "questions": per_question,
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

    print("\n💾 Results saved to:")
    print(OUTPUT_PATH)

    print(
        "\n✅ Candidate preservation "
        "experiment complete."
    )


if __name__ == "__main__":
    main()