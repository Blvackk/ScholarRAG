

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

OUTPUT_PATH = RESULTS_DIR / "weighted_rrf_experiment.json"

TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60

# FAISS weight varies; BM25 remains fixed at 1.0.
WEIGHT_CONFIGS = [
    (1.00, 1.00),
    (1.25, 1.00),
    (1.50, 1.00),
    (1.75, 1.00),
    (2.00, 1.00),
]


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
# PDFs
# ============================================================

def load_papers():
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

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

    print(f"✅ Loaded {len(documents)} pages.")

    return documents


# ============================================================
# Stable Document Identity
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

        ranked_indices = np.argsort(scores)[::-1]

        return [
            self.documents[int(index)]
            for index in ranked_indices[:k]
        ]


# ============================================================
# Weighted Reciprocal Rank Fusion
# ============================================================

def weighted_rrf(
    faiss_documents,
    bm25_documents,
    faiss_weight,
    bm25_weight,
):
    scores = {}
    documents = {}

    # Dense / FAISS
    for rank, document in enumerate(
        faiss_documents,
        start=1,
    ):
        key = document_key(document)

        documents[key] = document

        scores[key] = (
            scores.get(key, 0.0)
            + faiss_weight / (RRF_K + rank)
        )

    # Lexical / BM25
    for rank, document in enumerate(
        bm25_documents,
        start=1,
    ):
        key = document_key(document)

        documents[key] = document

        scores[key] = (
            scores.get(key, 0.0)
            + bm25_weight / (RRF_K + rank)
        )

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        documents[key]
        for key in ranked_keys[:TOP_K]
    ]


# ============================================================
# Evidence Evaluation
# ============================================================

def evaluate_documents(
    documents,
    expected_evidence,
):
    matched = []
    missing = []

    first_rank = None
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

        evidence_ranks[evidence] = found_rank

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
# Aggregate
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
        "\n⚖️ Starting ScholarRAG "
        "Weighted RRF Experiment"
    )

    dataset = load_dataset()

    answerable = [
        item
        for item in dataset
        if (
            item.get("category") != "unanswerable"
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
    # Load corpus
    # --------------------------------------------------------

    documents = load_papers()

    print("\n✂️ Creating chunks...")

    chunks = split_documents(documents)

    print(f"✅ Created {len(chunks)} chunks.")

    # --------------------------------------------------------
    # Embeddings / FAISS
    # --------------------------------------------------------

    print("\n🧠 Loading embeddings...")

    embeddings = get_embeddings()

    print("✅ Embeddings ready.")

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
    # Cache retrieval candidates
    #
    # Important: retrieval itself does not change between
    # weight configurations, so retrieve once per question.
    # --------------------------------------------------------

    print("\n📦 Retrieving candidate sets...")

    candidate_cache = {}

    for item in answerable:

        question = item["question"]

        faiss_documents = (
            vector_store.similarity_search(
                question,
                k=CANDIDATE_K,
            )
        )

        bm25_documents = bm25.search(
            question,
            CANDIDATE_K,
        )

        candidate_cache[item["id"]] = {
            "faiss": faiss_documents,
            "bm25": bm25_documents,
        }

    print("✅ Candidate sets ready.")

    # --------------------------------------------------------
    # Experiment
    # --------------------------------------------------------

    experiment_results = []

    print("\n" + "=" * 92)

    print(
        "SCHOLARRAG WEIGHTED RRF EXPERIMENT"
    )

    print("=" * 92)

    for faiss_weight, bm25_weight in WEIGHT_CONFIGS:

        print(
            f"\n⚖️ FAISS={faiss_weight:.2f} "
            f"| BM25={bm25_weight:.2f}"
        )

        question_results = []
        metric_results = []

        for item in answerable:

            cached = candidate_cache[item["id"]]

            fused_documents = weighted_rrf(
                cached["faiss"],
                cached["bm25"],
                faiss_weight,
                bm25_weight,
            )

            result = evaluate_documents(
                fused_documents,
                item["expected_evidence"],
            )

            metric_results.append(result)

            question_results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "category": item.get(
                        "category"
                    ),
                    "coverage": result["coverage"],
                    "hit": result["hit"],
                    "full": result["full"],
                    "rr": result["rr"],
                    "matched": result["matched"],
                    "missing": result["missing"],
                    "evidence_ranks": (
                        result["evidence_ranks"]
                    ),
                }
            )

            status = (
                "✅"
                if result["full"]
                else "⚠️"
            )

            print(
                f"   {item['id']}: "
                f"{result['coverage']:.3f} "
                f"{status}"
            )

        metrics = aggregate(
            metric_results
        )

        experiment_results.append(
            {
                "faiss_weight": faiss_weight,
                "bm25_weight": bm25_weight,
                "metrics": metrics,
                "questions": question_results,
            }
        )

        print(
            f"\n   Hit@5     : "
            f"{metrics['hit_at_5']:.3f}"
        )

        print(
            f"   Full@5    : "
            f"{metrics['full_at_5']:.3f}"
        )

        print(
            f"   Coverage  : "
            f"{metrics['coverage']:.3f}"
        )

        print(
            f"   MRR       : "
            f"{metrics['mrr']:.3f}"
        )

    # ========================================================
    # Comparison Table
    # ========================================================

    print("\n" + "=" * 92)
    print("FINAL WEIGHT COMPARISON")
    print("=" * 92)

    print(
        f"\n{'FAISS':<10}"
        f"{'BM25':<10}"
        f"{'Hit@5':<12}"
        f"{'Full@5':<12}"
        f"{'Coverage':<14}"
        f"{'MRR':<12}"
    )

    print("-" * 70)

    for experiment in experiment_results:

        metrics = experiment["metrics"]

        print(
            f"{experiment['faiss_weight']:<10.2f}"
            f"{experiment['bm25_weight']:<10.2f}"
            f"{metrics['hit_at_5']:<12.3f}"
            f"{metrics['full_at_5']:<12.3f}"
            f"{metrics['coverage']:<14.3f}"
            f"{metrics['mrr']:<12.3f}"
        )

    # ========================================================
    # Select Best Configuration
    #
    # Priority:
    # 1. Full evidence
    # 2. Coverage
    # 3. Hit rate
    # 4. MRR
    #
    # Full evidence matters more than simply retrieving one
    # evidence phrase.
    # ========================================================

    best = max(
        experiment_results,
        key=lambda experiment: (
            experiment["metrics"]["full_at_5"],
            experiment["metrics"]["coverage"],
            experiment["metrics"]["hit_at_5"],
            experiment["metrics"]["mrr"],
        ),
    )

    print("\n" + "=" * 92)
    print("BEST CONFIGURATION")
    print("=" * 92)

    print(
        f"\nFAISS weight : "
        f"{best['faiss_weight']:.2f}"
    )

    print(
        f"BM25 weight  : "
        f"{best['bm25_weight']:.2f}"
    )

    print(
        f"Hit@5        : "
        f"{best['metrics']['hit_at_5']:.3f}"
    )

    print(
        f"Full@5       : "
        f"{best['metrics']['full_at_5']:.3f}"
    )

    print(
        f"Coverage     : "
        f"{best['metrics']['coverage']:.3f}"
    )

    print(
        f"MRR          : "
        f"{best['metrics']['mrr']:.3f}"
    )

    # --------------------------------------------------------
    # Important questions
    # --------------------------------------------------------

    print("\nCritical Questions:")

    for target_id in [
        "q04",
        "q08",
        "q10",
    ]:

        question_result = next(
            result
            for result in best["questions"]
            if result["id"] == target_id
        )

        print(
            f"  {target_id}: "
            f"coverage="
            f"{question_result['coverage']:.3f}"
            f" | full="
            f"{question_result['full']}"
        )

    # ========================================================
    # Save
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": "weighted_rrf",
        "configuration": {
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "weight_configs": WEIGHT_CONFIGS,
            "questions": len(answerable),
            "chunks": len(chunks),
        },
        "experiments": experiment_results,
        "best_configuration": {
            "faiss_weight": (
                best["faiss_weight"]
            ),
            "bm25_weight": (
                best["bm25_weight"]
            ),
            "metrics": best["metrics"],
        },
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
        "\n✅ Weighted RRF experiment complete."
    )


if __name__ == "__main__":
    main()