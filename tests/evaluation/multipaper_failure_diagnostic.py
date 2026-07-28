import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.embeddings import get_embeddings


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "tests" / "evaluation" / "dataset.json"
PAPERS_DIR = BASE_DIR / "tests" / "evaluation" / "papers"
RESULTS_DIR = BASE_DIR / "tests" / "evaluation" / "results"

OUTPUT_PATH = RESULTS_DIR / "multipaper_failure_diagnostic.json"

CHUNK_SIZE = 2048
CHUNK_OVERLAP = 512

# Retrieve much deeper than production Top-5.
DIAGNOSTIC_K = 20
RRF_K = 60


# ============================================================
# TARGET QUESTIONS
# ============================================================

TARGETS = {
    # Hybrid regressions
    "p01_q08": "hybrid_regression",
    "p04_q07": "hybrid_regression",

    # Failed across retrievers
    "p03_q06": "retrieval_failure",
    "p03_q08": "retrieval_failure",

    # Hybrid successes for comparison
    "p01_q04": "hybrid_success",
    "p02_q10": "hybrid_success",
    "p03_q09": "hybrid_success",
}


# ============================================================
# NORMALISATION
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = text.lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def compact_text(text):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        normalize_text(text)
    )


def evidence_in_text(evidence, text):
    evidence_norm = normalize_text(evidence)
    text_norm = normalize_text(text)

    if evidence_norm in text_norm:
        return True

    evidence_compact = compact_text(evidence)
    text_compact = compact_text(text)

    return (
        bool(evidence_compact)
        and evidence_compact in text_compact
    )


# ============================================================
# DATASET
# ============================================================

def load_dataset():
    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# ============================================================
# PDF + CHUNKING
# ============================================================

def load_paper(paper_name):
    path = PAPERS_DIR / paper_name

    if not path.exists():
        raise FileNotFoundError(
            f"Paper not found: {path}"
        )

    loader = PyPDFLoader(str(path))
    pages = loader.load()

    for page_number, page in enumerate(
        pages,
        start=1
    ):
        page.metadata["paper"] = paper_name
        page.metadata["page_number"] = page_number

    return pages


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


# ============================================================
# BM25
# ============================================================

def tokenize(text):
    return re.findall(
        r"[a-z0-9]+",
        normalize_text(text)
    )


def build_bm25(chunks):
    corpus = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    return BM25Okapi(corpus)


def bm25_search(query, bm25, chunks, k):
    scores = bm25.get_scores(
        tokenize(query)
    )

    ranked = np.argsort(scores)[::-1]

    return [
        chunks[int(index)]
        for index in ranked[:k]
    ]


# ============================================================
# DOCUMENT IDENTITY
# ============================================================

def doc_key(doc):
    return (
        doc.metadata.get("paper"),
        doc.metadata.get("chunk_id"),
    )


# ============================================================
# RRF
# ============================================================

def rrf_search(
    faiss_docs,
    bm25_docs,
    top_k
):
    scores = defaultdict(float)
    docs = {}

    for rank, doc in enumerate(
        faiss_docs,
        start=1
    ):
        key = doc_key(doc)

        docs[key] = doc

        scores[key] += (
            1.0 / (RRF_K + rank)
        )

    for rank, doc in enumerate(
        bm25_docs,
        start=1
    ):
        key = doc_key(doc)

        docs[key] = doc

        scores[key] += (
            1.0 / (RRF_K + rank)
        )

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        docs[key]
        for key in ranked_keys[:top_k]
    ]


# ============================================================
# PRESERVED RRF
# ============================================================

def preserved_rrf(
    faiss_docs,
    bm25_docs,
    top_k
):
    if not faiss_docs:
        return rrf_search(
            faiss_docs,
            bm25_docs,
            top_k
        )

    first = faiss_docs[0]
    first_key = doc_key(first)

    total_unique = len(
        set(
            [doc_key(doc) for doc in faiss_docs]
            +
            [doc_key(doc) for doc in bm25_docs]
        )
    )

    fused = rrf_search(
        faiss_docs,
        bm25_docs,
        total_unique
    )

    results = [first]

    for doc in fused:
        if doc_key(doc) == first_key:
            continue

        results.append(doc)

        if len(results) >= top_k:
            break

    return results


# ============================================================
# EVIDENCE RANK
# ============================================================

def find_evidence_rank(
    evidence,
    documents
):
    for rank, doc in enumerate(
        documents,
        start=1
    ):
        if evidence_in_text(
            evidence,
            doc.page_content
        ):
            return {
                "rank": rank,
                "page": doc.metadata.get(
                    "page_number"
                ),
                "chunk_id": doc.metadata.get(
                    "chunk_id"
                ),
            }

    return None


# ============================================================
# TOP-5 COVERAGE
# ============================================================

def calculate_coverage(
    documents,
    evidence_items,
    k=5
):
    docs = documents[:k]

    matched = []
    missing = []

    for evidence in evidence_items:
        found = any(
            evidence_in_text(
                evidence,
                doc.page_content
            )
            for doc in docs
        )

        if found:
            matched.append(evidence)
        else:
            missing.append(evidence)

    if evidence_items:
        coverage = (
            len(matched)
            / len(evidence_items)
        )
    else:
        coverage = 0.0

    return {
        "coverage": coverage,
        "matched": matched,
        "missing": missing,
        "full": (
            len(evidence_items) > 0
            and not missing
        ),
    }


# ============================================================
# PREVIEW
# ============================================================

def preview(text, length=170):
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if len(text) <= length:
        return text

    return text[:length] + "..."


# ============================================================
# PRINT RANK
# ============================================================

def print_rank(name, result):
    if result is None:
        print(
            f"{name:<12}: "
            f"Not found in Top-{DIAGNOSTIC_K}"
        )
        return

    print(
        f"{name:<12}: "
        f"Rank {result['rank']:<2} | "
        f"Page {result['page']} | "
        f"Chunk {result['chunk_id']}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print(
        "🔬 Starting ScholarRAG "
        "Multi-Paper Failure Diagnostic"
    )

    dataset = load_dataset()

    dataset_by_id = {
        item["id"]: item
        for item in dataset
    }

    missing_ids = [
        question_id
        for question_id in TARGETS
        if question_id not in dataset_by_id
    ]

    if missing_ids:
        raise ValueError(
            "These diagnostic IDs are missing "
            f"from dataset.json: {missing_ids}"
        )

    target_items = [
        dataset_by_id[question_id]
        for question_id in TARGETS
    ]

    papers = sorted(
        set(
            item["paper"]
            for item in target_items
        )
    )

    print(
        f"🎯 Diagnostic questions: "
        f"{len(target_items)}"
    )

    print(
        f"📚 Papers involved: "
        f"{len(papers)}"
    )

    print()

    for question_id, label in TARGETS.items():
        print(
            f"   {question_id:<10} → {label}"
        )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print()
    print("🧠 Loading embeddings...")

    embeddings = get_embeddings()

    print("✅ Embeddings ready.")

    results = []

    # ========================================================
    # PAPER LOOP
    # ========================================================

    for paper_name in papers:

        print()
        print("=" * 100)
        print(f"📄 {paper_name}")
        print("=" * 100)

        pages = load_paper(
            paper_name
        )

        print(
            f"Loaded {len(pages)} pages."
        )

        chunks = create_chunks(
            pages
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        bm25 = build_bm25(
            chunks
        )

        paper_questions = [
            item
            for item in target_items
            if item["paper"] == paper_name
        ]

        # ====================================================
        # QUESTION LOOP
        # ====================================================

        for item in paper_questions:

            question_id = item["id"]
            question = item["question"]

            expected = item[
                "expected_evidence"
            ]

            diagnostic_type = TARGETS[
                question_id
            ]

            print()
            print("-" * 100)

            print(
                f"[{question_id}] "
                f"{diagnostic_type.upper()}"
            )

            print(
                f"Question: {question}"
            )

            print()

            # -----------------------------------------------
            # Retrieve
            # -----------------------------------------------

            max_k = min(
                DIAGNOSTIC_K,
                len(chunks)
            )

            faiss_docs = (
                vectorstore.similarity_search(
                    question,
                    k=max_k
                )
            )

            bm25_docs = bm25_search(
                question,
                bm25,
                chunks,
                max_k
            )

            rrf_docs = rrf_search(
                faiss_docs,
                bm25_docs,
                max_k
            )

            preserved_docs = preserved_rrf(
                faiss_docs,
                bm25_docs,
                max_k
            )

            # -----------------------------------------------
            # Coverage
            # -----------------------------------------------

            faiss_cov = calculate_coverage(
                faiss_docs,
                expected
            )

            bm25_cov = calculate_coverage(
                bm25_docs,
                expected
            )

            rrf_cov = calculate_coverage(
                rrf_docs,
                expected
            )

            preserved_cov = calculate_coverage(
                preserved_docs,
                expected
            )

            print("TOP-5 COVERAGE")
            print()

            print(
                f"FAISS      : "
                f"{faiss_cov['coverage']:.3f}"
            )

            print(
                f"BM25       : "
                f"{bm25_cov['coverage']:.3f}"
            )

            print(
                f"RRF        : "
                f"{rrf_cov['coverage']:.3f}"
            )

            print(
                f"PRESERVED  : "
                f"{preserved_cov['coverage']:.3f}"
            )

            # -----------------------------------------------
            # Evidence rank
            # -----------------------------------------------

            evidence_results = {}

            print()
            print("EVIDENCE RANKS")

            for evidence in expected:

                print()
                print(f"Evidence: {evidence}")

                faiss_rank = find_evidence_rank(
                    evidence,
                    faiss_docs
                )

                bm25_rank = find_evidence_rank(
                    evidence,
                    bm25_docs
                )

                rrf_rank = find_evidence_rank(
                    evidence,
                    rrf_docs
                )

                preserved_rank = find_evidence_rank(
                    evidence,
                    preserved_docs
                )

                print_rank(
                    "FAISS",
                    faiss_rank
                )

                print_rank(
                    "BM25",
                    bm25_rank
                )

                print_rank(
                    "RRF",
                    rrf_rank
                )

                print_rank(
                    "PRESERVED",
                    preserved_rank
                )

                evidence_results[
                    evidence
                ] = {
                    "faiss": faiss_rank,
                    "bm25": bm25_rank,
                    "rrf": rrf_rank,
                    "preserved": preserved_rank,
                }

            # -----------------------------------------------
            # Top 5 inspection
            # -----------------------------------------------

            print()
            print("FAISS TOP-5")
            print()

            for rank, doc in enumerate(
                faiss_docs[:5],
                start=1
            ):
                found = [
                    evidence
                    for evidence in expected
                    if evidence_in_text(
                        evidence,
                        doc.page_content
                    )
                ]

                print(
                    f"Rank {rank} | "
                    f"Page "
                    f"{doc.metadata.get('page_number')} | "
                    f"Chunk "
                    f"{doc.metadata.get('chunk_id')}"
                )

                print(
                    f"Evidence: "
                    f"{found if found else 'None'}"
                )

                print(
                    f"Text: "
                    f"{preview(doc.page_content)}"
                )

                print()

            print("BM25 TOP-5")
            print()

            for rank, doc in enumerate(
                bm25_docs[:5],
                start=1
            ):
                found = [
                    evidence
                    for evidence in expected
                    if evidence_in_text(
                        evidence,
                        doc.page_content
                    )
                ]

                print(
                    f"Rank {rank} | "
                    f"Page "
                    f"{doc.metadata.get('page_number')} | "
                    f"Chunk "
                    f"{doc.metadata.get('chunk_id')}"
                )

                print(
                    f"Evidence: "
                    f"{found if found else 'None'}"
                )

                print(
                    f"Text: "
                    f"{preview(doc.page_content)}"
                )

                print()

            print("RRF TOP-5")
            print()

            for rank, doc in enumerate(
                rrf_docs[:5],
                start=1
            ):
                found = [
                    evidence
                    for evidence in expected
                    if evidence_in_text(
                        evidence,
                        doc.page_content
                    )
                ]

                print(
                    f"Rank {rank} | "
                    f"Page "
                    f"{doc.metadata.get('page_number')} | "
                    f"Chunk "
                    f"{doc.metadata.get('chunk_id')}"
                )

                print(
                    f"Evidence: "
                    f"{found if found else 'None'}"
                )

                print(
                    f"Text: "
                    f"{preview(doc.page_content)}"
                )

                print()

            # -----------------------------------------------
            # Automatic diagnosis
            # -----------------------------------------------

            evidence_found_anywhere = all(
                any(
                    result is not None
                    for result in [
                        evidence_results[e]["faiss"],
                        evidence_results[e]["bm25"],
                    ]
                )
                for e in expected
            )

            print("DIAGNOSIS")
            print()

            if (
                faiss_cov["coverage"] > 0
                and rrf_cov["coverage"]
                < faiss_cov["coverage"]
            ):
                diagnosis = (
                    "FUSION_REGRESSION"
                )

                print(
                    "🔴 Fusion is removing or "
                    "demoting useful FAISS evidence."
                )

            elif (
                faiss_cov["coverage"] == 0
                and bm25_cov["coverage"] > 0
                and rrf_cov["coverage"] > 0
            ):
                diagnosis = (
                    "SPARSE_RETRIEVAL_RESCUE"
                )

                print(
                    "🟢 BM25 rescues evidence that "
                    "dense retrieval misses."
                )

            elif (
                faiss_cov["coverage"] == 0
                and bm25_cov["coverage"] == 0
                and evidence_found_anywhere
            ):
                diagnosis = (
                    "RANK_DEPTH_FAILURE"
                )

                print(
                    "🟠 Evidence exists in deeper "
                    "candidate ranks but misses Top-5."
                )

            elif not evidence_found_anywhere:
                diagnosis = (
                    "CANDIDATE_RECALL_FAILURE"
                )

                print(
                    "🔴 At least one expected evidence "
                    "item is absent even from the deep "
                    "FAISS/BM25 candidate sets."
                )

                print(
                    "Likely investigate chunking, "
                    "query formulation or evidence "
                    "representation."
                )

            elif (
                rrf_cov["coverage"]
                > faiss_cov["coverage"]
            ):
                diagnosis = (
                    "HYBRID_IMPROVEMENT"
                )

                print(
                    "🟢 Fusion successfully improves "
                    "dense retrieval."
                )

            else:
                diagnosis = (
                    "MIXED_OR_UNCHANGED"
                )

                print(
                    "🟡 No single failure mechanism "
                    "identified automatically."
                )

            results.append({
                "id": question_id,
                "paper": paper_name,
                "type": diagnostic_type,
                "question": question,
                "expected_evidence": expected,

                "top5": {
                    "faiss": faiss_cov,
                    "bm25": bm25_cov,
                    "rrf": rrf_cov,
                    "preserved": preserved_cov,
                },

                "evidence_ranks": (
                    evidence_results
                ),

                "diagnosis": diagnosis,
            })

    # ========================================================
    # SUMMARY
    # ========================================================

    counts = defaultdict(int)

    for result in results:
        counts[
            result["diagnosis"]
        ] += 1

    print()
    print("=" * 100)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 100)
    print()

    for diagnosis, count in sorted(
        counts.items()
    ):
        print(
            f"{diagnosis:<30}: {count}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "diagnostic_k": DIAGNOSTIC_K,
            "production_k": 5,
            "rrf_k": RRF_K,
        },

        "targets": TARGETS,

        "summary": dict(counts),

        "results": results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("💾 Results saved to:")
    print(OUTPUT_PATH)

    print()
    print(
        "✅ Multi-paper failure "
        "diagnostic complete."
    )


if __name__ == "__main__":
    main()