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

OUTPUT_PATH = RESULTS_DIR / "reranker_experiment.json"

CHUNK_SIZE = 2048
CHUNK_OVERLAP = 512

TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60


# ============================================================
# TEXT NORMALISATION
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    text = text.lower()

    for old, new in [
        ("–", "-"),
        ("—", "-"),
        ("−", "-"),
        ("’", "'"),
        ("“", '"'),
        ("”", '"'),
    ]:
        text = text.replace(old, new)

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

    return (
        compact_text(evidence)
        in compact_text(text)
    )


def tokenize(text):
    return re.findall(
        r"[a-z0-9]+",
        normalize_text(text)
    )


# ============================================================
# DATA
# ============================================================

def load_dataset():
    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_paper(paper_name):
    path = PAPERS_DIR / paper_name

    if not path.exists():
        raise FileNotFoundError(
            f"Paper not found: {path}"
        )

    pages = PyPDFLoader(str(path)).load()

    for page_num, page in enumerate(
        pages,
        start=1
    ):
        page.metadata["paper"] = paper_name
        page.metadata["page_number"] = page_num

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

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


# ============================================================
# BM25
# ============================================================

def build_bm25(chunks):
    corpus = [
        tokenize(doc.page_content)
        for doc in chunks
    ]

    return BM25Okapi(corpus)


def bm25_search(query, bm25, chunks, k):
    scores = bm25.get_scores(
        tokenize(query)
    )

    indices = np.argsort(scores)[::-1]

    return [
        chunks[int(i)]
        for i in indices[:k]
    ]


# ============================================================
# IDENTIFICATION
# ============================================================

def doc_key(doc):
    return (
        doc.metadata.get("paper"),
        doc.metadata.get("chunk_id")
    )


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
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
            1 / (RRF_K + rank)
        )

    for rank, doc in enumerate(
        bm25_docs,
        start=1
    ):
        key = doc_key(doc)

        docs[key] = doc
        scores[key] += (
            1 / (RRF_K + rank)
        )

    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        docs[key]
        for key in ranked[:top_k]
    ]


# ============================================================
# CANDIDATE POOL
# ============================================================

def build_candidate_pool(
    faiss_docs,
    bm25_docs
):
    """
    Merge FAISS and BM25 candidates without duplicates.
    """

    pool = {}
    faiss_rank = {}
    bm25_rank = {}

    for rank, doc in enumerate(
        faiss_docs,
        start=1
    ):
        key = doc_key(doc)

        pool[key] = doc
        faiss_rank[key] = rank

    for rank, doc in enumerate(
        bm25_docs,
        start=1
    ):
        key = doc_key(doc)

        pool[key] = doc
        bm25_rank[key] = rank

    return (
        pool,
        faiss_rank,
        bm25_rank
    )


# ============================================================
# LIGHTWEIGHT RERANKER
# ============================================================

def query_overlap_score(query, document):
    """
    Lexical overlap between question and chunk.

    Important:
    This uses ONLY the user question and document.
    It does NOT use expected evaluation evidence.
    """

    query_tokens = set(tokenize(query))
    doc_tokens = set(
        tokenize(document.page_content)
    )

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens & doc_tokens
    )

    return (
        len(overlap)
        / len(query_tokens)
    )


def rerank_candidates(
    query,
    faiss_docs,
    bm25_docs,
    top_k
):
    """
    Rerank the union of FAISS + BM25 candidates.

    Features:
        1. FAISS rank
        2. BM25 rank
        3. Query-document lexical overlap

    No expected evidence is used here.
    """

    (
        pool,
        faiss_rank,
        bm25_rank
    ) = build_candidate_pool(
        faiss_docs,
        bm25_docs
    )

    scored = []

    for key, doc in pool.items():

        f_rank = faiss_rank.get(key)
        b_rank = bm25_rank.get(key)

        dense_score = (
            1 / f_rank
            if f_rank is not None
            else 0.0
        )

        sparse_score = (
            1 / b_rank
            if b_rank is not None
            else 0.0
        )

        overlap_score = query_overlap_score(
            query,
            doc
        )

        # Balanced ranking score.
        score = (
            0.40 * dense_score
            +
            0.40 * sparse_score
            +
            0.20 * overlap_score
        )

        scored.append(
            (
                score,
                doc,
                f_rank,
                b_rank,
                overlap_score
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        item[1]
        for item in scored[:top_k]
    ]


# ============================================================
# EVALUATION
# ============================================================

def evaluate_docs(
    docs,
    expected_evidence
):
    if not expected_evidence:
        return {
            "hit": False,
            "full": False,
            "coverage": 0.0,
            "rr": 0.0,
            "matched": [],
            "missing": [],
        }

    matched = []
    missing = []
    ranks = []

    for evidence in expected_evidence:

        found_rank = None

        for rank, doc in enumerate(
            docs,
            start=1
        ):
            if evidence_in_text(
                evidence,
                doc.page_content
            ):
                found_rank = rank
                break

        if found_rank is None:
            missing.append(evidence)
        else:
            matched.append(evidence)
            ranks.append(found_rank)

    coverage = (
        len(matched)
        / len(expected_evidence)
    )

    return {
        "hit": bool(matched),
        "full": not missing,
        "coverage": coverage,
        "rr": (
            1 / min(ranks)
            if ranks
            else 0.0
        ),
        "matched": matched,
        "missing": missing,
    }


def aggregate(records, method):
    results = [
        item[method]
        for item in records
    ]

    if not results:
        return {
            "questions": 0,
            "hit_at_5": 0.0,
            "full_at_5": 0.0,
            "coverage": 0.0,
            "mrr": 0.0,
        }

    n = len(results)

    return {
        "questions": n,

        "hit_at_5": sum(
            r["hit"] for r in results
        ) / n,

        "full_at_5": sum(
            r["full"] for r in results
        ) / n,

        "coverage": sum(
            r["coverage"]
            for r in results
        ) / n,

        "mrr": sum(
            r["rr"] for r in results
        ) / n,
    }


# ============================================================
# DISPLAY
# ============================================================

def icon(value):
    if value >= 0.999:
        return "✅"

    if value > 0:
        return "⚠️"

    return "❌"


def print_metrics(
    faiss,
    rrf,
    reranker
):
    print(
        f"{'Metric':<25}"
        f"{'FAISS':<15}"
        f"{'RRF':<15}"
        f"{'RERANKER':<15}"
    )

    print("-" * 70)

    metrics = [
        ("Evidence Hit@5", "hit_at_5"),
        ("Full Evidence@5", "full_at_5"),
        ("Mean Coverage", "coverage"),
        ("MRR", "mrr"),
    ]

    for label, key in metrics:
        print(
            f"{label:<25}"
            f"{faiss[key]:<15.3f}"
            f"{rrf[key]:<15.3f}"
            f"{reranker[key]:<15.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "🧪 Starting ScholarRAG "
        "Candidate Reranker Experiment"
    )

    dataset = load_dataset()

    answerable = [
        item
        for item in dataset
        if item["category"]
        != "unanswerable"
    ]

    papers = sorted(
        set(
            item["paper"]
            for item in answerable
        )
    )

    print(
        f"📋 Total questions : "
        f"{len(dataset)}"
    )

    print(
        f"🎯 Answerable      : "
        f"{len(answerable)}"
    )

    print(
        f"📚 Papers          : "
        f"{len(papers)}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print()
    print("🧠 Loading embeddings...")

    embeddings = get_embeddings()

    print("✅ Embeddings ready.")

    # --------------------------------------------------------
    # Group questions
    # --------------------------------------------------------

    questions_by_paper = defaultdict(
        list
    )

    for item in answerable:
        questions_by_paper[
            item["paper"]
        ].append(item)

    records = []
    per_paper = defaultdict(list)

    # ========================================================
    # PAPERS
    # ========================================================

    for paper_index, paper_name in enumerate(
        papers,
        start=1
    ):

        print()
        print("=" * 100)

        print(
            f"PAPER {paper_index}/"
            f"{len(papers)}: "
            f"{paper_name}"
        )

        print("=" * 100)

        documents = load_paper(
            paper_name
        )

        chunks = create_chunks(
            documents
        )

        print(
            f"📄 Pages  : {len(documents)}"
        )

        print(
            f"✂️ Chunks : {len(chunks)}"
        )

        print("🔎 Building FAISS...")

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        print("✅ FAISS ready.")

        print("🔤 Building BM25...")

        bm25 = build_bm25(
            chunks
        )

        print("✅ BM25 ready.")

        # ====================================================
        # QUESTIONS
        # ====================================================

        for item in questions_by_paper[
            paper_name
        ]:

            question = item["question"]
            expected = item[
                "expected_evidence"
            ]

            k = min(
                CANDIDATE_K,
                len(chunks)
            )

            faiss_candidates = (
                vectorstore.similarity_search(
                    question,
                    k=k
                )
            )

            bm25_candidates = bm25_search(
                question,
                bm25,
                chunks,
                k
            )

            faiss_top = (
                faiss_candidates[:TOP_K]
            )

            rrf_top = reciprocal_rank_fusion(
                faiss_candidates,
                bm25_candidates,
                TOP_K
            )

            reranked_top = rerank_candidates(
                question,
                faiss_candidates,
                bm25_candidates,
                TOP_K
            )

            faiss_result = evaluate_docs(
                faiss_top,
                expected
            )

            rrf_result = evaluate_docs(
                rrf_top,
                expected
            )

            reranker_result = evaluate_docs(
                reranked_top,
                expected
            )

            record = {
                "id": item["id"],
                "paper": paper_name,
                "question": question,
                "expected_evidence": expected,

                "faiss": faiss_result,
                "rrf": rrf_result,
                "reranker": reranker_result,
            }

            records.append(record)

            per_paper[
                paper_name
            ].append(record)

            print()
            print(
                f"[{item['id']}] "
                f"{question}"
            )

            print(
                f"FAISS    → "
                f"{faiss_result['coverage']:.3f} "
                f"{icon(faiss_result['coverage'])}"
            )

            print(
                f"RRF      → "
                f"{rrf_result['coverage']:.3f} "
                f"{icon(rrf_result['coverage'])}"
            )

            print(
                f"RERANKER → "
                f"{reranker_result['coverage']:.3f} "
                f"{icon(reranker_result['coverage'])}"
            )

    # ========================================================
    # GLOBAL RESULTS
    # ========================================================

    faiss_global = aggregate(
        records,
        "faiss"
    )

    rrf_global = aggregate(
        records,
        "rrf"
    )

    reranker_global = aggregate(
        records,
        "reranker"
    )

    print()
    print("=" * 100)
    print("GLOBAL RESULTS")
    print("=" * 100)
    print()

    print_metrics(
        faiss_global,
        rrf_global,
        reranker_global
    )

    # ========================================================
    # PER-PAPER RESULTS
    # ========================================================

    paper_summary = {}

    print()
    print("=" * 100)
    print("PER-PAPER RESULTS")
    print("=" * 100)

    for paper_name in papers:

        paper_records = per_paper[
            paper_name
        ]

        f = aggregate(
            paper_records,
            "faiss"
        )

        r = aggregate(
            paper_records,
            "rrf"
        )

        rr = aggregate(
            paper_records,
            "reranker"
        )

        paper_summary[
            paper_name
        ] = {
            "faiss": f,
            "rrf": r,
            "reranker": rr,
        }

        print()
        print(f"📄 {paper_name}")

        print_metrics(
            f,
            r,
            rr
        )

    # ========================================================
    # QUESTION LEVEL CHANGES
    # ========================================================

    improved_vs_faiss = []
    regressed_vs_faiss = []

    improved_vs_rrf = []
    regressed_vs_rrf = []

    for record in records:

        reranker_cov = (
            record["reranker"]["coverage"]
        )

        faiss_cov = (
            record["faiss"]["coverage"]
        )

        rrf_cov = (
            record["rrf"]["coverage"]
        )

        if reranker_cov > faiss_cov:
            improved_vs_faiss.append(
                record["id"]
            )

        elif reranker_cov < faiss_cov:
            regressed_vs_faiss.append(
                record["id"]
            )

        if reranker_cov > rrf_cov:
            improved_vs_rrf.append(
                record["id"]
            )

        elif reranker_cov < rrf_cov:
            regressed_vs_rrf.append(
                record["id"]
            )

    print()
    print("=" * 100)
    print("QUESTION-LEVEL CHANGES")
    print("=" * 100)

    print()
    print("RERANKER vs FAISS")
    print(
        f"Improved  : "
        f"{len(improved_vs_faiss)}"
    )
    print(
        f"Regressed : "
        f"{len(regressed_vs_faiss)}"
    )

    if improved_vs_faiss:
        print(
            "Improved IDs:",
            improved_vs_faiss
        )

    if regressed_vs_faiss:
        print(
            "Regressed IDs:",
            regressed_vs_faiss
        )

    print()
    print("RERANKER vs RRF")

    print(
        f"Improved  : "
        f"{len(improved_vs_rrf)}"
    )

    print(
        f"Regressed : "
        f"{len(regressed_vs_rrf)}"
    )

    if improved_vs_rrf:
        print(
            "Improved IDs:",
            improved_vs_rrf
        )

    if regressed_vs_rrf:
        print(
            "Regressed IDs:",
            regressed_vs_rrf
        )

    # ========================================================
    # CRITICAL QUESTIONS
    # ========================================================

    critical_ids = {
        "p01_q04",
        "p01_q08",
        "p02_q10",
        "p03_q06",
        "p03_q08",
        "p03_q09",
        "p04_q07",
    }

    print()
    print("=" * 100)
    print("CRITICAL QUESTION CHECK")
    print("=" * 100)

    critical_results = {}

    for record in records:

        if record["id"] not in critical_ids:
            continue

        critical_results[
            record["id"]
        ] = {
            "faiss": (
                record["faiss"]["coverage"]
            ),
            "rrf": (
                record["rrf"]["coverage"]
            ),
            "reranker": (
                record[
                    "reranker"
                ]["coverage"]
            ),
        }

        print()
        print(record["id"])

        print(
            f"  FAISS    : "
            f"{record['faiss']['coverage']:.3f}"
        )

        print(
            f"  RRF      : "
            f"{record['rrf']['coverage']:.3f}"
        )

        print(
            f"  RERANKER : "
            f"{record['reranker']['coverage']:.3f}"
        )

    # ========================================================
    # WINNER
    # ========================================================

    configurations = {
        "FAISS": faiss_global,
        "RRF": rrf_global,
        "RERANKER": reranker_global,
    }

    winner = max(
        configurations,
        key=lambda name: (
            configurations[name]["coverage"],
            configurations[name]["full_at_5"],
            configurations[name]["mrr"],
        )
    )

    print()
    print("=" * 100)
    print("FINAL DECISION")
    print("=" * 100)

    print()
    print(
        f"🏆 Best configuration: {winner}"
    )

    print(
        f"Coverage : "
        f"{configurations[winner]['coverage']:.3f}"
    )

    print(
        f"Full@5   : "
        f"{configurations[winner]['full_at_5']:.3f}"
    )

    print(
        f"Hit@5    : "
        f"{configurations[winner]['hit_at_5']:.3f}"
    )

    print(
        f"MRR      : "
        f"{configurations[winner]['mrr']:.3f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output = {
        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,

            "reranker": {
                "dense_weight": 0.40,
                "sparse_weight": 0.40,
                "query_overlap_weight": 0.20,
            },
        },

        "dataset": {
            "total_questions": len(dataset),
            "answerable_questions": len(
                answerable
            ),
            "papers": papers,
        },

        "global": {
            "faiss": faiss_global,
            "rrf": rrf_global,
            "reranker": reranker_global,
        },

        "per_paper": paper_summary,

        "question_changes": {
            "reranker_vs_faiss": {
                "improved": (
                    improved_vs_faiss
                ),
                "regressed": (
                    regressed_vs_faiss
                ),
            },

            "reranker_vs_rrf": {
                "improved": (
                    improved_vs_rrf
                ),
                "regressed": (
                    regressed_vs_rrf
                ),
            },
        },

        "critical_questions": (
            critical_results
        ),

        "winner": winner,

        "questions": records,
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
        "✅ Reranker experiment complete."
    )


if __name__ == "__main__":
    main()