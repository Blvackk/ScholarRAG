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

OUTPUT_PATH = RESULTS_DIR / "multipaper_retrieval_eval.json"

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
# PDF LOADING
# ============================================================

def load_paper(paper_name):
    pdf_path = PAPERS_DIR / paper_name

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    for index, page in enumerate(pages, start=1):
        page.metadata["paper"] = paper_name
        page.metadata["page_number"] = index

    return pages


# ============================================================
# CHUNKING
# ============================================================

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

    chunks = splitter.split_documents(documents)

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


def bm25_search(
    query,
    bm25,
    chunks,
    k
):
    scores = bm25.get_scores(
        tokenize(query)
    )

    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indices[:k]:
        results.append(chunks[int(index)])

    return results


# ============================================================
# DOCUMENT IDENTITY
# ============================================================

def doc_key(doc):
    """
    Stable identity for a chunk inside one paper.
    """

    return (
        doc.metadata.get("paper"),
        doc.metadata.get("chunk_id"),
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
# PRESERVED HYBRID
# ============================================================

def preserved_hybrid(
    faiss_docs,
    bm25_docs,
    top_k
):
    """
    Preserve FAISS rank #1 and fill remaining slots
    using equal-weight RRF.

    This is the strategy that removed the q10 regression
    during the single-paper experiments.
    """

    if not faiss_docs:
        return reciprocal_rank_fusion(
            faiss_docs,
            bm25_docs,
            top_k
        )

    preserved = faiss_docs[0]
    preserved_key = doc_key(preserved)

    fused = reciprocal_rank_fusion(
        faiss_docs,
        bm25_docs,
        top_k=len(
            set(
                [doc_key(d) for d in faiss_docs]
                +
                [doc_key(d) for d in bm25_docs]
            )
        )
    )

    results = [preserved]

    for doc in fused:
        if doc_key(doc) == preserved_key:
            continue

        results.append(doc)

        if len(results) >= top_k:
            break

    return results[:top_k]


# ============================================================
# RETRIEVAL EVALUATION
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

    evidence_ranks = []

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

        if found_rank is not None:
            matched.append(evidence)
            evidence_ranks.append(found_rank)
        else:
            missing.append(evidence)

    coverage = (
        len(matched)
        / len(expected_evidence)
    )

    hit = len(matched) > 0
    full = len(missing) == 0

    # Reciprocal rank of first relevant evidence-containing chunk.
    if evidence_ranks:
        rr = 1.0 / min(evidence_ranks)
    else:
        rr = 0.0

    return {
        "hit": hit,
        "full": full,
        "coverage": coverage,
        "rr": rr,
        "matched": matched,
        "missing": missing,
    }


# ============================================================
# METRIC AGGREGATION
# ============================================================

def aggregate(records, method):
    if not records:
        return {
            "questions": 0,
            "hit_at_5": 0.0,
            "full_at_5": 0.0,
            "coverage": 0.0,
            "mrr": 0.0,
        }

    results = [
        record[method]
        for record in records
    ]

    return {
        "questions": len(results),

        "hit_at_5": sum(
            result["hit"]
            for result in results
        ) / len(results),

        "full_at_5": sum(
            result["full"]
            for result in results
        ) / len(results),

        "coverage": sum(
            result["coverage"]
            for result in results
        ) / len(results),

        "mrr": sum(
            result["rr"]
            for result in results
        ) / len(results),
    }


# ============================================================
# DISPLAY
# ============================================================

def status_icon(value):
    if value >= 0.999:
        return "✅"
    elif value > 0:
        return "⚠️"
    return "❌"


def print_metric_table(
    faiss_metrics,
    rrf_metrics,
    preserved_metrics
):
    print(
        f"{'Metric':<25}"
        f"{'FAISS':<15}"
        f"{'RRF':<15}"
        f"{'PRESERVED':<15}"
    )

    print("-" * 70)

    rows = [
        (
            "Evidence Hit@5",
            "hit_at_5"
        ),
        (
            "Full Evidence@5",
            "full_at_5"
        ),
        (
            "Mean Coverage",
            "coverage"
        ),
        (
            "MRR",
            "mrr"
        ),
    ]

    for label, key in rows:
        print(
            f"{label:<25}"
            f"{faiss_metrics[key]:<15.3f}"
            f"{rrf_metrics[key]:<15.3f}"
            f"{preserved_metrics[key]:<15.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print(
        "🧪 Starting ScholarRAG "
        "Multi-Paper Retrieval Evaluation"
    )

    dataset = load_dataset()

    answerable = [
        item
        for item in dataset
        if item["category"] != "unanswerable"
    ]

    papers = sorted(
        set(
            item["paper"]
            for item in dataset
        )
    )

    print(
        f"📋 Loaded {len(dataset)} total questions."
    )
    print(
        f"🎯 Evaluating {len(answerable)} "
        f"answerable questions."
    )
    print(
        f"📚 Papers: {len(papers)}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Embeddings once
    # --------------------------------------------------------

    print()
    print("🧠 Loading embedding model...")

    embeddings = get_embeddings()

    print("✅ Embeddings ready.")

    # --------------------------------------------------------
    # Group questions
    # --------------------------------------------------------

    questions_by_paper = defaultdict(list)

    for item in answerable:
        questions_by_paper[
            item["paper"]
        ].append(item)

    all_records = []
    per_paper_records = defaultdict(list)

    # ========================================================
    # PAPER LOOP
    # ========================================================

    for paper_number, paper_name in enumerate(
        papers,
        start=1
    ):
        questions = questions_by_paper.get(
            paper_name,
            []
        )

        if not questions:
            continue

        print()
        print("=" * 100)
        print(
            f"PAPER {paper_number}/{len(papers)}: "
            f"{paper_name}"
        )
        print("=" * 100)

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        print()
        print("📄 Loading PDF...")

        documents = load_paper(
            paper_name
        )

        print(
            f"✅ Loaded {len(documents)} pages."
        )

        # ----------------------------------------------------
        # Chunk
        # ----------------------------------------------------

        print("✂️ Creating chunks...")

        chunks = create_chunks(
            documents
        )

        print(
            f"✅ Created {len(chunks)} chunks."
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        print("🔎 Building FAISS...")

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        print("✅ FAISS ready.")

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        print("🔤 Building BM25...")

        bm25 = build_bm25(
            chunks
        )

        print("✅ BM25 ready.")

        # ----------------------------------------------------
        # Questions
        # ----------------------------------------------------

        for item in questions:
            question_id = item["id"]
            question = item["question"]
            expected = item[
                "expected_evidence"
            ]

            # Dense candidates
            faiss_candidates = (
                vectorstore.similarity_search(
                    question,
                    k=min(
                        CANDIDATE_K,
                        len(chunks)
                    )
                )
            )

            # Sparse candidates
            bm25_candidates = bm25_search(
                question,
                bm25,
                chunks,
                k=min(
                    CANDIDATE_K,
                    len(chunks)
                )
            )

            # FAISS baseline
            faiss_top = (
                faiss_candidates[:TOP_K]
            )

            # Equal RRF
            rrf_top = reciprocal_rank_fusion(
                faiss_candidates,
                bm25_candidates,
                TOP_K
            )

            # Preserved hybrid
            preserved_top = preserved_hybrid(
                faiss_candidates,
                bm25_candidates,
                TOP_K
            )

            # Evaluate
            faiss_result = evaluate_docs(
                faiss_top,
                expected
            )

            rrf_result = evaluate_docs(
                rrf_top,
                expected
            )

            preserved_result = evaluate_docs(
                preserved_top,
                expected
            )

            record = {
                "id": question_id,
                "paper": paper_name,
                "question": question,
                "expected_evidence": expected,
                "faiss": faiss_result,
                "rrf": rrf_result,
                "preserved": preserved_result,
            }

            all_records.append(record)
            per_paper_records[
                paper_name
            ].append(record)

            print()
            print(
                f"[{question_id}] {question}"
            )

            print(
                f"FAISS     → "
                f"{faiss_result['coverage']:.3f} "
                f"{status_icon(faiss_result['coverage'])}"
            )

            print(
                f"RRF       → "
                f"{rrf_result['coverage']:.3f} "
                f"{status_icon(rrf_result['coverage'])}"
            )

            print(
                f"PRESERVED → "
                f"{preserved_result['coverage']:.3f} "
                f"{status_icon(preserved_result['coverage'])}"
            )

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    faiss_global = aggregate(
        all_records,
        "faiss"
    )

    rrf_global = aggregate(
        all_records,
        "rrf"
    )

    preserved_global = aggregate(
        all_records,
        "preserved"
    )

    print()
    print("=" * 100)
    print("GLOBAL RESULTS")
    print("=" * 100)
    print()

    print_metric_table(
        faiss_global,
        rrf_global,
        preserved_global
    )

    # ========================================================
    # PER PAPER
    # ========================================================

    per_paper_summary = {}

    print()
    print("=" * 100)
    print("PER-PAPER RESULTS")
    print("=" * 100)

    for paper_name in papers:
        records = per_paper_records.get(
            paper_name,
            []
        )

        if not records:
            continue

        faiss_metrics = aggregate(
            records,
            "faiss"
        )

        rrf_metrics = aggregate(
            records,
            "rrf"
        )

        preserved_metrics = aggregate(
            records,
            "preserved"
        )

        per_paper_summary[
            paper_name
        ] = {
            "faiss": faiss_metrics,
            "rrf": rrf_metrics,
            "preserved": preserved_metrics,
        }

        print()
        print(f"📄 {paper_name}")
        print_metric_table(
            faiss_metrics,
            rrf_metrics,
            preserved_metrics
        )

    # ========================================================
    # REGRESSION ANALYSIS
    # ========================================================

    print()
    print("=" * 100)
    print(
        "PRESERVED VS FAISS "
        "QUESTION-LEVEL COMPARISON"
    )
    print("=" * 100)

    improved = []
    regressed = []
    unchanged = []

    for record in all_records:
        baseline = record[
            "faiss"
        ]["coverage"]

        preserved = record[
            "preserved"
        ]["coverage"]

        delta = preserved - baseline

        item = {
            "id": record["id"],
            "paper": record["paper"],
            "faiss_coverage": baseline,
            "preserved_coverage": preserved,
            "delta": delta,
        }

        if delta > 1e-9:
            improved.append(item)

        elif delta < -1e-9:
            regressed.append(item)

        else:
            unchanged.append(item)

    print()
    print(
        f"Improved  : {len(improved)}"
    )
    print(
        f"Regressed : {len(regressed)}"
    )
    print(
        f"Unchanged : {len(unchanged)}"
    )

    if improved:
        print()
        print("📈 Improved questions:")

        for item in improved:
            print(
                f"   {item['id']} | "
                f"{item['faiss_coverage']:.3f} "
                f"→ "
                f"{item['preserved_coverage']:.3f} "
                f"({item['delta']:+.3f})"
            )

    if regressed:
        print()
        print("📉 Regressed questions:")

        for item in regressed:
            print(
                f"   {item['id']} | "
                f"{item['faiss_coverage']:.3f} "
                f"→ "
                f"{item['preserved_coverage']:.3f} "
                f"({item['delta']:+.3f})"
            )

    # ========================================================
    # WINNER
    # ========================================================

    configurations = {
        "FAISS": faiss_global,
        "RRF": rrf_global,
        "PRESERVED": preserved_global,
    }

    # Primary objective = coverage.
    # Full evidence is secondary.
    # MRR breaks remaining ties.
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
        f"🏆 Best aggregate configuration: "
        f"{winner}"
    )

    print(
        f"Coverage      : "
        f"{configurations[winner]['coverage']:.3f}"
    )

    print(
        f"Full Evidence : "
        f"{configurations[winner]['full_at_5']:.3f}"
    )

    print(
        f"Hit@5         : "
        f"{configurations[winner]['hit_at_5']:.3f}"
    )

    print(
        f"MRR           : "
        f"{configurations[winner]['mrr']:.3f}"
    )

    print()

    if (
        winner == "PRESERVED"
        and len(regressed) == 0
    ):
        decision = (
            "PRESERVED_HYBRID_RECOMMENDED"
        )

        print(
            "✅ Preserved hybrid wins without "
            "question-level coverage regressions."
        )

        print(
            "It is a strong candidate for the "
            "production ScholarRAG retriever."
        )

    elif winner == "PRESERVED":
        decision = (
            "PRESERVED_HYBRID_PROMISING"
        )

        print(
            "⚠️ Preserved hybrid has the best "
            "aggregate result, but regressions remain."
        )

        print(
            "Inspect the regressed questions before "
            "production integration."
        )

    elif winner == "RRF":
        decision = "RRF_RECOMMENDED"

        print(
            "Equal RRF performs best across the "
            "multi-paper benchmark."
        )

    else:
        decision = "FAISS_RECOMMENDED"

        print(
            "FAISS remains strongest on this "
            "multi-paper benchmark."
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
            "preserved": preserved_global,
        },

        "per_paper": per_paper_summary,

        "comparison": {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        },

        "winner": winner,
        "decision": decision,

        "questions": all_records,
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
        "✅ Multi-paper retrieval "
        "evaluation complete."
    )


if __name__ == "__main__":
    main()