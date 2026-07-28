import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from src.rag.chunker import split_documents
from src.rag.embeddings import get_embeddings


# ==========================================================
# Configuration
# ==========================================================

EVAL_DIR = Path(__file__).resolve().parent

DATASET_PATH = EVAL_DIR / "dataset.json"
PAPERS_DIR = EVAL_DIR / "papers"
RESULTS_DIR = EVAL_DIR / "results"

TOP_K = 3


# ==========================================================
# Load Evaluation Dataset
# ==========================================================

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ==========================================================
# Build Temporary Evaluation Vector Store
# ==========================================================

def build_vector_store():

    print("\n📄 Loading evaluation papers...")

    documents = []

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PAPERS_DIR}"
        )

    for pdf_path in pdf_files:

        print(f"   Loading: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))
        pdf_documents = loader.load()

        documents.extend(pdf_documents)

    print(f"✅ Loaded {len(documents)} PDF pages.")

    print("\n✂️ Splitting documents into chunks...")

    chunks = split_documents(documents)

    print(f"✅ Evaluation contains {len(chunks)} chunks.")

    print("\n🧠 Loading embedding model...")

    embeddings = get_embeddings()

    print("✅ Embedding model ready.")

    print("\n🔎 Building temporary FAISS index...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("✅ Evaluation FAISS index created.")

    return vector_store


# ==========================================================
# Helpers
# ==========================================================

def get_pdf_page(document):
    """
    PyPDFLoader stores page numbers using zero-based indexing.

    page = 0 -> PDF page 1
    page = 1 -> PDF page 2
    """

    page = document.metadata.get("page")

    if page is None:
        return None

    return int(page) + 1


def normalize_text(text):
    """
    Basic normalization for evidence matching.
    """

    return " ".join(
        text.lower().split()
    )


def evidence_found_in_text(text, evidence):
    """
    Check whether an evidence phrase occurs in text.

    Matching is case-insensitive and whitespace-normalized.
    """

    normalized_text = normalize_text(text)
    normalized_evidence = normalize_text(evidence)

    return normalized_evidence in normalized_text


# ==========================================================
# Evidence Evaluation
# ==========================================================

def evaluate_evidence(retrieved_docs, expected_evidence):
    """
    Evaluate evidence across the complete Top-K retrieval set.

    Returns:
        matched_evidence:
            Evidence phrases found anywhere in Top-K.

        missing_evidence:
            Evidence phrases not found in Top-K.

        evidence_coverage:
            Fraction of expected evidence phrases retrieved.

        evidence_hit:
            True when at least one expected evidence phrase
            is found.

        full_evidence_hit:
            True when every expected evidence phrase is found.

        first_evidence_rank:
            Rank of the first retrieved chunk containing
            expected evidence.
    """

    if not expected_evidence:
        return {
            "matched_evidence": [],
            "missing_evidence": [],
            "evidence_coverage": None,
            "evidence_hit": None,
            "full_evidence_hit": None,
            "first_evidence_rank": None
        }

    matched_evidence = []
    first_evidence_rank = None

    for evidence in expected_evidence:

        found = False

        for rank, document in enumerate(
            retrieved_docs,
            start=1
        ):

            if evidence_found_in_text(
                document.page_content,
                evidence
            ):

                found = True

                if (
                    first_evidence_rank is None
                    or rank < first_evidence_rank
                ):
                    first_evidence_rank = rank

                break

        if found:
            matched_evidence.append(evidence)

    missing_evidence = [
        evidence
        for evidence in expected_evidence
        if evidence not in matched_evidence
    ]

    evidence_coverage = (
        len(matched_evidence)
        / len(expected_evidence)
    )

    evidence_hit = len(matched_evidence) > 0

    full_evidence_hit = (
        len(matched_evidence)
        == len(expected_evidence)
    )

    return {
        "matched_evidence": matched_evidence,
        "missing_evidence": missing_evidence,
        "evidence_coverage": evidence_coverage,
        "evidence_hit": evidence_hit,
        "full_evidence_hit": full_evidence_hit,
        "first_evidence_rank": first_evidence_rank
    }


# ==========================================================
# Evaluate Retrieval
# ==========================================================

def evaluate(vector_store, dataset):

    results = []

    page_hits = 0
    page_reciprocal_ranks = []

    evidence_hits = 0
    full_evidence_hits = 0
    evidence_reciprocal_ranks = []
    evidence_coverages = []

    answerable_count = 0

    print("\n" + "=" * 70)
    print("SCHOLARRAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for item in dataset:

        question_id = item["id"]
        question = item["question"]

        expected_pages = item.get(
            "expected_pages",
            []
        )

        expected_evidence = item.get(
            "expected_evidence",
            []
        )

        category = item.get(
            "category",
            "unknown"
        )

        print(f"\n[{question_id}] {question}")

        # --------------------------------------------------
        # Unanswerable Questions
        # --------------------------------------------------

        if not expected_pages and not expected_evidence:

            print("Type: UNANSWERABLE")
            print(
                "Skipped for retrieval metrics. "
                "Will be tested during generation evaluation."
            )

            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "category": category,
                    "expected_pages": [],
                    "expected_evidence": [],
                    "retrieved_pages": [],
                    "page_hit": None,
                    "page_reciprocal_rank": None,
                    "matched_evidence": [],
                    "missing_evidence": [],
                    "evidence_coverage": None,
                    "evidence_hit": None,
                    "full_evidence_hit": None,
                    "evidence_reciprocal_rank": None
                }
            )

            continue

        answerable_count += 1

        # --------------------------------------------------
        # Retrieve Top-K
        # --------------------------------------------------

        retrieved_docs = vector_store.similarity_search(
            question,
            k=TOP_K
        )

        retrieved_pages = [
            get_pdf_page(doc)
            for doc in retrieved_docs
        ]

        # --------------------------------------------------
        # Page-Level Evaluation
        # --------------------------------------------------

        page_hit = any(
            page in expected_pages
            for page in retrieved_pages
        )

        if page_hit:
            page_hits += 1

        page_rr = 0.0

        for rank, page in enumerate(
            retrieved_pages,
            start=1
        ):

            if page in expected_pages:
                page_rr = 1.0 / rank
                break

        page_reciprocal_ranks.append(page_rr)

        # --------------------------------------------------
        # Evidence-Level Evaluation
        # --------------------------------------------------

        evidence_result = evaluate_evidence(
            retrieved_docs,
            expected_evidence
        )

        evidence_hit = evidence_result[
            "evidence_hit"
        ]

        full_evidence_hit = evidence_result[
            "full_evidence_hit"
        ]

        evidence_coverage = evidence_result[
            "evidence_coverage"
        ]

        first_evidence_rank = evidence_result[
            "first_evidence_rank"
        ]

        if evidence_hit:
            evidence_hits += 1

        if full_evidence_hit:
            full_evidence_hits += 1

        if evidence_coverage is not None:
            evidence_coverages.append(
                evidence_coverage
            )

        evidence_rr = (
            1.0 / first_evidence_rank
            if first_evidence_rank
            else 0.0
        )

        evidence_reciprocal_ranks.append(
            evidence_rr
        )

        # --------------------------------------------------
        # Display Result
        # --------------------------------------------------

        print(f"Expected pages    : {expected_pages}")
        print(f"Retrieved pages   : {retrieved_pages}")

        print(
            f"Page Hit@{TOP_K}        : "
            f"{'✅' if page_hit else '❌'}"
        )

        print(
            f"Page RR           : "
            f"{page_rr:.3f}"
        )

        print(
            f"Expected evidence : "
            f"{expected_evidence}"
        )

        print(
            f"Matched evidence  : "
            f"{evidence_result['matched_evidence']}"
        )

        print(
            f"Missing evidence  : "
            f"{evidence_result['missing_evidence']}"
        )

        print(
            f"Evidence Hit@{TOP_K}    : "
            f"{'✅' if evidence_hit else '❌'}"
        )

        print(
            f"Full Evidence@{TOP_K}   : "
            f"{'✅' if full_evidence_hit else '❌'}"
        )

        print(
            f"Evidence Coverage : "
            f"{evidence_coverage:.3f}"
        )

        print(
            f"Evidence RR       : "
            f"{evidence_rr:.3f}"
        )

        # --------------------------------------------------
        # Save Question Result
        # --------------------------------------------------

        results.append(
            {
                "id": question_id,
                "question": question,
                "category": category,

                "expected_pages": expected_pages,
                "retrieved_pages": retrieved_pages,

                "page_hit": page_hit,
                "page_reciprocal_rank": page_rr,

                "expected_evidence": expected_evidence,

                "matched_evidence":
                    evidence_result["matched_evidence"],

                "missing_evidence":
                    evidence_result["missing_evidence"],

                "evidence_coverage":
                    evidence_coverage,

                "evidence_hit":
                    evidence_hit,

                "full_evidence_hit":
                    full_evidence_hit,

                "evidence_reciprocal_rank":
                    evidence_rr
            }
        )

    # ======================================================
    # Aggregate Metrics
    # ======================================================

    page_hit_at_k = (
        page_hits / answerable_count
        if answerable_count
        else 0.0
    )

    page_mrr = (
        sum(page_reciprocal_ranks)
        / len(page_reciprocal_ranks)
        if page_reciprocal_ranks
        else 0.0
    )

    evidence_hit_at_k = (
        evidence_hits / answerable_count
        if answerable_count
        else 0.0
    )

    full_evidence_at_k = (
        full_evidence_hits / answerable_count
        if answerable_count
        else 0.0
    )

    evidence_mrr = (
        sum(evidence_reciprocal_ranks)
        / len(evidence_reciprocal_ranks)
        if evidence_reciprocal_ranks
        else 0.0
    )

    mean_evidence_coverage = (
        sum(evidence_coverages)
        / len(evidence_coverages)
        if evidence_coverages
        else 0.0
    )

    # ======================================================
    # Display Final Results
    # ======================================================

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"Top K                    : {TOP_K}")
    print(f"Answerable Questions     : {answerable_count}")

    print("\n--- Page-Level Metrics ---")

    print(
        f"Page Hit@{TOP_K}               : "
        f"{page_hit_at_k:.3f}"
    )

    print(
        f"Page MRR                 : "
        f"{page_mrr:.3f}"
    )

    print("\n--- Evidence-Level Metrics ---")

    print(
        f"Evidence Hit@{TOP_K}           : "
        f"{evidence_hit_at_k:.3f}"
    )

    print(
        f"Full Evidence@{TOP_K}          : "
        f"{full_evidence_at_k:.3f}"
    )

    print(
        f"Mean Evidence Coverage    : "
        f"{mean_evidence_coverage:.3f}"
    )

    print(
        f"Evidence MRR              : "
        f"{evidence_mrr:.3f}"
    )

    return {
        "top_k": TOP_K,

        "answerable_questions":
            answerable_count,

        "page_metrics": {
            "hit_at_k": page_hit_at_k,
            "mrr": page_mrr
        },

        "evidence_metrics": {
            "hit_at_k": evidence_hit_at_k,
            "full_evidence_at_k":
                full_evidence_at_k,
            "mean_coverage":
                mean_evidence_coverage,
            "mrr": evidence_mrr
        },

        "questions": results
    }


# ==========================================================
# Save Results
# ==========================================================

def save_results(results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        RESULTS_DIR
        / "evidence_retrieval_baseline.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\n💾 Results saved to: "
        f"{output_path}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\n🚀 Starting ScholarRAG "
        "Evidence-Level Retrieval Evaluation"
    )

    dataset = load_dataset()

    print(
        f"📋 Loaded {len(dataset)} "
        f"evaluation questions."
    )

    vector_store = build_vector_store()

    results = evaluate(
        vector_store,
        dataset
    )

    save_results(results)

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()