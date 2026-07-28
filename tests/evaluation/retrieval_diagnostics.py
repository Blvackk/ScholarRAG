import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from src.rag.chunker import split_documents
from src.rag.embeddings import get_embeddings


# ==========================================================
# Paths
# ==========================================================

EVAL_DIR = Path(__file__).resolve().parent

DATASET_PATH = EVAL_DIR / "dataset.json"
PAPERS_DIR = EVAL_DIR / "papers"
RESULTS_DIR = EVAL_DIR / "results"

OUTPUT_PATH = RESULTS_DIR / "retrieval_diagnostics.json"


# ==========================================================
# Helpers
# ==========================================================

def normalize_text(text):
    """Normalize text for evidence matching."""
    return " ".join(text.lower().split())


def contains_evidence(text, evidence):
    """Check whether evidence occurs inside a chunk."""
    return (
        normalize_text(evidence)
        in normalize_text(text)
    )


def get_pdf_page(document):
    """Convert zero-based PyPDF page to human-readable page."""

    page = document.metadata.get("page")

    if page is None:
        return None

    return int(page) + 1


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ==========================================================
# Build Evaluation Index
# ==========================================================

def build_vector_store():

    print("\n📄 Loading evaluation paper...")

    documents = []

    pdf_files = sorted(
        PAPERS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PAPERS_DIR}"
        )

    for pdf_path in pdf_files:

        print(f"   Loading: {pdf_path.name}")

        loader = PyPDFLoader(
            str(pdf_path)
        )

        documents.extend(
            loader.load()
        )

    print(
        f"✅ Loaded {len(documents)} pages."
    )

    print("\n✂️ Creating chunks...")

    chunks = split_documents(
        documents
    )

    print(
        f"✅ Diagnostic index contains "
        f"{len(chunks)} chunks."
    )

    print("\n🧠 Loading embeddings...")

    embeddings = get_embeddings()

    print("✅ Embeddings ready.")

    print("\n🔎 Building FAISS index...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    print("✅ FAISS ready.")

    return vector_store, len(chunks)


# ==========================================================
# Find Evidence Rank
# ==========================================================

def find_evidence_ranks(
    retrieved_docs,
    expected_evidence
):
    """
    Find the first retrieval rank at which each expected
    evidence phrase occurs.
    """

    evidence_results = []

    for evidence in expected_evidence:

        found_rank = None
        found_page = None
        preview = None

        for rank, document in enumerate(
            retrieved_docs,
            start=1
        ):

            if contains_evidence(
                document.page_content,
                evidence
            ):

                found_rank = rank
                found_page = get_pdf_page(
                    document
                )

                preview = (
                    document.page_content[:250]
                    .replace("\n", " ")
                )

                break

        evidence_results.append(
            {
                "evidence": evidence,
                "rank": found_rank,
                "page": found_page,
                "preview": preview
            }
        )

    return evidence_results


# ==========================================================
# Diagnostics
# ==========================================================

def run_diagnostics(
    vector_store,
    total_chunks,
    dataset
):

    results = []

    print("\n" + "=" * 72)
    print("SCHOLARRAG RETRIEVAL RANK DIAGNOSTICS")
    print("=" * 72)

    for item in dataset:

        question_id = item["id"]
        question = item["question"]

        expected_evidence = item.get(
            "expected_evidence",
            []
        )

        # Skip unanswerable questions
        if not expected_evidence:
            continue

        print(
            f"\n[{question_id}] {question}"
        )

        # Retrieve every chunk in ranked order
        retrieved_docs = (
            vector_store.similarity_search(
                question,
                k=total_chunks
            )
        )

        evidence_results = (
            find_evidence_ranks(
                retrieved_docs,
                expected_evidence
            )
        )

        ranks = []

        for evidence_result in evidence_results:

            evidence = evidence_result[
                "evidence"
            ]

            rank = evidence_result["rank"]
            page = evidence_result["page"]

            if rank is None:

                print(
                    f"❌ {evidence}"
                )
                print(
                    "   Not found in any chunk"
                )

            else:

                ranks.append(rank)

                marker = (
                    "✅"
                    if rank <= 3
                    else "⚠️"
                )

                print(
                    f"{marker} {evidence}"
                )

                print(
                    f"   Rank: {rank}"
                    f" | Page: {page}"
                )

        # ----------------------------------------------
        # Question-Level Diagnostic
        # ----------------------------------------------

        if ranks:

            best_rank = min(ranks)
            worst_rank = max(ranks)

        else:

            best_rank = None
            worst_rank = None

        results.append(
            {
                "id": question_id,
                "question": question,
                "evidence": evidence_results,
                "best_evidence_rank": best_rank,
                "worst_evidence_rank": worst_rank
            }
        )

    return results


# ==========================================================
# Save
# ==========================================================

def save_results(results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
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
        f"\n💾 Diagnostics saved to:\n"
        f"{OUTPUT_PATH}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\n🔬 Starting ScholarRAG "
        "Retrieval Diagnostics"
    )

    dataset = load_dataset()

    vector_store, total_chunks = (
        build_vector_store()
    )

    print(
        f"\n📊 Searching across "
        f"{total_chunks} chunks."
    )

    results = run_diagnostics(
        vector_store,
        total_chunks,
        dataset
    )

    save_results(results)

    print(
        "\n✅ Retrieval diagnostics complete."
    )


if __name__ == "__main__":
    main()