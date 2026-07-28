import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.embeddings import get_embeddings


# ==========================================================
# Configuration
# ==========================================================

EVAL_DIR = Path(__file__).resolve().parent

DATASET_PATH = EVAL_DIR / "dataset.json"
PAPERS_DIR = EVAL_DIR / "papers"
RESULTS_DIR = EVAL_DIR / "results"

OUTPUT_PATH = RESULTS_DIR / "chunk_experiment.json"

TOP_K = 5

CHUNK_CONFIGS = [
    {
        "chunk_size": 512,
        "chunk_overlap": 128
    },
    {
        "chunk_size": 1024,
        "chunk_overlap": 256
    },
    {
        "chunk_size": 2048,
        "chunk_overlap": 512
    }
]


# ==========================================================
# Helpers
# ==========================================================

def normalize_text(text):
    return " ".join(text.lower().split())


def contains_evidence(text, evidence):
    return (
        normalize_text(evidence)
        in normalize_text(text)
    )


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
# Load PDFs
# ==========================================================

def load_documents():

    print("\n📄 Loading evaluation papers...")

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

    return documents


# ==========================================================
# Create Chunks
# ==========================================================

def create_chunks(
    documents,
    chunk_size,
    chunk_overlap
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# ==========================================================
# Evaluate Configuration
# ==========================================================

def evaluate_configuration(
    documents,
    embeddings,
    dataset,
    chunk_size,
    chunk_overlap
):

    # ------------------------------------------------------
    # Create chunks
    # ------------------------------------------------------

    chunks = create_chunks(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # ------------------------------------------------------
    # Build temporary FAISS index
    # ------------------------------------------------------

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    answerable_count = 0

    evidence_hits = 0
    full_evidence_hits = 0

    coverages = []
    reciprocal_ranks = []

    question_results = []

    # ------------------------------------------------------
    # Evaluate questions
    # ------------------------------------------------------

    for item in dataset:

        expected_evidence = item.get(
            "expected_evidence",
            []
        )

        # Skip unanswerable questions
        if not expected_evidence:
            continue

        answerable_count += 1

        question = item["question"]

        retrieved_docs = (
            vector_store.similarity_search(
                question,
                k=TOP_K
            )
        )

        matched_evidence = []
        first_evidence_rank = None

        # --------------------------------------------------
        # Find evidence
        # --------------------------------------------------

        for evidence in expected_evidence:

            for rank, document in enumerate(
                retrieved_docs,
                start=1
            ):

                if contains_evidence(
                    document.page_content,
                    evidence
                ):

                    matched_evidence.append(
                        evidence
                    )

                    if (
                        first_evidence_rank is None
                        or rank < first_evidence_rank
                    ):
                        first_evidence_rank = rank

                    break

        # --------------------------------------------------
        # Question Metrics
        # --------------------------------------------------

        coverage = (
            len(matched_evidence)
            / len(expected_evidence)
        )

        evidence_hit = (
            len(matched_evidence) > 0
        )

        full_evidence_hit = (
            len(matched_evidence)
            == len(expected_evidence)
        )

        reciprocal_rank = (
            1.0 / first_evidence_rank
            if first_evidence_rank
            else 0.0
        )

        if evidence_hit:
            evidence_hits += 1

        if full_evidence_hit:
            full_evidence_hits += 1

        coverages.append(
            coverage
        )

        reciprocal_ranks.append(
            reciprocal_rank
        )

        question_results.append(
            {
                "id": item["id"],
                "coverage": coverage,
                "evidence_hit": evidence_hit,
                "full_evidence_hit":
                    full_evidence_hit,
                "first_evidence_rank":
                    first_evidence_rank,
                "matched_evidence":
                    matched_evidence
            }
        )

    # ======================================================
    # Aggregate Metrics
    # ======================================================

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

    mean_coverage = (
        sum(coverages) / len(coverages)
        if coverages
        else 0.0
    )

    evidence_mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
        if reciprocal_ranks
        else 0.0
    )

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "number_of_chunks": len(chunks),

        "evidence_hit_at_k":
            evidence_hit_at_k,

        "full_evidence_at_k":
            full_evidence_at_k,

        "mean_evidence_coverage":
            mean_coverage,

        "evidence_mrr":
            evidence_mrr,

        "questions":
            question_results
    }


# ==========================================================
# Run Experiment
# ==========================================================

def run_experiment(
    documents,
    embeddings,
    dataset
):

    results = []

    print("\n" + "=" * 90)
    print("SCHOLARRAG CHUNKING EXPERIMENT")
    print("=" * 90)

    print(
        f"\n{'Size':<10}"
        f"{'Overlap':<10}"
        f"{'Chunks':<10}"
        f"{'Hit@5':<12}"
        f"{'Full@5':<12}"
        f"{'Coverage':<12}"
        f"{'MRR':<10}"
    )

    print("-" * 76)

    for config in CHUNK_CONFIGS:

        chunk_size = config[
            "chunk_size"
        ]

        chunk_overlap = config[
            "chunk_overlap"
        ]

        result = evaluate_configuration(
            documents=documents,
            embeddings=embeddings,
            dataset=dataset,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        results.append(result)

        print(
            f"{chunk_size:<10}"
            f"{chunk_overlap:<10}"
            f"{result['number_of_chunks']:<10}"
            f"{result['evidence_hit_at_k']:<12.3f}"
            f"{result['full_evidence_at_k']:<12.3f}"
            f"{result['mean_evidence_coverage']:<12.3f}"
            f"{result['evidence_mrr']:<10.3f}"
        )

    return results


# ==========================================================
# Save Results
# ==========================================================

def save_results(results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "experiment": "chunk_size_and_overlap",
        "retriever": "similarity",
        "top_k": TOP_K,
        "configurations": results
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\n💾 Results saved to:\n"
        f"{OUTPUT_PATH}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\n🧪 Starting ScholarRAG "
        "Chunking Experiment"
    )

    dataset = load_dataset()

    print(
        f"📋 Loaded {len(dataset)} "
        f"evaluation questions."
    )

    documents = load_documents()

    print("\n🧠 Loading embedding model...")

    embeddings = get_embeddings()

    print("✅ Embeddings ready.")

    results = run_experiment(
        documents=documents,
        embeddings=embeddings,
        dataset=dataset
    )

    save_results(results)

    print(
        "\n✅ Chunking experiment complete."
    )


if __name__ == "__main__":
    main()