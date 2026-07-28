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

OUTPUT_PATH = RESULTS_DIR / "k_experiment.json"

K_VALUES = [1, 3, 5, 8, 10]


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
# Build Temporary Evaluation Index
# ==========================================================

def build_vector_store():

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

    print("\n✂️ Creating chunks...")

    chunks = split_documents(
        documents
    )

    print(
        f"✅ Evaluation contains "
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

    print("✅ FAISS index ready.")

    return vector_store


# ==========================================================
# Evaluate One K
# ==========================================================

def evaluate_k(
    vector_store,
    dataset,
    k
):

    evidence_hits = 0
    full_evidence_hits = 0

    coverages = []
    reciprocal_ranks = []

    question_results = []

    answerable_count = 0

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
                k=k
            )
        )

        matched = []
        first_rank = None

        # ----------------------------------------------
        # Search for each evidence phrase
        # ----------------------------------------------

        for evidence in expected_evidence:

            found = False

            for rank, doc in enumerate(
                retrieved_docs,
                start=1
            ):

                if contains_evidence(
                    doc.page_content,
                    evidence
                ):

                    matched.append(
                        evidence
                    )

                    found = True

                    if (
                        first_rank is None
                        or rank < first_rank
                    ):
                        first_rank = rank

                    break

            # Explicit for readability
            if not found:
                pass

        # ----------------------------------------------
        # Metrics for this question
        # ----------------------------------------------

        coverage = (
            len(matched)
            / len(expected_evidence)
        )

        evidence_hit = (
            len(matched) > 0
        )

        full_evidence_hit = (
            len(matched)
            == len(expected_evidence)
        )

        rr = (
            1.0 / first_rank
            if first_rank
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
            rr
        )

        question_results.append(
            {
                "id": item["id"],
                "coverage": coverage,
                "evidence_hit": evidence_hit,
                "full_evidence_hit":
                    full_evidence_hit,
                "first_evidence_rank":
                    first_rank,
                "matched_evidence":
                    matched
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
        "k": k,
        "answerable_questions":
            answerable_count,
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
    vector_store,
    dataset
):

    results = []

    print("\n" + "=" * 78)
    print("SCHOLARRAG TOP-K RETRIEVAL EXPERIMENT")
    print("=" * 78)

    print(
        f"\n{'K':<6}"
        f"{'Hit@K':<14}"
        f"{'Full@K':<14}"
        f"{'Coverage':<14}"
        f"{'MRR':<10}"
    )

    print("-" * 58)

    for k in K_VALUES:

        result = evaluate_k(
            vector_store,
            dataset,
            k
        )

        results.append(result)

        print(
            f"{k:<6}"
            f"{result['evidence_hit_at_k']:<14.3f}"
            f"{result['full_evidence_at_k']:<14.3f}"
            f"{result['mean_evidence_coverage']:<14.3f}"
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
        f"\n💾 Results saved to:\n"
        f"{OUTPUT_PATH}"
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\n🧪 Starting ScholarRAG "
        "Top-K Experiment"
    )

    dataset = load_dataset()

    print(
        f"📋 Loaded {len(dataset)} "
        f"evaluation questions."
    )

    vector_store = (
        build_vector_store()
    )

    results = run_experiment(
        vector_store,
        dataset
    )

    save_results(results)

    print(
        "\n✅ Top-K experiment complete."
    )


if __name__ == "__main__":
    main()