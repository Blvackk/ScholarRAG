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

OUTPUT_PATH = RESULTS_DIR / "retriever_experiment.json"

K_VALUES = [3, 5]

# MMR retrieves a larger candidate pool first, then selects
# a diverse/relevant subset of K documents.
MMR_FETCH_K = 15
MMR_LAMBDA = 0.5


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
# Build Evaluation Vector Store
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
# Retrieval
# ==========================================================

def retrieve_documents(
    vector_store,
    question,
    method,
    k
):

    if method == "similarity":

        return vector_store.similarity_search(
            question,
            k=k
        )

    if method == "mmr":

        return vector_store.max_marginal_relevance_search(
            question,
            k=k,
            fetch_k=MMR_FETCH_K,
            lambda_mult=MMR_LAMBDA
        )

    raise ValueError(
        f"Unknown retrieval method: {method}"
    )


# ==========================================================
# Evaluate Configuration
# ==========================================================

def evaluate_configuration(
    vector_store,
    dataset,
    method,
    k
):

    answerable_count = 0

    evidence_hits = 0
    full_evidence_hits = 0

    coverages = []
    reciprocal_ranks = []

    question_results = []

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

        retrieved_docs = retrieve_documents(
            vector_store=vector_store,
            question=question,
            method=method,
            k=k
        )

        matched_evidence = []
        first_evidence_rank = None

        # --------------------------------------------------
        # Find evidence across retrieved chunks
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
                "question": question,
                "matched_evidence":
                    matched_evidence,
                "evidence_coverage":
                    coverage,
                "evidence_hit":
                    evidence_hit,
                "full_evidence_hit":
                    full_evidence_hit,
                "first_evidence_rank":
                    first_evidence_rank
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

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
        if reciprocal_ranks
        else 0.0
    )

    return {
        "method": method,
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
            mrr,
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

    configurations = [
        ("similarity", 3),
        ("similarity", 5),
        ("mmr", 3),
        ("mmr", 5),
    ]

    results = []

    print("\n" + "=" * 82)
    print("SCHOLARRAG RETRIEVER EXPERIMENT")
    print("=" * 82)

    print(
        f"\n{'Method':<15}"
        f"{'K':<6}"
        f"{'Hit@K':<13}"
        f"{'Full@K':<13}"
        f"{'Coverage':<13}"
        f"{'MRR':<10}"
    )

    print("-" * 70)

    for method, k in configurations:

        result = evaluate_configuration(
            vector_store=vector_store,
            dataset=dataset,
            method=method,
            k=k
        )

        results.append(result)

        print(
            f"{method:<15}"
            f"{k:<6}"
            f"{result['evidence_hit_at_k']:<13.3f}"
            f"{result['full_evidence_at_k']:<13.3f}"
            f"{result['mean_evidence_coverage']:<13.3f}"
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
        "experiment": "similarity_vs_mmr",
        "mmr_settings": {
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA
        },
        "results": results
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
        "Retriever Experiment"
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
        "\n✅ Retriever experiment complete."
    )


if __name__ == "__main__":
    main()