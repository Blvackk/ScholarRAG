import json
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.rag.loader import load_pdf
from src.rag.chunker import split_documents
from src.rag.embeddings import get_embeddings


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
PAPERS_DIR = BASE_DIR / "papers"
RESULTS_DIR = BASE_DIR / "results"

OUTPUT_PATH = RESULTS_DIR / "query_expansion_experiment.json"

TOP_K = 5


# ============================================================
# Queries to test
# ============================================================

EXPANDED_QUERIES = {
    "q04": (
        "What research methodology, study design, literature review "
        "method, data collection approach, experimental procedure, "
        "and research method were used in this study?"
    ),

    "q08": (
        "What mitigation strategies, interventions, adaptive autonomy "
        "methods, explainable AI interface designs, micro-interventions, "
        "and workload management techniques are proposed for reducing "
        "mental fatigue in human-in-the-loop AI systems?"
    ),
}


# ============================================================
# Normalisation
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


def evidence_in_text(text, evidence):

    return (
        normalize_text(evidence)
        in normalize_text(text)
    )


# ============================================================
# Load Dataset
# ============================================================

def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# Load PDFs
# ============================================================

def load_papers():

    pdf_files = sorted(
        PAPERS_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {PAPERS_DIR}"
        )

    documents = []

    print("\n📄 Loading evaluation papers...")

    for pdf_path in pdf_files:

        print(
            f"   Loading: {pdf_path.name}"
        )

        docs = load_pdf(
            str(pdf_path)
        )

        documents.extend(docs)

    print(
        f"✅ Loaded {len(documents)} pages."
    )

    return documents


# ============================================================
# Evaluate Query
# ============================================================

def evaluate_query(
    vector_store,
    query,
    expected_evidence,
):

    documents = (
        vector_store.similarity_search(
            query,
            k=TOP_K,
        )
    )

    matched = []
    missing = []
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

        evidence_ranks[evidence] = (
            found_rank
        )

        if found_rank is not None:
            matched.append(evidence)
        else:
            missing.append(evidence)

    total = len(expected_evidence)

    coverage = (
        len(matched) / total
        if total
        else 0.0
    )

    full = (
        total > 0
        and len(missing) == 0
    )

    pages = []

    for document in documents:

        page = document.metadata.get(
            "page"
        )

        if page is not None:
            page = int(page) + 1

        pages.append(page)

    return {
        "query": query,
        "matched_evidence": matched,
        "missing_evidence": missing,
        "coverage": coverage,
        "full_evidence": full,
        "evidence_ranks": evidence_ranks,
        "retrieved_pages": pages,
    }


# ============================================================
# Print Result
# ============================================================

def print_result(
    name,
    result,
):

    print(f"\n{name}")

    print(
        f"Query    : {result['query']}"
    )

    print(
        f"Matched  : "
        f"{result['matched_evidence']}"
    )

    print(
        f"Missing  : "
        f"{result['missing_evidence']}"
    )

    print(
        f"Coverage : "
        f"{result['coverage']:.3f}"
    )

    print(
        "Full     : "
        f"{'✅' if result['full_evidence'] else '❌'}"
    )

    print(
        f"Pages    : "
        f"{result['retrieved_pages']}"
    )

    print("Evidence ranks:")

    for evidence, rank in (
        result["evidence_ranks"].items()
    ):

        if rank is None:

            print(
                f"   ❌ {evidence}: "
                f"not in Top-{TOP_K}"
            )

        else:

            print(
                f"   ✅ {evidence}: "
                f"rank {rank}"
            )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n🧪 Starting ScholarRAG "
        "Query Expansion Experiment"
    )

    dataset = load_dataset()

    dataset_by_id = {
        item["id"]: item
        for item in dataset
    }

    # --------------------------------------------------------
    # Papers
    # --------------------------------------------------------

    documents = load_papers()

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

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

    print(
        "\n🔎 Building FAISS index..."
    )

    vector_store = (
        FAISS.from_documents(
            chunks,
            embeddings,
        )
    )

    print("✅ FAISS ready.")

    # --------------------------------------------------------
    # Experiments
    # --------------------------------------------------------

    results = []

    print("\n" + "=" * 88)
    print(
        "SCHOLARRAG QUERY EXPANSION EXPERIMENT"
    )
    print("=" * 88)

    for question_id, expanded_query in (
        EXPANDED_QUERIES.items()
    ):

        item = dataset_by_id[
            question_id
        ]

        original_query = (
            item["question"]
        )

        expected_evidence = (
            item["expected_evidence"]
        )

        print(
            f"\n[{question_id}] "
            f"{original_query}"
        )

        print(
            "\nExpected Evidence:"
        )

        for evidence in (
            expected_evidence
        ):
            print(
                f"  • {evidence}"
            )

        # Original
        original_result = (
            evaluate_query(
                vector_store,
                original_query,
                expected_evidence,
            )
        )

        # Expanded
        expanded_result = (
            evaluate_query(
                vector_store,
                expanded_query,
                expected_evidence,
            )
        )

        print("\n" + "-" * 88)

        print_result(
            "ORIGINAL QUERY",
            original_result,
        )

        print("\n" + "-" * 88)

        print_result(
            "EXPANDED QUERY",
            expanded_result,
        )

        improvement = (
            expanded_result["coverage"]
            - original_result["coverage"]
        )

        print("\nResult:")

        if improvement > 0:

            print(
                "🟢 Query expansion improved "
                f"coverage by "
                f"{improvement:.3f}"
            )

        elif improvement == 0:

            print(
                "🟡 Query expansion produced "
                "no coverage improvement."
            )

        else:

            print(
                "🔴 Query expansion reduced "
                f"coverage by "
                f"{abs(improvement):.3f}"
            )

        results.append(
            {
                "id": question_id,
                "question": (
                    original_query
                ),
                "expected_evidence": (
                    expected_evidence
                ),
                "original": (
                    original_result
                ),
                "expanded": (
                    expanded_result
                ),
                "coverage_improvement": (
                    improvement
                ),
            }
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 88)
    print("FINAL COMPARISON")
    print("=" * 88)

    print(
        f"\n{'ID':<6}"
        f"{'Original':<15}"
        f"{'Expanded':<15}"
        f"{'Change':<15}"
    )

    print("-" * 51)

    for result in results:

        original = (
            result["original"][
                "coverage"
            ]
        )

        expanded = (
            result["expanded"][
                "coverage"
            ]
        )

        change = (
            result[
                "coverage_improvement"
            ]
        )

        print(
            f"{result['id']:<6}"
            f"{original:<15.3f}"
            f"{expanded:<15.3f}"
            f"{change:+.3f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "experiment": (
            "query_expansion"
        ),
        "top_k": TOP_K,
        "results": results,
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

    print(
        "\n💾 Results saved to:"
    )

    print(OUTPUT_PATH)

    print(
        "\n✅ Query expansion "
        "experiment complete."
    )


if __name__ == "__main__":
    main()