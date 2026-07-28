from src.rag.vector_store import load_or_create_index
from src.rag.hybrid_retriever import create_hybrid_retriever


QUESTION = (
    "What mitigation strategies are proposed for reducing "
    "mental fatigue in HITL systems?"
)

EVIDENCE = [
    "Adaptive Autonomy",
    "XAI Interface Design",
    "Micro-Interventions",
    "Workload Triage",
]


def check_evidence(document):

    text = document.page_content.lower()

    return [
        evidence
        for evidence in EVIDENCE
        if evidence.lower() in text
    ]


def page_number(document):

    page = document.metadata.get(
        "page",
        "Unknown"
    )

    if isinstance(page, int):
        return page + 1

    return page


def print_results(name, documents):

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    for rank, document in enumerate(
        documents,
        start=1
    ):

        matched = check_evidence(
            document
        )

        print(
            f"Rank {rank:<2} | "
            f"Page {page_number(document)} | "
            f"Evidence: "
            f"{matched if matched else 'None'}"
        )


def main():

    print(
        "\nScholarRAG q08 Production Ranking Diagnostic"
    )

    vector_store = load_or_create_index()

    retriever = create_hybrid_retriever(
        vector_store=vector_store,
        top_k=10,
        candidate_k=20,
    )

    # ---------------------------------------------
    # FAISS
    # ---------------------------------------------

    faiss_documents = retriever._faiss_search(
        QUESTION
    )

    # ---------------------------------------------
    # BM25
    # ---------------------------------------------

    bm25_documents = retriever._bm25_search(
        QUESTION
    )

    # ---------------------------------------------
    # RRF
    # ---------------------------------------------

    rrf_documents = retriever._rrf_fusion(
        faiss_documents,
        bm25_documents,
    )

    print(f"\nQuestion: {QUESTION}")

    print_results(
        "FAISS TOP-20",
        faiss_documents
    )

    print_results(
        "BM25 TOP-20",
        bm25_documents
    )

    print_results(
        "RRF TOP-10",
        rrf_documents
    )

    print("\n" + "=" * 80)
    print("EXPECTED EVIDENCE LOCATION")
    print("=" * 80)

    for evidence in EVIDENCE:

        print(f"\n{evidence}")

        for name, documents in [
            ("FAISS", faiss_documents),
            ("BM25", bm25_documents),
            ("RRF", rrf_documents),
        ]:

            rank_found = None
            page_found = None

            for rank, document in enumerate(
                documents,
                start=1
            ):

                if (
                    evidence.lower()
                    in document.page_content.lower()
                ):

                    rank_found = rank
                    page_found = page_number(
                        document
                    )

                    break

            if rank_found is None:

                print(
                    f"  {name:<6}: Not found"
                )

            else:

                print(
                    f"  {name:<6}: "
                    f"Rank {rank_found} | "
                    f"Page {page_found}"
                )


if __name__ == "__main__":
    main()