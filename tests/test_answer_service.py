from src.rag.vector_store import load_or_create_index
from src.services.answer_service import answer_question


def main():

    print("\n========================================")
    print("ScholarRAG Answer Service Test")
    print("========================================\n")

    # 1. Load production vector store
    print("Loading vector store...")

    vector_store = load_or_create_index()

    if vector_store is None:
        raise RuntimeError(
            "Vector store could not be loaded."
        )

    print("Vector store ready.\n")

    # 2. Ask a real question
    question = "What methodology was used in this study?"

    print("Question:")
    print(question)

    print("\nGenerating answer...\n")

    # 3. Run complete application-level QA pipeline
    result = answer_question(
        vector_store=vector_store,
        question=question,
    )

    # 4. Display answer
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    # 5. Display structured retrieved sources
    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    sources = result["sources"]

    if not sources:
        print("No sources returned.")

    for source in sources:

        paper = source["paper"]
        page = source["page"]

        if page is not None:
            print(f"- {paper} — Page {page}")
        else:
            print(f"- {paper}")

    print("\n" + "=" * 70)

    # 6. Basic validation
    print("\nValidation:")

    print(
        "Answer present :",
        bool(result["answer"])
    )

    print(
        "Sources present:",
        bool(result["sources"])
    )

    print(
        "Documents      :",
        len(result["source_documents"])
    )


if __name__ == "__main__":
    main()