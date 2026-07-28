from src.rag.vector_store import load_or_create_index
from src.rag.hybrid_retriever import create_hybrid_retriever
from src.utils.citations import extract_citations, format_citations


def main():

    print("\n========================================")
    print("ScholarRAG Real Citation Test")
    print("========================================\n")

    # Load production FAISS index
    vector_store = load_or_create_index()

    if vector_store is None:
        raise RuntimeError("Vector store could not be loaded.")

    # Create the same hybrid retriever used in production
    retriever = create_hybrid_retriever(
        vector_store=vector_store,
        top_k=5,
        candidate_k=10,
    )

    question = "What methodology was used in this study?"

    print("Question:")
    print(question)

    # Retrieve real chunks
    documents = retriever.invoke(question)

    print(f"\nRetrieved documents: {len(documents)}")

    # Structured citations
    citations = extract_citations(documents)

    print("\nStructured citations:")
    for citation in citations:
        print(citation)

    # User-facing citations
    print("\n" + "=" * 60)
    print(format_citations(documents))
    print("=" * 60)


if __name__ == "__main__":
    main()