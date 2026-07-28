from src.services.qa_service import get_qa_chain
from src.utils.citations import extract_citations


def answer_question(vector_store, question):
    """
    Run the ScholarRAG QA pipeline and return a clean,
    structured response for the application layer.
    """

    qa_chain = get_qa_chain(vector_store)

    response = qa_chain.invoke({
        "query": question
    })

    answer = response.get("result", "").strip()

    source_documents = response.get(
        "source_documents",
        []
    )

    citations = extract_citations(
        source_documents
    )

    return {
        "answer": answer,
        "sources": citations,
        "source_documents": source_documents,
    }