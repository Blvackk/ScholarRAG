from langchain_core.documents import Document

from src.utils.citations import (
    extract_citations,
    format_citations,
)


def main():

    documents = [
        Document(
            page_content="Methodology evidence...",
            metadata={
                "source": r"C:\papers\paper_01.pdf",
                "page": 1,
            },
        ),
        Document(
            page_content="Additional methodology evidence...",
            metadata={
                "source": r"C:\papers\paper_01.pdf",
                "page": 2,
            },
        ),
        # Duplicate citation — should be removed
        Document(
            page_content="Another chunk from page 2...",
            metadata={
                "source": r"C:\papers\paper_01.pdf",
                "page": 1,
            },
        ),
    ]

    citations = extract_citations(documents)

    print("\nStructured citations:")
    print(citations)

    print("\nFormatted citations:")
    print(format_citations(documents))


if __name__ == "__main__":
    main()