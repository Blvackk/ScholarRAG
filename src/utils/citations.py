import os


def extract_citations(source_documents):
    """
    Extract unique paper/page citations from retrieved documents.

    PyPDFLoader stores page numbers as zero-based indexes,
    so they are converted to human-readable one-based page numbers.
    """

    citations = []
    seen = set()

    for document in source_documents:

        metadata = document.metadata or {}

        source = metadata.get("source", "Unknown")
        page = metadata.get("page")

        # Show only filename instead of complete local path
        paper = os.path.basename(str(source))

        if isinstance(page, int):
            page = page + 1

        citation_key = (paper, page)

        # Avoid duplicate paper/page citations
        if citation_key in seen:
            continue

        seen.add(citation_key)

        citations.append({
            "paper": paper,
            "page": page,
        })

    return citations


def format_citations(source_documents):
    """
    Convert retrieved source documents into a readable
    citation block.
    """

    citations = extract_citations(source_documents)

    if not citations:
        return ""

    lines = ["Sources:"]

    for citation in citations:

        paper = citation["paper"]
        page = citation["page"]

        if page is not None:
            lines.append(
                f"- {paper} — Page {page}"
            )
        else:
            lines.append(
                f"- {paper}"
            )

    return "\n".join(lines)