# app.py

import os
import time
import psutil

from src.rag.vector_store import (
    load_or_create_index,
    create_index_for_pdf,
)
from src.services.qa_service import get_qa_chain
from src.services.upload_service import save_uploaded_pdf
from src.ui.app_ui import create_ui


# ==========================================================
# Application State
# ==========================================================

vector_store = None
qa_chain = None
active_paper = None


# ==========================================================
# Load Existing Index
# ==========================================================

print("📚 Initializing ScholarRAG...")

try:

    vector_store = load_or_create_index()

    if vector_store is not None:

        qa_chain = get_qa_chain(
            vector_store
        )

        print("✅ Existing FAISS index loaded.")

    else:

        print(
            "ℹ️ No indexed paper found. "
            "Upload a research paper from the UI."
        )

except Exception as e:

    print(
        f"⚠️ Could not load existing index: {e}"
    )


# ==========================================================
# Upload + Index
# ==========================================================

def handle_upload(pdf_path):

    global vector_store
    global qa_chain
    global active_paper

    success, message, saved_path = (
        save_uploaded_pdf(pdf_path)
    )

    if not success:
        return message

    try:

        print(
            f"📄 Indexing: "
            f"{os.path.basename(saved_path)}"
        )

        new_vector_store = create_index_for_pdf(
            saved_path
        )

        new_qa_chain = get_qa_chain(
            new_vector_store
        )

        # Only replace application state after
        # indexing succeeds completely.
        vector_store = new_vector_store
        qa_chain = new_qa_chain

        active_paper = os.path.basename(
            saved_path
        )

        return (
            "✅ Research paper ready!\n\n"
            f"📄 {active_paper}\n\n"
            "You can now ask anything about this paper."
        )

    except Exception as e:

        return (
            "❌ The PDF was saved, but ScholarRAG "
            "could not index it.\n\n"
            f"{e}"
        )


# ==========================================================
# Ask Research Paper
# ==========================================================

def respond(message, history):

    if history is None:
        history = []

    if not message or not message.strip():
        return history

    if qa_chain is None:

        answer = (
            "📄 Upload a research paper first. "
            "Once it has been indexed, you can ask "
            "anything about it."
        )

        history.append(
            (message, answer)
        )

        return history

    try:

        start = time.time()

        result = qa_chain.invoke(
            {"query": message}
        )

        elapsed = time.time() - start

        answer = result.get(
            "result",
            "No answer was generated."
        )

        sources = result.get(
            "source_documents",
            []
        )

        if sources:

            source_lines = set()

            for document in sources:

                filename = os.path.basename(
                    document.metadata.get(
                        "source",
                        active_paper or "Research Paper"
                    )
                )

                page = document.metadata.get(
                    "page"
                )

                # PyPDFLoader uses zero-based page numbers.
                if isinstance(page, int):
                    page = page + 1
                else:
                    page = "?"

                source_lines.add(
                    f"- {filename} — Page {page}"
                )

            answer += (
                "\n\n**Sources**\n"
                + "\n".join(
                    sorted(source_lines)
                )
            )

        memory_mb = (
            psutil.Process(os.getpid())
            .memory_info()
            .rss
            / 1e6
        )

        answer += (
            f"\n\n_Time: {elapsed:.2f}s"
            f" · Memory: {memory_mb:.1f} MB_"
        )

    except Exception as e:

        answer = (
            "❌ ScholarRAG could not generate "
            "an answer.\n\n"
            f"{e}"
        )

    history.append(
        (message, answer)
    )

    return history


# ==========================================================
# UI
# ==========================================================

if __name__ == "__main__":

    demo = create_ui(
        respond=respond,
        upload_pdf=handle_upload
    )

    demo.launch(
        inbrowser=True,
        share=False,
        show_api=False
    )