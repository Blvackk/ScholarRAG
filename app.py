import os
import time
import psutil
import gradio as gr

from src.services.qa_service import get_qa_chain
from src.rag.vector_store import load_or_create_index


print("📚 Loading Vector Store...")
vector_store = load_or_create_index()

print("🤖 Initializing QA Chain...")

try:
    qa_chain = get_qa_chain(vector_store)
    print("✅ QA Chain Initialized!")
except Exception:
    print("❌ Error while creating QA Chain:")
    raise


def respond(message: str, history: list):

    start = time.time()

    result = qa_chain({"query": message})

    elapsed = time.time() - start

    answer = result.get("result", "❗ No answer generated.")
    sources = result.get("source_documents", [])

    if sources:
        unique_sources = sorted(
            list(
                set(
                    f"- {os.path.basename(doc.metadata.get('source', ''))} "
                    f"(p.{doc.metadata.get('page', '?')})"
                    for doc in sources
                )
            )
        )

        answer += "\n\n**Sources:**\n" + "\n".join(unique_sources)

    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6

    answer += f"\n\n_Time: {elapsed:.2f}s · Mem: {mem_mb:.1f} MB_"

    return answer


if __name__ == "__main__":

    demo = gr.ChatInterface(
        fn=respond,
        examples=[
            ["What is attention mechanism?"],
            ["Explain fine-tuning."]
        ],
        title=" ScholarRAG: AI Research Paper Explainer",
        description="Ask questions about your PDF documents.",
        theme="soft",
        cache_examples=False
    )

    demo.launch()