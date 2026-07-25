# src/services/qa_service.py

from functools import lru_cache

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from config import Config


# ==========================================================
# ScholarRAG Prompt
# ==========================================================

PROMPT = PromptTemplate(
    template="""
You are ScholarRAG, an AI Research Assistant.

Your job is to help the user understand the uploaded research paper.

Answer the user's question using the provided research-paper context.

The user can ask anything about the paper, such as:
- explanations
- summaries
- methodology
- results
- datasets
- models
- limitations
- conclusions
- technical concepts

Do not invent information that is not supported by the context.

If the requested information cannot be found in the provided context,
clearly say:

"I couldn't find that information in the uploaded research paper."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


# ==========================================================
# Load Local LLM
# ==========================================================

@lru_cache(maxsize=1)
def _load_llm():
    """
    Load the local Gemma GGUF model once and reuse it.
    """

    print("🧠 Loading Gemma model...")

    llm = LlamaCpp(
        model_path=Config.MODEL_PATH,
        n_threads=Config.LLM_THREADS,
        n_ctx=Config.LLM_CTX,
        max_tokens=512,
        temperature=0.2,
        verbose=False,
    )

    print("✅ Gemma model loaded successfully!")

    return llm


# ==========================================================
# Create QA Chain
# ==========================================================

def get_qa_chain(vector_store):
    """
    Create a RetrievalQA chain for the supplied vector store.
    """

    if vector_store is None:
        raise ValueError(
            "Cannot create QA chain because vector store is None."
        )

    print("🚀 Creating RetrievalQA chain...")

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=_load_llm(),
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT
        },
    )

    print("✅ QA Chain Ready!")

    return qa_chain