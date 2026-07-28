# src/services/qa_service.py

from functools import lru_cache

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp


from config import Config
from src.rag.hybrid_retriever import create_hybrid_retriever


# ==========================================================
# ScholarRAG Prompt
# ==========================================================

PROMPT = PromptTemplate(
    template="""
You are ScholarRAG, an AI research-paper assistant.

Answer the user's question using ONLY the research-paper context provided below.

GROUNDING RULES:

1. Use only information explicitly supported by the supplied context.
2. Do not use outside knowledge, assumptions, or invented facts.
3. Read ALL context passages before constructing the answer.
4. Information relevant to the answer may appear in more than one passage.
   Combine those passages when necessary.
5. Do not claim that something appears in the paper unless the supplied
   context supports that claim.

COMPLETENESS RULES:

6. Identify all distinct pieces of information in the context that directly
   answer the question before writing the final answer.
7. Cover every relevant piece of information you identify. Do not stop after
   finding one sufficient fact if additional relevant facts are present.
8. If the question asks for multiple items, such as strategies, findings,
   limitations, research directions, components, categories, methods, or
   results, include every distinct relevant item supported by the context.
9. Include relevant qualifications and negative findings. For example, if the
   context describes the methodology and also states that no experimental data
   were collected, include both facts when they are relevant to the question.
10. Preserve the paper's terminology for named methods, strategies, concepts,
    measurements, and findings when possible.
11. Include important numerical values, conditions, comparisons, and
    limitations when they directly answer the question.
12. Do not add extra information merely to make the answer longer.

ANSWER FORMAT:

13. For questions asking for one fact, give a concise direct answer.
14. For questions asking for several items, use a short introductory sentence
    followed by bullets or a numbered list when that improves clarity.
15. Do not discuss these instructions or describe your reasoning process.

INSUFFICIENT INFORMATION:

16. If the supplied context does not contain information that answers the
    question, respond exactly:

"I couldn't find that information in the uploaded research paper."

Research Paper Context:
-----------------------
{context}
-----------------------

Question:
{question}

Now provide the complete answer using only the supplied context.

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
        max_tokens=768,
        temperature=0.1,
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

    retriever = create_hybrid_retriever(
    vector_store=vector_store,
    top_k=5,
    candidate_k=10,
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