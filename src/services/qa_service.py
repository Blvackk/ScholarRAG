# qa.py

from functools import lru_cache

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from config import Config


PROMPT = PromptTemplate(
    template="""
You are an AI Research Assistant.

Answer ONLY from the given context.

If the answer is not present in the context, say:
"I couldn't find that information in the uploaded research paper."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


@lru_cache(maxsize=1)
def _load_llm():
    print("🧠 Loading Gemma model...")

    llm = LlamaCpp(
        model_path=Config.MODEL_PATH,
        n_threads=Config.LLM_THREADS,
        n_ctx=Config.LLM_CTX,
        max_tokens=512,
        temperature=0.2,
        verbose=True,
    )

    print("✅ Gemma model loaded successfully!")

    return llm


@lru_cache(maxsize=1)
def get_qa_chain(vector_store):
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