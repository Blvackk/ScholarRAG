# 🎓 ScholarRAG

**AI-Powered Research Paper Explainer & Research Assistant**

ScholarRAG is a local **Retrieval-Augmented Generation (RAG)** application that allows users to upload research papers in PDF format and interact with them using natural-language questions.

Instead of manually searching through long and technical research papers, users can upload a PDF directly through the web interface and ask questions such as:

```text
What is this paper about?

Summarize this research paper.

Explain the methodology in simple terms.

Which dataset was used?

What are the main findings?

What limitations did the authors mention?
```

ScholarRAG retrieves relevant sections from the uploaded paper and provides them as context to a locally running **Gemma LLM**, allowing responses to remain grounded in the research paper.

---

## ✨ Current Features

### 📄 Frontend PDF Upload

Research papers can be uploaded directly through the Gradio interface.

The user no longer needs to manually place PDFs into the backend project directory.

### ⚡ Automatic PDF Processing

After upload, ScholarRAG automatically:

1. Validates the uploaded PDF
2. Saves the document locally
3. Extracts its text
4. Splits the text into chunks
5. Generates embeddings
6. Creates a FAISS vector index
7. Connects the new index to the QA pipeline

Once processing finishes, the paper is ready for questions.

### 💬 Free-Form Research Paper Q&A

ScholarRAG does **not** restrict users to predefined actions or buttons.

Users can prompt the assistant naturally:

```text
Summarize this paper.
```

```text
Explain section 3 in beginner-friendly language.
```

```text
What model architecture did the researchers use?
```

```text
What are the limitations of this approach?
```

The user's prompt determines what the assistant should do.

### 🔎 Semantic Retrieval

ScholarRAG uses vector similarity search to retrieve sections of the paper relevant to the user's question.

### 🧠 Local Embeddings

Document chunks are converted into embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 📦 FAISS Vector Store

Document embeddings are indexed using FAISS for efficient semantic retrieval.

### 🤖 Local Gemma LLM

Answers are generated using a locally running:

```text
Gemma 2 2B Instruct
```

GGUF model through `llama-cpp-python`.

This means the core LLM inference pipeline can run locally without requiring a cloud LLM API.

### 📚 Source Information

Retrieved source documents are returned by the QA pipeline so the application can display source/page information alongside generated answers.

### 📊 Runtime Information

ScholarRAG tracks response time and process memory usage during question answering.

---

# 🏗️ Architecture

```text
                         USER
                           │
                           │
                    Upload PDF
                           │
                           ▼
                  ┌─────────────────┐
                  │    Gradio UI    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Upload Service  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   PDF Loader    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Text Chunking   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Embeddings    │
                  │  MiniLM-L6-v2   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │      FAISS      │
                  │  Vector Index   │
                  └────────┬────────┘
                           │
                           │
                  User asks anything
                           │
                           ▼
                  ┌─────────────────┐
                  │    Retriever    │
                  └────────┬────────┘
                           │
                    Relevant Chunks
                           │
                           ▼
                  ┌─────────────────┐
                  │   Gemma LLM     │
                  │   LlamaCpp      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     Answer      │
                  └─────────────────┘
```

---

# 🔄 How ScholarRAG Works

## 1. Upload Research Paper

The user selects a PDF from the Gradio interface.

```text
User
 ↓
Select PDF
 ↓
Upload & Process
```

The upload service validates the document and saves it inside:

```text
uploads/
```

Uploaded documents are excluded from Git.

---

## 2. Extract PDF Content

The PDF is loaded using `PyPDFLoader`.

The loader extracts text and document metadata such as the source file and page number.

```text
PDF
 ↓
PyPDFLoader
 ↓
Documents
```

---

## 3. Chunk the Document

A research paper may contain thousands of words and cannot simply be passed to the LLM as one large prompt.

ScholarRAG splits the extracted document into smaller overlapping chunks.

Current configuration:

```python
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 512
```

The overlap helps preserve context between neighbouring chunks.

---

## 4. Generate Embeddings

Each chunk is transformed into a dense numerical representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Conceptually:

```text
Text Chunk
    ↓
Embedding Model
    ↓
Semantic Vector
```

Text with similar meaning should have vectors located closer together in the embedding space.

---

## 5. Build FAISS Index

The generated embeddings and associated document chunks are stored in a FAISS vector index.

```text
Chunks
   ↓
Embeddings
   ↓
FAISS
```

The index is persisted locally inside:

```text
faiss_index/
```

The generated FAISS index is excluded from Git.

---

## 6. Ask a Question

Once the uploaded paper has been processed, the user can enter any natural-language question.

For example:

```text
Why did the researchers choose this architecture?
```

There are no predefined question categories.

---

## 7. Retrieve Relevant Context

The user's question is compared against the indexed document embeddings.

The retriever currently requests:

```python
search_kwargs={"k": 3}
```

meaning the QA pipeline retrieves up to three relevant chunks for the question.

```text
Question
   ↓
Semantic Retrieval
   ↓
Relevant Paper Chunks
```

---

## 8. Generate the Answer

The retrieved context and user question are passed to the local Gemma model.

```text
Retrieved Context
        +
User Question
        ↓
     Gemma
        ↓
     Answer
```

The QA prompt instructs the model to base its response on the retrieved research-paper context and to indicate when requested information cannot be found.

---

# 🧰 Tech Stack

| Area                 | Technology                         |
| -------------------- | ---------------------------------- |
| Programming Language | Python                             |
| Frontend             | Gradio                             |
| RAG Framework        | LangChain                          |
| PDF Processing       | PyPDF / PyPDFLoader                |
| Embeddings           | Hugging Face Sentence Transformers |
| Embedding Model      | all-MiniLM-L6-v2                   |
| Vector Store         | FAISS                              |
| Local LLM            | Gemma 2 2B Instruct                |
| Model Format         | GGUF                               |
| LLM Runtime          | llama.cpp / llama-cpp-python       |
| System Monitoring    | psutil                             |

---

# 📁 Project Structure

```text
ScholarRAG/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── model.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── indexing_service.py
│   │   ├── qa_service.py
│   │   └── upload_service.py
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   └── app_ui.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
│
├── assets/
├── data/
├── models/
├── uploads/
├── faiss_index/
└── tests/
```

Generated directories such as `__pycache__/` and `.gradio/` are not shown.

---

# 📂 Core Components

### `app.py`

Main application entry point.

It connects:

```text
UI
 ↓
Upload
 ↓
Indexing
 ↓
Vector Store
 ↓
QA Pipeline
```

It also manages the currently active vector store and QA chain.

---

### `config.py`

Central configuration for ScholarRAG.

It contains settings such as:

```python
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 512

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_THREADS = 8
LLM_CTX = 4096
```

It also defines:

- project paths
- upload directory
- FAISS directory
- model path
- allowed upload extensions
- maximum upload size

---

### `src/ui/app_ui.py`

Contains the Gradio frontend.

The UI provides:

- PDF upload
- upload/processing status
- research assistant chat
- question input
- conversation clearing

---

### `src/services/upload_service.py`

Responsible for:

- validating uploaded files
- checking supported extensions
- checking file size
- saving uploaded PDFs

---

### `src/services/indexing_service.py`

Coordinates PDF processing before vector indexing.

It supports processing existing documents as well as a newly uploaded PDF.

---

### `src/rag/loader.py`

Loads PDFs and extracts their text.

---

### `src/rag/chunker.py`

Splits extracted documents into overlapping text chunks.

---

### `src/rag/embeddings.py`

Loads the Hugging Face embedding model used to convert document chunks into semantic vectors.

---

### `src/rag/vector_store.py`

Responsible for:

- creating FAISS indexes
- loading existing FAISS indexes
- persisting the index locally
- creating an index for a newly uploaded paper

---

### `src/services/qa_service.py`

Creates the RAG question-answering pipeline.

It connects:

```text
FAISS Retriever
       ↓
Retrieved Context
       ↓
Gemma LLM
       ↓
Answer
```

The local LLM is cached so that Gemma does not need to be loaded again every time a new QA chain is created.

---

# 🚀 Installation

## Prerequisites

Recommended:

```text
Python 3.11
```

The current development environment uses Python 3.11.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Blvackk/ScholarRAG.git
cd ScholarRAG
```

---

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate again:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

The main dependency versions currently used by ScholarRAG are:

```text
gradio==5.38.2

langchain==0.3.24
langchain-community==0.3.22
langchain-core==0.3.56
langchain-huggingface==0.1.2

sentence-transformers==4.1.0

faiss-cpu==1.9.0.post1

llama-cpp-python==0.3.9

pypdf==5.3.1

psutil==5.9.0
```

---

# 🧠 Gemma Model Setup

The local GGUF model is intentionally **not stored in this Git repository** because model files are large.

ScholarRAG currently expects:

```text
gemma-2-2b-it-Q4_K_M.gguf
```

Place the model inside:

```text
ScholarRAG/
└── models/
    └── gemma-2-2b-it-Q4_K_M.gguf
```

The expected model path is configured in `config.py`.

---

# ▶️ Run ScholarRAG

Make sure your virtual environment is activated.

Then:

```powershell
python app.py
```

Gradio will start a local server, typically at:

```text
http://127.0.0.1:7860
```

The application may automatically open in your browser.

---

# 💻 Using the Application

### Step 1 — Upload a Paper

Select a `.pdf` research paper from the interface.

Click:

```text
Upload & Process Paper
```

ScholarRAG will:

```text
Upload
   ↓
Extract
   ↓
Chunk
   ↓
Embed
   ↓
Index
```

Wait until the interface confirms that the research paper is ready.

### Step 2 — Ask Anything

Enter any question about the paper.

For example:

```text
Summarize this research paper.
```

or:

```text
What problem are the authors trying to solve?
```

or:

```text
Explain the methodology in simple terms.
```

or:

```text
What are the main results?
```

### Step 3 — Get a Grounded Response

ScholarRAG retrieves relevant sections from the paper and sends those sections to Gemma as context for answer generation.

---

# 🔒 Privacy & Local Processing

ScholarRAG is designed around a local RAG pipeline.

The Gemma model runs locally through `llama-cpp-python`, while FAISS indexes are stored locally.

Uploaded research papers are stored under:

```text
uploads/
```

and generated indexes under:

```text
faiss_index/
```

Both directories are excluded from Git.

> Local execution does not automatically guarantee complete privacy or security. Review dependencies, model-download mechanisms, application configuration, and deployment settings before using sensitive documents.

---

# 🛡️ Git-Ignored Files

The repository intentionally excludes local, generated, sensitive, and large files such as:

```text
venv/
.venv/

.env
.env.*

.vscode/
.gradio/

__pycache__/
*.py[cod]
*.log

faiss_index/
uploads/

models/
*.gguf

freeze.txt

.DS_Store
Thumbs.db
```

---

# ⚠️ Current Limitations

ScholarRAG v1.1 is focused on building a reliable single-paper RAG workflow.

Current limitations include:

- PDF is currently the supported document format
- Local LLM inference can be slow on CPU-only machines
- Answer quality depends on retrieval quality and the local model
- Complex tables, diagrams, mathematical notation, and figures may not be fully understood from extracted PDF text
- Scanned/image-only PDFs may require OCR support
- Very broad requests such as full-paper summaries may require improved retrieval strategies beyond top-k chunk retrieval
- Generated responses should be verified against the original research paper
- Multi-paper retrieval and comparison are not yet part of the current workflow

---

# 🗺️ Roadmap

## Version 1.0 — Core RAG Pipeline

- [X] Local Gemma GGUF model
- [X] PDF loading
- [X] Text chunking
- [X] Sentence Transformer embeddings
- [X] FAISS indexing
- [X] Semantic retrieval
- [X] Research-paper question answering

---

## Version 1.1 — Frontend PDF Workflow

- [X] Gradio frontend
- [X] PDF upload from UI
- [X] PDF validation
- [X] Automatic PDF processing
- [X] Dynamic FAISS index creation
- [X] Connect uploaded paper to QA pipeline
- [X] Free-form user questions
- [X] Source/page information
- [X] Runtime statistics

---

## Version 1.2 — Retrieval & Conversation Improvements

Planned:

- [ ] Streaming responses
- [ ] Improved retrieval strategies
- [ ] Better citation formatting
- [ ] Conversation-aware follow-up questions
- [ ] Retrieval evaluation
- [ ] Better handling of broad summary questions

---

## Version 2.0 — User & Paper Management

Planned:

- [ ] Login / Register
- [ ] User dashboard
- [ ] Personal research-paper library
- [ ] Multiple paper management
- [ ] Chat history
- [ ] Paper-specific conversations

---

## Version 3.0 — Advanced Research Assistant

Planned:

- [ ] Citation extraction
- [ ] Reference graph visualisation
- [ ] Multi-paper comparison
- [ ] Literature review assistance
- [ ] Search within papers
- [ ] Export responses and summaries
- [ ] Improved table/equation processing
- [ ] OCR support for scanned papers

---

## Version 4.0 — MLOps & Deployment

Planned:

- [ ] Docker
- [ ] Cloud deployment
- [ ] CI/CD
- [ ] Application logging
- [ ] Monitoring
- [ ] RAG evaluation and observability

---

# 🧪 Current Development Status

**Current Version: v1.1**

The current end-to-end workflow is:

```text
Upload Research Paper
        ↓
Validate PDF
        ↓
Extract Text
        ↓
Create Chunks
        ↓
Generate Embeddings
        ↓
Build FAISS Index
        ↓
Ask Any Question
        ↓
Retrieve Relevant Context
        ↓
Gemma Generates Answer
        ↓
Display Answer + Sources
```

The current focus is building a reliable RAG foundation before adding authentication, multi-paper libraries, advanced research features, and deployment infrastructure.

---

# 🤝 Contributing

Suggestions, issues, and contributions are welcome.

If you find a bug or want to propose an improvement, open an issue or submit a pull request.

---

# 👨‍💻 Author

**Pratik Lagishetty**

ScholarRAG — AI-Powered Research Paper Explainer & Research Assistant
