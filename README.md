# 🎓 ScholarRAG

> **AI-Powered Research Intelligence Platform using Retrieval-Augmented Generation (RAG)**

ScholarRAG is a fully local AI-powered research assistant that enables users to interact with research papers through natural language conversations. The system uses a **Retrieval-Augmented Generation (RAG)** pipeline to retrieve relevant information from uploaded PDF documents and generate accurate, context-aware responses using a **local Large Language Model (LLM)**.

Unlike cloud-based AI assistants, ScholarRAG processes documents locally, ensuring **data privacy**, **offline accessibility**, and **zero dependency on external APIs**.

---

## 🚀 Features

- 📄 **Research Paper Processing**

  - Extracts text from PDF research papers.
  - Supports multiple PDF documents.
- 🧠 **Local Large Language Model**

  - Uses **Google Gemma-2-2B-IT GGUF** running locally through **Llama.cpp**.
  - No OpenAI API or cloud services required.
- 🔍 **Semantic Search**

  - Generates embeddings using **Sentence Transformers**.
  - Retrieves the most relevant document chunks using **FAISS Vector Database**.
- 💬 **Conversational Question Answering**

  - Ask questions in natural language.
  - Generates responses using only retrieved document context.
- 📚 **Source Citations**

  - Displays the document name and page number for retrieved information.
- ⚡ **Fast Retrieval**

  - Automatically caches the FAISS index.
  - Documents are indexed only once.
- 🖥️ **Interactive Interface**

  - Clean web interface powered by **Gradio**.
- 🔒 **Privacy First**

  - Runs entirely on your local machine.
  - No data leaves your computer.

---

# 🏗 System Architecture

```text
                    Research Papers (PDF)
                            │
                            ▼
                    PDF Text Extraction
                            │
                            ▼
                  Text Cleaning & Processing
                            │
                            ▼
                     Document Chunking
                            │
                            ▼
                Sentence Transformer Embeddings
                            │
                            ▼
                  FAISS Vector Database
                            │
                            ▼
                    User Question (Query)
                            │
                            ▼
                  Similarity Search (Top-K)
                            │
                            ▼
                 Retrieved Context + Prompt
                            │
                            ▼
                 Gemma-2 Local LLM (GGUF)
                            │
                            ▼
                Context-Aware AI Response
```

---

# ⚙️ How It Works

ScholarRAG follows a standard Retrieval-Augmented Generation (RAG) workflow divided into two stages.

## 1️⃣ Document Ingestion & Indexing (Offline)

Performed only once for newly added documents.

### Step 1 — Load PDFs

Research papers stored inside the `docs/` directory are loaded using **PyPDFLoader**.

### Step 2 — Clean Text

The extracted text is cleaned by removing:

- Extra whitespace
- Invalid characters
- Non-ASCII symbols

### Step 3 — Chunk Documents

Large documents are divided into overlapping chunks using:

- RecursiveCharacterTextSplitter

### Step 4 — Generate Embeddings

Each chunk is converted into dense vector embeddings using:

- all-MiniLM-L6-v2

### Step 5 — Store in FAISS

The generated embeddings are stored in a FAISS vector database for efficient similarity search.

---

## 2️⃣ Retrieval & Question Answering (Online)

Executed every time a user submits a question.

### Step 1

User enters a question.

↓

### Step 2

The question is converted into an embedding.

↓

### Step 3

FAISS retrieves the most relevant document chunks.

↓

### Step 4

The retrieved context is combined with the user query.

↓

### Step 5

The complete prompt is sent to the local Gemma LLM.

↓

### Step 6

The model generates a context-aware answer with source citations.

---

# 📂 Project Structure

```text
ScholarRAG/
│
├── app.py                 # Gradio Application
├── config.py              # Project Configuration
├── processing.py          # PDF Processing & FAISS Indexing
├── qa.py                  # Question Answering Pipeline
├── requirements.txt
├── README.md
│
├── docs/                  # Research Papers
│
├── models/                # GGUF Models
│
├── faiss_index/           # Cached Vector Database
│
└── .gitignore
```

---

# 🛠 Technology Stack

| Category             | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| LLM                  | Google Gemma-2-2B-IT      |
| LLM Runtime          | Llama.cpp                 |
| Framework            | LangChain                 |
| Embedding Model      | Sentence Transformers     |
| Vector Database      | FAISS                     |
| PDF Processing       | PyPDF                     |
| UI                   | Gradio                    |
| Machine Learning     | Hugging Face Transformers |

---

# 💻 Installation

## Prerequisites

- Python **3.11**
- Git
- Windows / Linux / macOS
- Visual Studio Build Tools (Windows)

---

## Clone Repository

```bash
git clone https://github.com/Blvackk/ScholarRAG.git

cd ScholarRAG
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download Local LLM

Download

**Gemma-2-2B-IT GGUF (Q4_K_M)**

Place the downloaded model inside

```text
models/
```

---

## Add Research Papers

Copy your PDF files into

```text
docs/
```

---

## Run Application

```bash
python app.py
```

The application will

- Create the FAISS index (first run only)
- Load the Gemma model
- Launch the Gradio interface

---


# 📈 Current Features

- ✅ Local AI Research Assistant
- ✅ PDF Question Answering
- ✅ Semantic Search
- ✅ FAISS Vector Database
- ✅ Local Gemma LLM
- ✅ Source Citations
- ✅ Gradio Web Interface

---

# 🚀 Planned Enhancements

The following features are planned for future releases:

- 🔐 User Authentication (JWT)
- 👤 User Dashboard
- 📤 PDF Upload from Frontend
- 💾 Chat History
- 📚 Multi-user Knowledge Base
- 📝 AI Research Summarization
- 📊 Research Analytics Dashboard
- 📄 Automatic Citation Generator
- 📑 Literature Review Generator
- ☁ Docker Deployment
- 🚀 CI/CD Pipeline
- 🌐 Cloud Deployment

---

# 🎯 Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Vector Databases
- Semantic Search
- LangChain
- Prompt Engineering
- Information Retrieval
- Natural Language Processing
- AI Application Development

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Pratik Lagishetty**

AI | Machine Learning | Data Science | Generative AI

GitHub: https://github.com/Blvackk/

k AI Similarity Search (FAISS)](https://faiss.ai/)
- **UI**: [Gradio](https://www.gradio.app/)
