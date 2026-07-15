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

GitHub: https://github.com/Blvac

# 🔬 Research Assistant RAG

This project is a fully local, chat-based Research Assistant that uses Retrieval Augmented Generation (RAG) to answer questions about scientific papers and other documents in PDF format. It leverages a local LLM and a vector database, ensuring your data remains private.

---

### Features

- **📄 PDF Document Processing**: Ingests and processes multiple PDF files from a local directory.
- **🧠 Local LLM**: Uses the CPU-optimized  e.g. `Gemma-2-2B-IT` model in GGUF format, running entirely on your machine. No API keys needed.
- **💾 Efficient Indexing**: Creates and caches a [FAISS](https://faiss.ai/) vector index for fast and efficient document retrieval.
- **💬 Interactive UI**: A simple and clean chat interface built with [Gradio](https://www.gradio.app/).
- **✅ Source Citations**: Each answer includes citations pointing to the source document and page number.
- **⚙️ Highly Configurable Paths**: All file paths (for documents, models, and index) are defined parametrically in `config.py` and are relative to the project root, making the project highly portable across different operating systems and local setups.

---

### How It Works

The application follows a standard RAG pipeline, divided into two main stages:

**1. Ingestion & Indexing (Offline)**
This happens the first time you run the app.

- **Load**: PDF documents in the `/docs` folder are loaded using `PyPDFLoader`.
- **Clean**: Text is cleaned to remove non-ASCII characters and extra whitespace.
- **Split**: The documents are split into smaller, overlapping chunks using `RecursiveCharacterTextSplitter`.
- **Embed & Store**: Each chunk is converted into a vector embedding using `all-MiniLM-L6-v2` and stored in a FAISS vector index. This index is saved to the `/faiss_index` directory so it doesn't need to be recreated on subsequent runs.

**2. Retrieval & Generation (Online)**
This happens every time you ask a question.

- **Retrieve**: Your question is embedded, and the FAISS index is searched to find the most relevant document chunks (the "context").
- **Augment**: The retrieved context and your original question are inserted into a prompt template.
- **Generate**: The complete prompt is sent to the local Gemma-2 LLM, which generates a concise answer based *only* on the provided context.

---

### Setup and Usage

Follow these steps to get the application running on your local machine.

**Prerequisites:**

- Python 3.10.x to 3.12.x (or specify the exact range you tested)
- Git
- Operating System: Tested on Windows 10/11, macOS, and Linux (Ubuntu 22.04+ recommended).

**1. Clone the Repository**

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
```

**2. Set Up Project Structure**
Create the necessary folders for your documents and the LLM model(s).

```bash
mkdir docs
mkdir models
mkdir faiss_index # This directory will be populated automatically on first run
```

**3. Add Your Files**

- Place your research papers (`.pdf` files) inside the newly created `docs` folder.
- Download the [Gemma-2-2B-IT GGUF model](https://huggingface.co/lmstudio-ai/gemma-2-2b-it-GGUF) (we recommend the `Q4_K_M` version for a good balance of quality and performance) and place the `.gguf` file inside the `models` folder.

**4. Install Dependencies**
It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

**5. Run the Application**

```bash
python app.py
```

The script will first create the FAISS index (this may take a few minutes depending on the number of documents). Once complete, it will launch a Gradio web server. Open the local URL provided in your terminal (e.g., `http://127.0.0.1:7860`) to start chatting with your documents!

---

### Technology Stack

##### Naive RAG:

- **Backend Framework**: [LangChain](https://www.langchain.com/)
- **LLM**: [Google Gemma-2](https://huggingface.co/google/gemma-2-2b-it) (via [Llama.cpp](https://github.com/ggerganov/llama.cpp))
- **Embedding Model**: [Sentence-Transformers](https://www.sbert.net/)
- **Vector Store**: [Facebook AI Similarity Search (FAISS)](https://faiss.ai/)
- **UI**: [Gradio](https://www.gradio.app/)
