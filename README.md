# 🎓 ScholarRAG

> AI-Powered Research Paper Explainer & Research Assistant using Retrieval-Augmented Generation (RAG)

ScholarRAG is a fully local AI-powered research assistant that enables users to chat with research papers using natural language. The system extracts information from PDF documents, creates semantic embeddings, stores them in a FAISS vector database, and uses a local Large Language Model (Gemma 2) to answer questions grounded in the uploaded research papers.

---

## ✨ Features

- 📄 Research Paper Question Answering
- 🤖 Local LLM Inference (Gemma 2 GGUF)
- 🔍 Semantic Search using FAISS
- 🧠 Sentence Transformer Embeddings
- 📚 Retrieval-Augmented Generation (RAG)
- 📑 Automatic PDF Parsing
- 📖 Source Citation with Page Numbers
- 💻 Fully Offline (No OpenAI API Required)
- 🏗️ Modular Architecture

---

## 🏗️ Project Architecture

```
                ┌───────────────┐
                │   PDF Files   │
                └──────┬────────┘
                       │
                PDF Loader
                       │
                Text Cleaning
                       │
                Document Chunking
                       │
         SentenceTransformer Embeddings
                       │
                FAISS Vector Store
                       │
              Similarity Retrieval
                       │
               Gemma 2 (LLM)
                       │
              Natural Language Answer
```

---

## 📂 Current Project Structure

```
ScholarRAG
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── src
│   ├── llm
│   │   └── model.py
│   │
│   ├── rag
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   │
│   ├── services
│   │   ├── indexing_service.py
│   │   ├── qa_service.py
│   │   └── upload_service.py
│   │
│   ├── ui
│   │   └── upload.py
│   │
│   └── utils
│       └── helpers.py
│
├── uploads/
├── faiss_index/
├── models/
└── docs/
```

---

## ⚙️ Tech Stack

### Backend

- Python
- LangChain
- LlamaCpp
- Gradio

### Vector Database

- FAISS

### Embedding Model

- sentence-transformers/all-MiniLM-L6-v2

### Local LLM

- Gemma 2 2B Instruct (GGUF)

### Document Processing

- PyPDFLoader
- RecursiveCharacterTextSplitter

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Blvackk/ScholarRAG.git

cd ScholarRAG
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Model Setup

Download a GGUF model (Gemma 2 Instruct recommended).

Place it inside

```
models/
```

Update the model path inside

```
config.py
```

---

## ▶️ Run

```bash
python app.py
```

The Gradio interface will launch locally.

---

## 🧠 Current Workflow

```
PDF
 ↓

Load PDF

 ↓

Clean Text

 ↓

Chunk Documents

 ↓

Generate Embeddings

 ↓

Create FAISS Index

 ↓

Retrieve Relevant Chunks

 ↓

Gemma 2

 ↓

Answer with Sources
```

---

## 📌 Current Capabilities

✔ Chat with research papers

✔ Semantic document retrieval

✔ Local inference (offline)

✔ Source citations

✔ Modular RAG pipeline

✔ FAISS indexing

✔ Clean project architecture

---

## 🚧 Roadmap

- [ ] Drag & Drop PDF Upload
- [ ] Incremental FAISS Indexing
- [ ] Multi-document Support
- [ ] Research Paper Summarization
- [ ] Citation Extraction
- [ ] Export Chat as PDF
- [ ] Chat History
- [ ] User Authentication
- [ ] Docker Support
- [ ] REST API
- [ ] Cloud Deployment

---

## 📸 Demo

Coming Soon

---

## 🤝 Contributing

Contributions are welcome!

Feel free to fork the project and submit a Pull Request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Pratik Lagishetty**

GitHub: https://github.com/Blvackk
