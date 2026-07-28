# chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config


def split_documents(documents):
    """Split documents into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ Loaded {len(documents)} pages")
    print(f"✅ Created {len(chunks)} chunks")

    return chunks