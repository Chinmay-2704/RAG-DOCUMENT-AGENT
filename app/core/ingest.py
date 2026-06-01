"""Document ingestion: load files, split into chunks, embed into FAISS."""
from __future__ import annotations

import os
import tempfile
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_file(path: str) -> List[Document]:
    """Pick the correct loader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext in (".txt", ".md"):
        return TextLoader(path, encoding="utf-8").load()
    if ext in (".docx",):
        return Docx2txtLoader(path).load()
    raise ValueError(f"Unsupported file type: {ext}")


def load_uploaded_file(uploaded_file) -> List[Document]:
    """Save a Streamlit UploadedFile to a temp path so loaders can read it."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        docs = load_file(tmp_path)
        for d in docs:
            d.metadata["source"] = uploaded_file.name
        return docs
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def split_documents(docs: List[Document]) -> List[Document]:
    """Recursive splitter preserves semantic boundaries (paragraphs > sentences)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
