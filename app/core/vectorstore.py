"""FAISS vector store wrapper — persisted locally on disk."""
from __future__ import annotations

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL, TOP_K, VECTORSTORE_DIR

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily build the embedding model — first call downloads the weights."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def load_vectorstore() -> Optional[FAISS]:
    """Return the persisted FAISS index, or None if it doesn't exist yet."""
    index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
    if not os.path.exists(index_path):
        return None
    return FAISS.load_local(
        VECTORSTORE_DIR,
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def add_documents(chunks: List[Document]) -> FAISS:
    """Add chunks to the persisted store (creating it if needed)."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    store = load_vectorstore()
    if store is None:
        store = FAISS.from_documents(chunks, get_embeddings())
    else:
        store.add_documents(chunks)
    store.save_local(VECTORSTORE_DIR)
    return store


def reset_vectorstore() -> None:
    """Wipe the on-disk index — used by the 'Clear knowledge base' button."""
    if not os.path.isdir(VECTORSTORE_DIR):
        return
    for name in os.listdir(VECTORSTORE_DIR):
        os.remove(os.path.join(VECTORSTORE_DIR, name))


def similarity_search(query: str, k: int = TOP_K) -> List[Document]:
    """Retrieve the k most relevant chunks for the query."""
    store = load_vectorstore()
    if store is None:
        return []
    return store.similarity_search(query, k=k)
