"""Streamlit UI for the RAG document agent."""
from __future__ import annotations

import streamlit as st

from app.config import GROQ_API_KEY, TAVILY_API_KEY, ENV_PATH
from app.core.agent import run_agent
from app.core.ingest import load_uploaded_file, split_documents
from app.core.vectorstore import add_documents, load_vectorstore, reset_vectorstore

st.set_page_config(
    page_title="RAG Document Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom styling ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }

    .trace-step {
        background-color: #1e293b;
        color: white;
        padding: 0.6rem 0.8rem;
        border-left: 3px solid #4f46e5;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
    }

    .source-card {
        background-color: #1e293b;
        color: white;
        padding: 0.6rem;
        border: 1px solid #475569;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }

    .source-card b {
        color: #60a5fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# ---------- Sidebar: config + ingestion ----------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    if not GROQ_API_KEY:
        st.error(f"GROQ_API_KEY missing. Add it to: {ENV_PATH}")
    else:
        st.success("Groq LLM connected")

    allow_web = st.toggle(
        "Enable web search",
        value=bool(TAVILY_API_KEY),
        disabled=not TAVILY_API_KEY,
        help="Requires TAVILY_API_KEY in .env",
    )
    if not TAVILY_API_KEY:
        st.caption("ℹ️ Add TAVILY_API_KEY to enable web fallback.")

    st.divider()
    st.markdown("### 📥 Upload documents")
    uploaded = st.file_uploader(
        "PDF, TXT, MD, or DOCX",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest documents", type="primary", use_container_width=True):
        with st.spinner("Embedding and indexing…"):
            total_chunks = 0
            for f in uploaded:
                docs = load_uploaded_file(f)
                chunks = split_documents(docs)
                add_documents(chunks)
                total_chunks += len(chunks)
        st.success(f"Indexed {len(uploaded)} file(s) → {total_chunks} chunks.")

    st.divider()
    store = load_vectorstore()
    chunk_count = store.index.ntotal if store else 0
    st.metric("Chunks in knowledge base", chunk_count)
    if st.button("🗑️ Clear knowledge base", use_container_width=True):
        reset_vectorstore()
        st.rerun()

# ---------- Main area ----------
st.markdown('<div class="main-title">📚 RAG Document Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask questions over your documents — with optional web fallback and a transparent agent trace.</div>',
    unsafe_allow_html=True,
)

col_chat, col_trace = st.columns([2, 1])

with col_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask anything about your documents…")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = run_agent(query, allow_web=allow_web)
                    answer = result.get("answer", "(no answer)")
                    st.session_state.last_trace = result.get("trace", [])
                    st.session_state.last_sources = [
                        {
                            "source": d.metadata.get("source", "?"),
                            "page": d.metadata.get("page"),
                            "snippet": d.page_content[:280],
                        }
                        for d in result.get("rag_docs", [])
                    ]
                except Exception as e:  # noqa: BLE001
                    answer = f"⚠️ Error: {e}"
                    st.session_state.last_trace = []
                    st.session_state.last_sources = []
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

with col_trace:
    st.markdown("### 🔍 Agent Trace")
    if not st.session_state.last_trace:
        st.caption("Run a query to see how the agent reasoned.")
    for step in st.session_state.last_trace:
        st.markdown(
            f'<div class="trace-step"><b>{step["step"]}</b><br/>{step["detail"]}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.last_sources:
        st.markdown("### 📄 Retrieved sources")
        for s in st.session_state.last_sources:
            page = f" (p.{s['page']})" if s.get("page") is not None else ""
            st.markdown(
                f'<div class="source-card"><b>{s["source"]}{page}</b><br/>{s["snippet"]}…</div>',
                unsafe_allow_html=True,
            )
