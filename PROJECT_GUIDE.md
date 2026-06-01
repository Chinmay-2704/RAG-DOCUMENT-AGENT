# 📖 Project Guide — RAG Document Agent

A complete walkthrough of the project: what every file does, how the pieces fit together,
and why each library was chosen. Read this top-to-bottom and you'll understand the whole codebase.

---

## 1. The big picture

This is a **RAG (Retrieval-Augmented Generation)** application. RAG means:

1. You give the AI a pile of documents.
2. When you ask a question, the system first **retrieves** the most relevant snippets from those documents.
3. Those snippets are stuffed into the prompt as **context** for the LLM.
4. The LLM **generates** an answer grounded in your documents instead of hallucinating.

On top of plain RAG, this project adds an **agent** (powered by LangGraph) that:

- Judges whether the retrieved chunks are *actually sufficient* to answer the question.
- If not, optionally falls back to a **live web search** (Tavily).
- Produces a final answer with **citations**.
- Exposes every step in a **trace panel** so you can see what the agent did.

### Architecture diagram

```
                  ┌──────────────────────┐
   User query ──▶ │   Streamlit UI       │
                  │  (streamlit_app.py)  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   LangGraph agent    │
                  │      (agent.py)      │
                  └──────────┬───────────┘
                             │
        ┌────────────┬───────┴────────┬───────────────┐
        ▼            ▼                ▼               ▼
    Retrieve      Judge            Web search       Compose
    (FAISS)      (Groq LLM)        (Tavily)         (Groq LLM)
```

---

## 2. File-by-file explanation

### 📄 `main.py`

```python
import runpy
if __name__ == "__main__":
    runpy.run_module("app.streamlit_app", run_name="__main__")
```

Single entry point. `streamlit run main.py` boots the Streamlit UI.
Using `runpy` lets us keep all real code inside the `app/` package while still having a
short top-level launcher.

---

### 📄 `app/config.py`

Loads environment variables from `.env` (via `python-dotenv`) and exposes constants used everywhere:

- `GROQ_API_KEY`, `TAVILY_API_KEY`, `GROQ_MODEL` — secrets / model choice.
- `EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"` — small, fast embedding model that
  runs on CPU and produces 384-dim vectors.
- `VECTORSTORE_DIR = "data/vectorstore"` — where FAISS persists the index.
- `CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 150` — chunking parameters. Larger chunks give more context
  but worse retrieval precision; the overlap ensures information at chunk boundaries isn't lost.
- `TOP_K = 4` — how many chunks to retrieve per query.

Centralising config in one file means tuning the system is a one-line change.

---

### 📄 `app/core/ingest.py`

Responsible for **getting documents into a uniform format**.

- `load_file(path)` — picks the right LangChain loader by file extension:
  - PDF → `PyPDFLoader` (uses `pypdf` under the hood; one Document per page).
  - TXT/MD → `TextLoader`.
  - DOCX → `Docx2txtLoader`.
- `load_uploaded_file(uploaded_file)` — Streamlit gives you an in-memory `UploadedFile`,
  but most loaders need a real file path. So we write to a `NamedTemporaryFile`, load it,
  then delete it. We also stamp the original filename into `metadata["source"]` so citations
  show the correct name.
- `split_documents(docs)` — uses `RecursiveCharacterTextSplitter`. It tries to split on
  paragraph breaks first (`"\n\n"`), then single newlines, then sentences (`". "`), then words.
  This keeps semantically related text together, which makes retrieval much better than naive
  fixed-size chunking.

---

### 📄 `app/core/vectorstore.py`

Thin wrapper around **FAISS**, Facebook AI's open-source similarity search library.

- `get_embeddings()` — lazily builds a `HuggingFaceEmbeddings` instance. The first call downloads
  `all-MiniLM-L6-v2` (~90 MB) and caches it under `~/.cache/huggingface/`. We set
  `normalize_embeddings=True` so cosine similarity becomes equivalent to dot-product (faster).
- `load_vectorstore()` — loads the persisted index from disk, or returns `None` if nothing has been
  ingested yet. `allow_dangerous_deserialization=True` is required because FAISS uses pickle;
  it's safe here because we only ever load files *we* wrote.
- `add_documents(chunks)` — appends to the existing index, or creates a new one. Persists to disk
  immediately so restarting the app doesn't lose your knowledge base.
- `reset_vectorstore()` — wipes the on-disk index (powers the "Clear" button).
- `similarity_search(query, k)` — embeds the query and returns the top-k most similar chunks.

**Why FAISS over Pinecone?** Pinecone is great but requires an account, an API key, and network
calls for every operation. FAISS runs locally, is free, and is more than fast enough for personal
knowledge bases (millions of vectors with no issue).

---

### 📄 `app/core/web_search.py`

Tiny wrapper around the Tavily Python SDK. Returns an empty list when `TAVILY_API_KEY` isn't
configured so the rest of the agent gracefully degrades to RAG-only mode.

Tavily is designed for LLMs: it returns clean, summarized snippets instead of raw HTML, which
saves tokens and improves answer quality.

---

### 📄 `app/core/agent.py` — the heart of the project

Built with **LangGraph**, which lets you express an agent as an explicit state machine:
nodes = functions, edges = transitions. Compared to plain LangChain "agents", LangGraph is
far easier to debug because the control flow is visible in code.

#### The state

```python
class AgentState(TypedDict, total=False):
    query: str
    allow_web: bool
    rag_docs: List[Document]
    web_docs: List[Dict]
    sufficient: bool
    answer: str
    trace: List[Dict]
```

Every node receives this dict, mutates it, and returns it. `trace` is what powers the
sidebar's transparency panel.

#### The four nodes

1. **`retrieve_node`** — runs `similarity_search()` and stores the chunks.
2. **`judge_node`** — asks the LLM "is this context sufficient to answer the question? YES/NO".
   This is the crucial step that decides whether we need web search. Without it, the agent would
   confidently answer from irrelevant chunks.
3. **`web_node`** — only runs if judge said NO *and* the user enabled web search. Calls Tavily.
4. **`answer_node`** — composes the final prompt (system instruction + RAG context + web context)
   and asks the LLM for a cited answer.

#### The graph

```python
g.set_entry_point("retrieve")
g.add_edge("retrieve", "judge")
g.add_conditional_edges("judge", _route_after_judge, {"web": "web", "answer": "answer"})
g.add_edge("web", "answer")
g.add_edge("answer", END)
```

Linear with one conditional branch. Easy to read, easy to extend (you could add a "rewrite query"
node, a re-ranker, multi-hop retrieval, etc.).

#### `run_agent(query, allow_web)`

The single function the UI calls. Compiles the graph once (cached in a module-level variable)
and invokes it with an initial state.

---

### 📄 `app/streamlit_app.py` — the UI

Streamlit is just Python, top-to-bottom. Key bits:

- **Custom CSS** (the `st.markdown(..., unsafe_allow_html=True)` block at the top) styles the
  trace cards and source cards to look polished without pulling in a real CSS framework.
- **`st.session_state`** persists the chat history, last trace, and last sources across reruns
  (Streamlit re-executes the whole script on every interaction).
- **Sidebar** holds configuration + the document uploader. Showing the chunk count gives the
  user immediate feedback that ingestion worked.
- **Two-column layout** (`col_chat`, `col_trace`) — chat on the left, agent trace + sources on
  the right. This is what makes the "transparency" feature obvious instead of buried.
- **`st.chat_input` + `st.chat_message`** — Streamlit's built-in chat primitives. No need to
  build a custom message list.

---

## 3. The full tech stack and *why*

| Library                   | Purpose                                          | Why this one                                                                 |
| ------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| **Streamlit**             | Web UI                                           | Pure Python, instant hot-reload, perfect for AI/data demos.                  |
| **python-dotenv**         | Load `.env` into env vars                        | Standard, zero-config secret management.                                     |
| **LangChain**             | Document loaders + text splitters + LLM adapters | Huge ecosystem of pre-built integrations; saves writing glue code.           |
| **langchain-community**   | PDF/DOCX/TXT loaders, FAISS integration          | Community-maintained adapters that "just work".                              |
| **langchain-text-splitters** | Recursive chunk splitter                      | Smarter than fixed-size splits; preserves semantic boundaries.               |
| **langchain-huggingface** | HuggingFace embedding wrapper                    | Lets us swap embedding models with a one-line config change.                 |
| **langchain-groq**        | Groq Chat model adapter                          | Cleanest way to call Groq from LangChain / LangGraph.                        |
| **LangGraph**             | Agent workflow as a state machine                | Explicit, debuggable, easy to extend — much nicer than legacy LC agents.     |
| **sentence-transformers** | The actual embedding model runtime               | Industry standard for sentence/passage embeddings.                           |
| **faiss-cpu**             | Vector similarity index                          | Free, local, blazing fast, no extra services needed.                         |
| **pypdf**                 | PDF parsing                                      | Pure Python, no system deps, used by LangChain's `PyPDFLoader`.              |
| **docx2txt**              | Word document parsing                            | Tiny dependency; sufficient for clean text extraction from `.docx`.          |
| **tavily-python**         | Web search                                       | LLM-optimized search API with a generous free tier.                          |
| **Groq (cloud)**          | LLM provider                                     | Free tier, extremely low latency (sub-second responses on Llama 3.1).        |

---

## 4. Things YOU need to do (checklist)

- [ ] Install **Python 3.10+**
- [ ] Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
- [ ] Install deps: `pip install -r requirements.txt`
- [ ] Sign up at **<https://console.groq.com>** → create an API key
- [ ] (Optional) Sign up at **<https://tavily.com>** for a web-search key
- [ ] Copy `.env.example` to `.env` and paste your keys:
      ```
      GROQ_API_KEY=gsk_...
      TAVILY_API_KEY=tvly-...    # optional
      ```
- [ ] First run: `streamlit run main.py` — it will download the embedding model (~90 MB) the
      first time. Subsequent starts are instant.
- [ ] Open the sidebar → upload PDFs/DOCX/TXT → click **Ingest documents**
- [ ] Ask questions in the chat and watch the **Agent Trace** panel update in real time

---

## 5. Ideas for extending the project

- Add **conversation memory** so follow-up questions ("what about the second one?") work.
- Swap FAISS for **Chroma** or **Qdrant** if you want metadata filtering.
- Add a **re-ranker** (e.g. `bge-reranker`) between retrieve and judge for better chunk ordering.
- Stream the LLM response token-by-token with `st.write_stream`.
- Add **user authentication** + per-user knowledge bases.
- Replace Groq with **Ollama** for a 100 % offline setup.

Have fun! 🚀
