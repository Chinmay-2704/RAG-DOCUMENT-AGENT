# 📚 RAG-Based AI Document Agent

A clean, end-to-end **Retrieval-Augmented Generation (RAG)** agent with an interactive **Streamlit** UI.
Upload your documents (PDF / DOCX / TXT / MD), ask questions, and get cited answers — with an optional
real-time **web search fallback** and a transparent **agent trace** that shows every step the AI took.

---

## ✨ Features

- 📥 **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- 🧠 **Local vector search** — FAISS + HuggingFace `all-MiniLM-L6-v2` embeddings (no cloud DB)
- 🤖 **Fast LLM** — Groq Llama 3.1 (free tier available)
- 🌐 **Optional web search** — Tavily fallback when documents aren't enough
- 🔍 **LangGraph agent** — retrieve → judge sufficiency → (web search) → answer
- 👁️ **Transparent trace** — see every reasoning step in the sidebar
- 💬 **Interactive Streamlit chat** with source citations

---

## 🚀 Quick Start

```bash
# 1. Clone / unzip the project, then:
cd ragapp

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# then edit .env and paste your keys

# 5. Run the app
streamlit run main.py
```

The app opens at <http://localhost:8501>.

---

## 🔑 Required setup (THINGS YOU MUST DO)

1. **Get a Groq API key** (required, free): https://console.groq.com → API Keys → create key.
2. **(Optional) Get a Tavily API key** for web search: https://tavily.com → free 1,000 searches/month.
3. Copy `.env.example` → `.env` and paste your keys:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxx
   TAVILY_API_KEY=tvly-xxxxxxxxxxx   # optional
   ```
4. Run `pip install -r requirements.txt` (first run downloads the embedding model ~90 MB).
5. Launch with `streamlit run main.py`.
6. Use the sidebar to upload documents → click **Ingest** → start asking questions.

---

## 🗂️ Project structure

```
ragapp/
├── main.py                    # Entry point
├── requirements.txt
├── .env.example
├── app/
│   ├── config.py              # Env vars + tunable constants
│   ├── streamlit_app.py       # UI
│   └── core/
│       ├── ingest.py          # File loaders + chunk splitter
│       ├── vectorstore.py     # FAISS wrapper (load / add / search)
│       ├── web_search.py      # Tavily client
│       └── agent.py           # LangGraph workflow
├── data/vectorstore/          # Persisted FAISS index (auto-created)
└── PROJECT_GUIDE.md           # Detailed code-walkthrough
```

See **PROJECT_GUIDE.md** for a full line-by-line explanation of every file and why each library is used.

---

## 🧩 Tech stack at a glance

| Layer       | Tool                                             | Why                                  |
| ----------- | ------------------------------------------------ | ------------------------------------ |
| UI          | Streamlit                                        | Fast Python UI, perfect for AI demos |
| Orchestration | LangGraph                                      | Explicit, debuggable agent flow      |
| LLM         | Groq (Llama 3.1)                                 | Free, very fast inference            |
| Embeddings  | HuggingFace `all-MiniLM-L6-v2`                   | Small, accurate, runs on CPU         |
| Vector DB   | FAISS                                            | Local, zero-config, fast             |
| Web search  | Tavily                                           | LLM-optimized search API             |
| Loaders     | pypdf, docx2txt                                  | Battle-tested document parsers       |

