# Fix Notes — "GROQ_API_KEY not in the env file"

## What was wrong

The old `app/config.py` did:

```python
load_dotenv(".env")
```

That's a **relative path**. It only works if your current working directory
is exactly the project root when you launch Streamlit. If you ran
`streamlit run main.py` from anywhere else (or Streamlit's reloader changed
the cwd), `.env` was never read, so `os.getenv("GROQ_API_KEY")` returned
`None` and the app reported the key as missing — even though your `.env`
file was right there.

## What changed

1. **`app/config.py`** now resolves `.env` to an **absolute path** based on
   the location of `config.py` itself:

   ```python
   PROJECT_ROOT = Path(__file__).resolve().parent.parent
   ENV_PATH = PROJECT_ROOT / ".env"
   load_dotenv(dotenv_path=ENV_PATH, override=False)
   ```

   It also strips stray quotes/whitespace from values (a common copy-paste bug).

2. **`main.py`** prepends the project root to `sys.path` so
   `from app.X import ...` always resolves.

3. **`VECTORSTORE_DIR`** is now an absolute path under the project root, so
   FAISS files don't end up scattered in random working directories.

4. The Streamlit sidebar error message now tells you the **exact file path**
   it looked for `.env` in, which makes mistakes obvious.

5. Hardcoded fallback API keys were removed from `config.py` (the originals
   were committed real keys — **rotate them immediately** in the Groq /
   Tavily dashboards if you haven't already).

## How to verify the fix

```bash
cd ragapp
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                  # then edit and paste your real keys
streamlit run main.py
```

You should see **"Groq LLM connected"** in the sidebar.

## Checklist if it still complains

1. Is the file literally named `.env` (not `.env.txt`)? On Windows, enable
   "File name extensions" in Explorer to confirm.
2. Is it in the **same folder as `main.py`**?
3. Does the line look like `GROQ_API_KEY=gsk_xxx` with **no spaces around `=`**
   and no `export ` prefix?
4. Did you **fully stop** Streamlit (Ctrl+C) and start it again? Hot-reload
   does not re-read `.env`.
5. Are you in the virtualenv where you installed `python-dotenv`? Run
   `pip show python-dotenv` to confirm.
