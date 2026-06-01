"""Entry point — run with `streamlit run main.py`.

Ensures the project root is on sys.path so `from app.X import ...` works
no matter where Streamlit was launched from.
"""
import sys
from pathlib import Path

print("Python executable:", sys.executable)
print("Python version:", sys.version)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import runpy

if __name__ == "__main__":
    runpy.run_module("app.streamlit_app", run_name="__main__")
