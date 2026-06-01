"""Optional web search via Tavily."""
from __future__ import annotations

from typing import List, Dict

from app.config import TAVILY_API_KEY


def web_search(query: str, max_results: int = 4) -> List[Dict]:
    """Return a list of {title, url, content} dicts, or [] if Tavily isn't configured."""
    if not TAVILY_API_KEY:
        return []
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        resp = client.search(query=query, max_results=max_results, search_depth="basic")
        return resp.get("results", [])
    except Exception as e:  # noqa: BLE001
        return [{"title": "Web search error", "url": "", "content": str(e)}]
