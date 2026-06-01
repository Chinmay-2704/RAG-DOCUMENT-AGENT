"""LangGraph agent: route -> retrieve -> judge -> (optional web) -> answer.

Each node appends a step to `trace` so the UI can show the agent's reasoning.
"""
from __future__ import annotations

from typing import List, Dict, Any, TypedDict, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.core.vectorstore import similarity_search
from app.core.web_search import web_search


class AgentState(TypedDict, total=False):
    query: str
    allow_web: bool
    rag_docs: List[Document]
    web_docs: List[Dict[str, Any]]
    sufficient: bool
    answer: str
    trace: List[Dict[str, Any]]


def _llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.2)


def _add_step(state: AgentState, name: str, detail: str) -> None:
    state.setdefault("trace", []).append({"step": name, "detail": detail})


# ---------- Nodes ----------

def retrieve_node(state: AgentState) -> AgentState:
    docs = similarity_search(state["query"])
    state["rag_docs"] = docs
    _add_step(
        state,
        "Retrieve (RAG)",
        f"Fetched {len(docs)} chunks from the knowledge base.",
    )
    return state


def judge_node(state: AgentState) -> AgentState:
    """Ask the LLM whether retrieved chunks are sufficient."""
    docs = state.get("rag_docs", [])
    if not docs:
        state["sufficient"] = False
        _add_step(state, "Judge sufficiency", "No chunks retrieved — insufficient.")
        return state

    context = "\n\n".join(d.page_content[:500] for d in docs)
    prompt = (
        "You are evaluating whether the CONTEXT is sufficient to answer the QUESTION. "
        "Reply with exactly 'YES' or 'NO'.\n\n"
        f"QUESTION: {state['query']}\n\nCONTEXT:\n{context}"
    )
    verdict = _llm().invoke([HumanMessage(content=prompt)]).content.strip().upper()
    state["sufficient"] = verdict.startswith("YES")
    _add_step(state, "Judge sufficiency", f"LLM verdict: {verdict}")
    return state


def web_node(state: AgentState) -> AgentState:
    if not state.get("allow_web"):
        state["web_docs"] = []
        _add_step(state, "Web search", "Skipped (web search disabled by user).")
        return state
    results = web_search(state["query"])
    state["web_docs"] = results
    _add_step(state, "Web search", f"Fetched {len(results)} web results via Tavily.")
    return state


def answer_node(state: AgentState) -> AgentState:
    rag_ctx = "\n\n".join(
        f"[Doc {i+1} | {d.metadata.get('source', '?')}]\n{d.page_content}"
        for i, d in enumerate(state.get("rag_docs", []))
    )
    web_ctx = "\n\n".join(
        f"[Web {i+1} | {r.get('url','')}]\n{r.get('content','')}"
        for i, r in enumerate(state.get("web_docs", []))
    )

    system = (
        "You are a helpful research assistant. Answer the user's question using the "
        "provided context. Cite sources inline like [Doc 1] or [Web 2]. If the context "
        "is empty or irrelevant, say so honestly."
    )
    user = (
        f"QUESTION: {state['query']}\n\n"
        f"KNOWLEDGE BASE CONTEXT:\n{rag_ctx or '(none)'}\n\n"
        f"WEB CONTEXT:\n{web_ctx or '(none)'}"
    )
    reply = _llm().invoke([SystemMessage(content=system), HumanMessage(content=user)])
    state["answer"] = reply.content
    _add_step(state, "Compose answer", "LLM produced final answer with citations.")
    return state


# ---------- Routing ----------

def _route_after_judge(state: AgentState) -> str:
    return "web" if not state.get("sufficient") else "answer"


# ---------- Graph builder ----------

_graph = None


def get_graph():
    global _graph
    if _graph is not None:
        return _graph

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("judge", judge_node)
    g.add_node("web", web_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "judge")
    g.add_conditional_edges("judge", _route_after_judge, {"web": "web", "answer": "answer"})
    g.add_edge("web", "answer")
    g.add_edge("answer", END)

    _graph = g.compile()
    return _graph


def run_agent(query: str, allow_web: bool = True) -> AgentState:
    """Convenience entry point used by the Streamlit UI."""
    graph = get_graph()
    initial: AgentState = {"query": query, "allow_web": allow_web, "trace": []}
    return graph.invoke(initial)
