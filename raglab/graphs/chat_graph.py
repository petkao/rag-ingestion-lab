"""The chatbot as a LangGraph graph: upload-to-ingest plus agentic RAG in one conversation.

                       ┌─(pending_pdf set)──▶ ingest ──▶ END        (runs ingest_graph as a subgraph)
    START ──▶ route ───┤
                       └─(a chat message)───▶ agent ◀──┐
                                                │      │
                                     tool_use?  ├─yes─▶ tools
                                                └─no──▶ END

* `route` is a conditional entry point: the same thread handles "here's a PDF" and "answer this".
* `agent` makes one Claude call (Anthropic SDK). If Claude asks for tools, `tools` runs them and
  loops back. This is the same loop as raglab/agent.py, but each step is a graph node.
* State is checkpointed per thread_id (InMemorySaver), which is what gives the chatbot memory.
  Swap in SqliteSaver/PostgresSaver to keep conversations across restarts.

Messages are kept in Anthropic's own format (role + content blocks) so they can be sent back to
the API unchanged, including thinking blocks.
"""

from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .. import llm
from ..agent import SYSTEM, TOOLS, search_tool_blocks
from ..store import list_documents
from .ingest_graph import ingest_graph

CHAT_SYSTEM = SYSTEM + """

The user can upload PDFs in this chat; each upload is ingested into the index and announced in \
the conversation. Use list_documents to see which PDFs are available. If nothing relevant has \
been uploaded, tell the user to upload a PDF."""

CHAT_TOOLS = TOOLS + [
    {
        "name": "list_documents",
        "description": "List the PDFs in the index, with page counts and how many text, table and image chunks each has.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    }
]


class ChatState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # Anthropic-format conversation, append-only
    events: Annotated[list[dict], operator.add]    # what happened (searches, ingests), for the UI
    pending_pdf: str | None                        # set by the UI on upload, cleared by `ingest`
    use_llm: bool                                  # Claude enrichment during ingestion


# ---- nodes ----------------------------------------------------------------------------------

def route(state: ChatState) -> str:
    return "ingest" if state.get("pending_pdf") else "agent"


def ingest(state: ChatState) -> ChatState:
    pdf = Path(state["pending_pdf"])
    # Invoked inside a node, so LangGraph treats it as a subgraph: with stream(subgraphs=True)
    # the UI receives parse/chunk/enrich/store progress events as they happen.
    result = ingest_graph.invoke({"pdf_path": str(pdf), "use_llm": state.get("use_llm", True)})

    if result.get("error"):
        reply = f"I couldn't ingest **{pdf.name}**: {result['error']}"
        event = {"kind": "ingest", "source": pdf.name, "ok": False, "error": result["error"]}
    else:
        s = result["stats"]
        c = s["chunks"]
        pages = max((e["page"] for e in result["elements"]), default=0)
        def n(count, word):
            return f"{count} {word}{'' if count == 1 else 's'}"
        reply = (f"I've added **{pdf.name}** to the knowledge base: {n(pages, 'page')} → "
                 f"{n(s['stored'], 'chunk')} ({c['text']} text, {n(c['table'], 'table')}, "
                 f"{n(c['image'], 'image')}). Ask me anything about it.")
        event = {"kind": "ingest", "source": pdf.name, "ok": True, "stats": s,
                 "report_path": result["report_path"]}

    # Recorded as a user/assistant pair so Claude knows about the upload in later turns.
    return {
        "messages": [
            {"role": "user", "content": f"(I uploaded the PDF '{pdf.name}'.)"},
            {"role": "assistant", "content": [{"type": "text", "text": reply}]},
        ],
        "events": [event],
        "pending_pdf": None,
    }


def agent(state: ChatState) -> ChatState:
    response = llm.create(messages=state["messages"], system=CHAT_SYSTEM, tools=CHAT_TOOLS, effort="medium")
    content = [b.model_dump(mode="json", exclude_none=True) for b in response.content]
    return {"messages": [{"role": "assistant", "content": content}]}


def tools(state: ChatState) -> ChatState:
    results, events = [], []
    for block in state["messages"][-1]["content"]:
        if block["type"] != "tool_use":
            continue
        try:
            if block["name"] == "search_document":
                content, hits = search_tool_blocks(block["input"])
                events.append({"kind": "search", "input": block["input"],
                               "hits": [h.__dict__ | {"citation": h.citation} for h in hits]})
            elif block["name"] == "list_documents":
                docs = list_documents()
                content = json.dumps(docs) if docs else "The index is empty."
                events.append({"kind": "list_documents", "count": len(docs)})
            else:
                raise ValueError(f"unknown tool {block['name']}")
            results.append({"type": "tool_result", "tool_use_id": block["id"], "content": content})
        except Exception as e:
            results.append({"type": "tool_result", "tool_use_id": block["id"],
                            "content": f"Error: {e}", "is_error": True})
    return {"messages": [{"role": "user", "content": results}], "events": events}


def after_agent(state: ChatState) -> str:
    last = state["messages"][-1]["content"]
    return "tools" if any(b["type"] == "tool_use" for b in last) else END


# ---- graph ----------------------------------------------------------------------------------

def build_chat_graph(checkpointer=None):
    g = StateGraph(ChatState)
    g.add_node("ingest", ingest)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_conditional_edges(START, route, ["ingest", "agent"])
    g.add_edge("ingest", END)
    g.add_conditional_edges("agent", after_agent, ["tools", END])
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or InMemorySaver())
