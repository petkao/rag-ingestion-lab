"""LangGraph version of the ingestion pipeline.

The same four stages as `raglab ingest`, but expressed as a graph so that:
  * each stage is a node whose output is checkpointed state you can inspect,
  * the UI can stream progress node by node (graph.stream(..., stream_mode="updates")),
  * the chatbot graph can run it as a subgraph when a user uploads a PDF.

    START → parse ─┬─(nothing found)──────────────────────────▶ END
                   └─▶ chunk → enrich → store ───────────────────▶ END

State holds plain dicts (not dataclasses) so any LangGraph checkpointer can serialise it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..chunk import build_chunks
from ..enrich import enrich_chunks
from ..models import Chunk, Element
from ..parse import parse_pdf
from ..report import write_report
from ..store import store_chunks


class IngestState(TypedDict, total=False):
    pdf_path: str            # input
    use_llm: bool            # input: Claude enrichment on/off
    elements: list[dict]     # after parse
    chunks: list[dict]       # after chunk / enrich
    stats: dict              # running counts, for the UI
    report_path: str         # after store
    error: str


def _count(items: list[dict]) -> dict:
    return {t: sum(1 for i in items if i["type"] == t) for t in ("text", "table", "image")}


def parse_node(state: IngestState) -> IngestState:
    elements = parse_pdf(Path(state["pdf_path"]))
    if not elements:
        return {"elements": [], "error": "No text, tables or images found. Is this a scanned PDF? It would need OCR first."}
    els = [e.to_dict() for e in elements]
    return {"elements": els, "stats": {"elements": _count(els)}}


def chunk_node(state: IngestState) -> IngestState:
    elements = [Element(**e) for e in state["elements"]]
    chunks = [c.to_dict() for c in build_chunks(elements, Path(state["pdf_path"]))]
    return {"chunks": chunks, "stats": {**state["stats"], "chunks": _count(chunks)}}


def enrich_node(state: IngestState) -> IngestState:
    chunks = [Chunk(**c) for c in state["chunks"]]
    enriched = enrich_chunks(chunks, Path(state["pdf_path"]), use_llm=state.get("use_llm", True))
    return {"chunks": [c.to_dict() for c in enriched]}


def store_node(state: IngestState) -> IngestState:
    chunks = [Chunk(**c) for c in state["chunks"]]
    n = store_chunks(chunks)
    report = write_report(Path(state["pdf_path"]))
    return {"stats": {**state["stats"], "stored": n}, "report_path": str(report)}


def after_parse(state: IngestState) -> str:
    return "chunk" if state.get("elements") else END


def build_ingest_graph():
    g = StateGraph(IngestState)
    g.add_node("parse", parse_node)
    g.add_node("chunk", chunk_node)
    g.add_node("enrich", enrich_node)
    g.add_node("store", store_node)
    g.add_edge(START, "parse")
    g.add_conditional_edges("parse", after_parse, ["chunk", END])
    g.add_edge("chunk", "enrich")
    g.add_edge("enrich", "store")
    g.add_edge("store", END)
    return g.compile()


ingest_graph = build_ingest_graph()
