"""RAG AGENT: Claude decides what to search for, reads the results (text, tables AND images),
and answers with page citations.

Why an agent instead of "retrieve top-5, stuff into prompt"?
  * Claude can rewrite the user's question into better search queries, search several times,
    and filter by content type ("find the table with ...").
  * Image hits are returned as real image blocks inside the tool result, so Claude looks at the
    original chart rather than relying only on the caption written at ingestion time.

The loop below is written out by hand (instead of the SDK tool runner) so every step is visible.
"""

from __future__ import annotations

import json

from . import config, llm
from .retrieve import search

SYSTEM = """You answer questions about PDF documents that have been ingested into a search index.
The index holds three kinds of chunks: text passages, tables (Markdown), and images (with the \
image itself attached so you can read it).

Use the search_document tool to find evidence before answering. Search more than once when the \
question has several parts or the first results are weak; use content_types to target tables or \
images when the question is about figures, charts, or tabular numbers.

Answer only from what the search results show. Cite each fact with its citation in square \
brackets, e.g. [report.pdf p.3]. If the documents don't contain the answer, say so."""

TOOLS = [
    {
        "name": "search_document",
        "description": (
            "Semantic search over the ingested PDF chunks. Returns the most similar text passages, "
            "tables (as Markdown) and images (as the image plus its description), each with a "
            "citation and similarity score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, phrased as a statement or question."},
                "content_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["text", "table", "image"]},
                    "description": "Restrict results to these chunk types. Omit to search all.",
                },
                # strict tool schemas don't support minimum/maximum; the range is enforced in run_search_tool
                "k": {"type": "integer", "description": "Number of results, 1-10 (default 5)."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


def run_search_tool(args: dict, verbose: bool) -> list[dict]:
    return search_tool_blocks(args, verbose)[0]


def search_tool_blocks(args: dict, verbose: bool = False):
    """Run a search and build the tool_result content. Returns (blocks, hits)."""
    k = max(1, min(10, args.get("k") or config.DEFAULT_TOP_K))
    hits = search(args["query"], k=k, types=args.get("content_types"))
    if verbose:
        print(f"  🔎 search_document({json.dumps(args)})")
        for h in hits:
            print(f"     {h.score:.3f}  {h.type:<5}  {h.citation:<22}  {h.section[:40]}")
    if not hits:
        return [{"type": "text", "text": "No results."}], hits

    blocks: list[dict] = []
    for i, h in enumerate(hits, 1):
        header = f"[{i}] type={h.type} citation=[{h.citation}] section=\"{h.section}\" score={h.score}"
        if h.type == "table":
            body = f"Summary: {h.summary}\n\n{h.content}"
        else:
            body = h.content
        blocks.append({"type": "text", "text": f"{header}\n{body}"})
        if h.type == "image" and h.image_path:
            blocks.append(llm.image_block(h.image_path))  # Claude sees the actual figure
    return blocks, hits


class RagAgent:
    def __init__(self, verbose: bool = True, max_steps: int = 8):
        self.messages: list[dict] = []   # kept across turns -> multi-turn chat
        self.verbose = verbose
        self.max_steps = max_steps

    def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})
        for _ in range(self.max_steps):
            response = llm.create(messages=self.messages, system=SYSTEM, tools=TOOLS, effort="medium")
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return llm.text_of(response)

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    content = run_search_tool(block.input, self.verbose)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
                except Exception as e:  # report tool failures to Claude instead of crashing
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": f"Error: {e}", "is_error": True})
            # all results for this turn go back in ONE user message
            self.messages.append({"role": "user", "content": results})
        return "(stopped: too many tool steps)"
