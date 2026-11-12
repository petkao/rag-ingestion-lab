"""The two data shapes that flow through the pipeline.

    PDF ──parse──▶ Element[] ──chunk──▶ Chunk[] ──enrich──▶ Chunk[] (+captions/summaries)
                                                  ──embed+store──▶ vector DB rows

An *Element* is something the parser found on a page (a text block, a table, an image).
A *Chunk* is a unit of retrieval: the thing that gets one embedding vector and comes back
from a search. Keeping them separate is the core idea of ingestion: what the PDF *is*
(elements) is not the same as what you want to *retrieve* (chunks).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

ElementType = Literal["text", "table", "image"]


@dataclass
class Element:
    id: str
    type: ElementType
    page: int                     # 1-based page number
    bbox: list[float]             # [x0, y0, x1, y1] in PDF points, origin top-left
    text: str = ""                # text content; for tables, the table as Markdown
    is_heading: bool = False
    image_path: str = ""          # images only: PNG written to output/<doc>/images/
    nearby_text: str = ""         # text just above/below an image or table (often its caption)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    id: str
    type: ElementType
    source: str                   # PDF file name
    page_start: int
    page_end: int
    section: str                  # nearest heading above the chunk
    content: str                  # what the agent/user is shown: text, table Markdown, or image caption
    embed_text: str = ""          # what actually gets embedded (often content + context)
    image_path: str = ""
    summary: str = ""             # LLM-written description (tables and images)

    def to_dict(self) -> dict:
        return asdict(self)
