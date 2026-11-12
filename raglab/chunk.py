"""Stage 2 - CHUNK: turn Elements into retrieval units.

Different content types need different chunking strategies:

  * Text   -> section-aware sliding window. A chunk never crosses a heading (topics stay
              clean), is capped at CHUNK_MAX_CHARS, and repeats CHUNK_OVERLAP_CHARS from the
              previous chunk so a sentence split at a boundary is still findable.
  * Tables -> one chunk per table, never split. Half a table without its header row is useless.
  * Images -> one chunk per image. The image itself can't be embedded by a text model, so its
              chunk is filled with a caption in Stage 3 (enrich).

Every chunk remembers its page(s) and section heading -> citations and metadata filters later.

Output: output/<doc>/chunks.json
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .models import Chunk, Element


def _split_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split on paragraph/sentence boundaries where possible, hard-split only as a last resort."""
    if len(text) <= max_chars:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # back off to the nearest paragraph break, else sentence end, else space
            for sep in ("\n\n", "\n", ". ", " "):
                cut = text.rfind(sep, start + max_chars // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        # don't start mid-word
        space = text.find(" ", start)
        if space != -1 and space < end:
            start = space + 1
    return [p for p in pieces if p]


def build_chunks(elements: list[Element], pdf_path: Path) -> list[Chunk]:
    source = pdf_path.name
    chunks: list[Chunk] = []
    section = "(untitled)"
    buffer: list[Element] = []  # text elements of the current section

    def flush_text():
        if not buffer:
            return
        joined = "\n\n".join(e.text for e in buffer)
        for i, piece in enumerate(
            _split_with_overlap(joined, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP_CHARS)
        ):
            chunks.append(
                Chunk(
                    id=f"{pdf_path.stem}:{buffer[0].id}:c{i}",
                    type="text",
                    source=source,
                    page_start=buffer[0].page,
                    page_end=buffer[-1].page,
                    section=section,
                    content=piece,
                )
            )
        buffer.clear()

    for el in elements:
        if el.type == "text":
            if el.is_heading:
                flush_text()          # a new heading closes the previous section's chunk
                section = el.text.replace("\n", " ")
                continue
            buffer.append(el)
        elif el.type == "table":
            chunks.append(
                Chunk(
                    id=f"{pdf_path.stem}:{el.id}",
                    type="table",
                    source=source,
                    page_start=el.page,
                    page_end=el.page,
                    section=section,
                    content=el.text,                # raw Markdown table (shown to the agent)
                    summary=el.nearby_text,         # caption for now; Stage 3 replaces it
                )
            )
        elif el.type == "image":
            chunks.append(
                Chunk(
                    id=f"{pdf_path.stem}:{el.id}",
                    type="image",
                    source=source,
                    page_start=el.page,
                    page_end=el.page,
                    section=section,
                    content=el.nearby_text,         # caption/context for now; Stage 3 adds a description
                    image_path=el.image_path,
                )
            )
    flush_text()

    out = config.doc_output_dir(pdf_path) / "chunks.json"
    out.write_text(json.dumps([c.to_dict() for c in chunks], indent=2))
    return chunks
