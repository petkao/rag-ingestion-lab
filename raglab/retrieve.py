"""RETRIEVER: query text -> embedding -> nearest chunks (optionally filtered by type/source/page)."""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .store import collection, embed


@dataclass
class Hit:
    id: str
    score: float          # cosine similarity, higher is better (1.0 = identical direction)
    type: str
    source: str
    page_start: int
    page_end: int
    section: str
    content: str
    summary: str
    image_path: str

    @property
    def citation(self) -> str:
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        return f"{self.source} {pages}"


def search(
    query: str,
    k: int = config.DEFAULT_TOP_K,
    types: list[str] | None = None,
    source: str | None = None,
) -> list[Hit]:
    filters = []
    if types:
        filters.append({"type": {"$in": types}})
    if source:
        filters.append({"source": source})
    where = None if not filters else filters[0] if len(filters) == 1 else {"$and": filters}

    res = collection().query(
        query_embeddings=embed([query]),
        n_results=k,
        where=where,
        include=["metadatas", "distances"],
    )
    hits = []
    for cid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(Hit(id=cid, score=round(1 - dist, 4), **{f: meta[f] for f in (
            "type", "source", "page_start", "page_end", "section", "content", "summary", "image_path")}))
    return hits
