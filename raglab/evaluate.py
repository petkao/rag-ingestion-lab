"""Measure retrieval quality: does the right chunk come back for known questions?

Each eval question names where its answer lives (page + chunk type). We report:
  hit@k  share of questions whose answer chunk appears anywhere in the top k
  MRR    mean reciprocal rank: 1.0 if always ranked first, 0.5 if always second, ...

Change a knob in config.py (chunk size, overlap, enrichment on/off), re-ingest, re-run this,
and you will see the effect of each ingestion decision in numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from .retrieve import search


def evaluate(questions_path: Path, k: int = 5) -> dict:
    questions = json.loads(questions_path.read_text())
    hits, rr = 0, 0.0
    print(f"{'rank':>4}  {'type':<5} {'page':>4}  question")
    for q in questions:
        results = search(q["question"], k=k)
        rank = next(
            (i for i, h in enumerate(results, 1)
             if h.type == q["expect_type"] and h.page_start <= q["expect_page"] <= h.page_end),
            None,
        )
        if rank:
            hits += 1
            rr += 1 / rank
        print(f"{rank or '-':>4}  {q['expect_type']:<5} {q['expect_page']:>4}  {q['question']}")
    n = len(questions)
    summary = {"questions": n, f"hit@{k}": round(hits / n, 3), "mrr": round(rr / n, 3)}
    print(json.dumps(summary))
    return summary
