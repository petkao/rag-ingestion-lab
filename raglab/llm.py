"""Thin wrapper around the Anthropic SDK used by enrichment and the agent."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import anthropic

from . import config

# Server-side fallback: if Opus 5.5's safety classifier declines a request, the API re-runs it
# on a fallback model inside the same call. Opt-in beta; harmless for ordinary documents.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@lru_cache
def client() -> anthropic.Anthropic:
    # Reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment / .env
    return anthropic.Anthropic()


def image_block(path: str | Path) -> dict:
    p = config.PROJECT_ROOT / path
    media = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    data = base64.standard_b64encode(p.read_bytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def create(*, messages, system=None, tools=None, effort="medium", max_tokens=16000):
    """One Messages API call with the project defaults (adaptive thinking, fallbacks)."""
    kwargs = dict(
        model=config.CLAUDE_MODEL,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
        betas=[FALLBACK_BETA],
        fallbacks="default",
        messages=messages,
    )
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    response = client().beta.messages.create(**kwargs)
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise RuntimeError(f"Claude declined this request (category={category})")
    return response


def text_of(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text").strip()
