"""Shared NVIDIA NIM helper (OpenAI-compatible API).

All LLM / embedding demos in this repo talk to NVIDIA NIM through its
OpenAI-compatible endpoint. This module centralizes:

  - loading the .env file (python-dotenv)
  - reading model names + base URL from the environment
  - a graceful `available()` check
  - `chat(...)` and `embed(...)` wrappers

If NVIDIA_API_KEY is unset, `available()` returns False and demos should
fall back to a local/offline path instead of crashing.
"""
from __future__ import annotations

import os
from typing import List, Sequence

try:
    from dotenv import load_dotenv
    # Load .env from repo root (parent of this file's parent) and cwd.
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)
    load_dotenv(os.path.join(_root, ".env"))
    load_dotenv()  # also honor a .env in the current working directory
except Exception:
    pass


BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
CHAT_MODEL = os.environ.get("NIM_CHAT_MODEL", "meta/llama-3.1-8b-instruct")
EMBED_MODEL = os.environ.get("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")


def available() -> bool:
    """True if an NVIDIA_API_KEY is present (non-empty)."""
    return bool(os.environ.get("NVIDIA_API_KEY", "").strip())


def _client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["NVIDIA_API_KEY"], base_url=BASE_URL)


def chat(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 256,
) -> str:
    """Single-turn chat completion against the NIM chat model.

    Raises if no key is available; callers should guard with `available()`.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = _client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def embed(texts: Sequence[str], input_type: str = "query") -> List[List[float]]:
    """Embed a list of texts with the NIM embedding model.

    `input_type` is "query" or "passage" (nv-embedqa models are asymmetric).
    Raises if no key is available; callers should guard with `available()`.
    """
    if isinstance(texts, str):
        texts = [texts]
    client = _client()
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=list(texts),
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return [d.embedding for d in resp.data]
