"""Small helpers shared by the SQLHeal modules (LLM protocol, token accounting, quoting)."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel


class LLMLike(Protocol):
    """Anything with the ``LLMClient.generate`` contract (lets tests inject a stub)."""

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]: ...


class TokenUsage(BaseModel):
    """Accumulated LLM usage; mirrors the fields tracked by ``QueryTurn``."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, response: Dict[str, Any]) -> None:
        self.prompt_tokens += int(response.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(response.get("completion_tokens", 0) or 0)
        self.calls += 1


def call_llm(llm: LLMLike, messages: List[Dict[str, str]], usage: TokenUsage) -> str:
    """Call the LLM, record token usage, return the text content."""
    response = llm.generate(messages)
    usage.add(response)
    return str(response.get("content", "") or "")


_FENCE_RE = re.compile(r"```[ \t]*(?:sql|sqlite|json)?[ \t]*\r?\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_block(text: str) -> str:
    """Return the first fenced code block, or the stripped text if there is none."""
    match = _FENCE_RE.search(text or "")
    return (match.group(1) if match else (text or "")).strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first JSON object embedded in an LLM reply."""
    decoder = json.JSONDecoder()
    for source in (extract_code_block(text), text or ""):
        for match in re.finditer(r"\{", source):
            try:
                obj, _ = decoder.raw_decode(source[match.start():])
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def quote_ident(name: str) -> str:
    """Quote an SQLite identifier (callers must whitelist names against PRAGMA output first)."""
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    """Quote an SQLite string literal."""
    return "'" + value.replace("'", "''") + "'"


def like_escape(value: str) -> str:
    """Escape LIKE wildcards; pair with ``ESCAPE '\\'``."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
