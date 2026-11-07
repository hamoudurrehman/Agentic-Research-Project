"""Gap 3 -- guarded empty-set verifier (counter to *over-correction* of genuine empties).

Sits between "query returned 0 rows" and "prober edits a literal". It answers one
question: is an empty result the *right* answer here (``EXPECTED_EMPTY``: leave the query
alone) or a symptom of a wrong filter (``MISMATCH``: the prober may heal it)?

Decision order
  1. Deterministic rule (needs ``db_path``): if a referenced base table has zero rows, no
     edit can ever produce rows -> ``EXPECTED_EMPTY`` with no LLM call.
  2. LLM judge over Question + Schema + Query (+ optional DB evidence from the prober,
     e.g. "literal 'usa' not found; closest stored value 'USA'"), returning JSON.
  3. Policy: ``EXPECTED_EMPTY`` is only returned when the judge's confidence reaches
     ``min_confidence``; any LLM failure / unparsable reply yields ``default_verdict``.
"""
from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Union

from pydantic import BaseModel
from sqlglot import exp

from sqlheal.modules._common import LLMLike, TokenUsage, call_llm, extract_json_object, quote_ident
from sqlheal.parsing import parse_sql
from sqlheal.sandbox import SQLiteSandbox
from sqlheal.state import ExecutionStatus

__all__ = ["GuardedVerifier", "EmptyVerdict", "VerifierResult"]


class EmptyVerdict(str, Enum):
    EXPECTED_EMPTY = "EXPECTED_EMPTY"  # genuine empty answer: do not touch the query
    MISMATCH = "MISMATCH"              # empty because of a wrong filter/literal: healing allowed


class VerifierResult(BaseModel):
    verdict: EmptyVerdict
    confidence: float = 0.5
    reason: str = ""
    method: str = "llm"  # "rule:empty_table" | "llm" | "default"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0

    @property
    def allow_healing(self) -> bool:
        return self.verdict == EmptyVerdict.MISMATCH


SYSTEM_PROMPT = (
    "You audit Text-to-SQL results. The SQLite query below executed without error but returned "
    "ZERO rows. Decide whether zero rows is the CORRECT answer to the question.\n"
    "EXPECTED_EMPTY: the question itself can legitimately have no answer (the entity, year, "
    "range or condition it asks about plausibly does not exist in the data, or it asks for "
    "things that 'never' / 'did not' happen), and the query faithfully encodes that.\n"
    "MISMATCH: the question presupposes that matching data exists (it asks to list, name, count "
    "or rank things that should exist) and the query most likely misses it because of a wrong "
    "literal (case, whitespace, punctuation, abbreviation, value encoding) or a wrong filter.\n"
    "Do not treat emptiness as genuine just because the query returned nothing. "
    "If database evidence shows a stored value that differs from a query literal only by "
    "case/spacing/format, that strongly indicates MISMATCH.\n"
    'Reply with JSON only: {"verdict": "EXPECTED_EMPTY" | "MISMATCH", '
    '"confidence": <0..1>, "reason": "<one sentence>"}'
)


class GuardedVerifier:
    """Gate that blocks healing when an empty result is legitimate.

    Args:
        llm: object with ``generate(messages)``.
        sandbox: used only for the empty-table rule; defaults to a fresh ``SQLiteSandbox``.
        min_confidence: judge confidence needed to return ``EXPECTED_EMPTY``. Raising it
            makes the system heal more (less protection from over-correction); lowering it
            protects more but may block valid Gap-2 repairs. Tune on the DEV split only.
        default_verdict: verdict when the LLM fails or replies with something unparsable.
    """

    def __init__(self, llm: LLMLike, sandbox: Optional[SQLiteSandbox] = None, dialect: str = "sqlite",
                 min_confidence: float = 0.5, default_verdict: EmptyVerdict = EmptyVerdict.MISMATCH) -> None:
        self.llm = llm
        self.sandbox = sandbox or SQLiteSandbox()
        self.dialect = dialect
        self.min_confidence = min_confidence
        self.default_verdict = default_verdict

    # -- deterministic rule ----------------------------------------------- #
    def _empty_tables(self, db_path: Union[str, Path], sql: str) -> List[str]:
        tree = parse_sql(sql, dialect=self.dialect)
        if tree is None:
            return []
        ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
        names: List[str] = []
        for t in tree.find_all(exp.Table):
            if t.name and t.name.lower() not in ctes and t.name not in names:
                names.append(t.name)
        empty = []
        for name in names:
            res = self.sandbox.execute(db_path, f"SELECT 1 FROM {quote_ident(name)} LIMIT 1")
            if res.status == ExecutionStatus.EMPTY:
                empty.append(name)
        return empty

    # -- main entry --------------------------------------------------------- #
    def verify(self, question: str, schema: str, sql: str, db_path: Optional[Union[str, Path]] = None,
               evidence: str = "", probe_hints: Optional[Sequence[str]] = None) -> VerifierResult:
        """Classify an EMPTY result as ``EXPECTED_EMPTY`` or ``MISMATCH``.

        Args:
            question: the natural-language question.
            schema: schema text shown to the LLM.
            sql: the query that returned 0 rows.
            db_path: optional; enables the empty-table rule.
            evidence: optional BIRD external-knowledge string.
            probe_hints: optional human-readable DB observations (from ``LiteralProber``).
        """
        if db_path is not None:
            empty = self._empty_tables(db_path, sql)
            if empty:
                return VerifierResult(verdict=EmptyVerdict.EXPECTED_EMPTY, confidence=1.0,
                                      method="rule:empty_table",
                                      reason=f"table(s) {', '.join(empty)} contain no rows")

        usage = TokenUsage()
        user = "\n".join(filter(None, [
            f"Question: {question}",
            f"External knowledge: {evidence}" if evidence else "",
            f"Database schema:\n{schema}",
            f"SQL (returned 0 rows):\n{sql}",
            ("Database evidence:\n" + "\n".join(f"- {h}" for h in probe_hints)) if probe_hints else "",
        ]))
        try:
            content = call_llm(self.llm, [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": user}], usage)
        except Exception as exc:  # network / quota: fail to the configured default
            return self._default(usage, f"LLM call failed: {exc}")

        obj = extract_json_object(content) or {}
        raw = str(obj.get("verdict", "")).strip().upper()
        if raw not in EmptyVerdict.__members__:
            found = re.search(r"EXPECTED_EMPTY|MISMATCH", content.upper())
            raw = found.group(0) if found else ""
        if not raw:
            return self._default(usage, "unparsable verifier reply")
        try:
            confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        reason = str(obj.get("reason", "")).strip()
        verdict = EmptyVerdict(raw)
        if verdict == EmptyVerdict.EXPECTED_EMPTY and confidence < self.min_confidence:
            verdict = EmptyVerdict.MISMATCH
            reason = f"(low-confidence EXPECTED_EMPTY {confidence:.2f} < {self.min_confidence}) {reason}"
        return VerifierResult(verdict=verdict, confidence=confidence, reason=reason, method="llm",
                              prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                              total_calls=usage.calls)

    def _default(self, usage: TokenUsage, reason: str) -> VerifierResult:
        return VerifierResult(verdict=self.default_verdict, confidence=0.0, reason=reason, method="default",
                              prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                              total_calls=usage.calls)
