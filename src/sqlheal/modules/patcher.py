"""Gap 1 -- AST-level localized patcher (counter to *Repair Drift*).

Whole-query regeneration (the CHASE-SQL-style fixer) asks the LLM to rewrite the entire
query from an error message, so clauses that were already correct get mutated. This
module instead

  1. *localises* the error to ONE clause from the SQLite error message
     (sqlglot AST lookup of columns / tables / functions / aggregates per clause, with a
     token-span fallback for syntax errors sqlglot itself cannot parse),
  2. *freezes* every other clause,
  3. asks the LLM to rewrite only the broken clause body,
  4. splices that body back with ``reconstruct.replace_clause_in_sql`` so the frozen
     clauses stay byte-identical, and verifies this before returning.

The patcher never executes SQL; the orchestrator re-runs the patched query in the sandbox.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional, Set

import sqlglot
from pydantic import BaseModel
from sqlglot import exp

from sqlheal.modules._common import LLMLike, TokenUsage, call_llm, extract_code_block
from sqlheal.parsing import extract_clause_table
from sqlheal.reconstruct import (
    CLAUSE_KEYWORD,
    CLAUSE_ORDER,
    ReconstructionError,
    SplitQuery,
    frozen_clauses_intact,
    mask_sql,
    replace_clause_in_sql,
    split_clauses,
)
from sqlheal.state import ClauseTable, ClauseType

__all__ = ["ASTPatcher", "PatchResult", "PatchStatus", "Localization", "locate_error_clause"]

REMOVE_SENTINEL = "<REMOVE>"
_OPTIONAL = {ClauseType.WHERE, ClauseType.GROUP_BY, ClauseType.HAVING, ClauseType.ORDER_BY, ClauseType.LIMIT}


class PatchStatus(str, Enum):
    PATCHED = "PATCHED"          # one clause rewritten, frozen clauses verified intact
    UNLOCALIZED = "UNLOCALIZED"  # could not map the error to a clause -> caller falls back
    UNSUPPORTED = "UNSUPPORTED"  # UNION / multiple SELECT / etc.
    REJECTED = "REJECTED"        # LLM output unusable after all attempts


class Localization(BaseModel):
    clause: ClauseType
    reason: str
    confidence: float = 0.9


class PatchResult(BaseModel):
    status: PatchStatus
    original_sql: str
    patched_sql: str                              # == original_sql unless PATCHED
    target_clause: Optional[ClauseType] = None
    localization_reason: str = ""
    broken_content: str = ""
    new_content: str = ""
    frozen_clauses: Dict[str, str] = {}
    frozen_intact: bool = True
    clause_table: Optional[ClauseTable] = None
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0
    error: Optional[str] = None

    @property
    def applied(self) -> bool:
        return self.status == PatchStatus.PATCHED


# --------------------------------------------------------------------------- #
# Error localisation
# --------------------------------------------------------------------------- #
_SKELETON = {
    ClauseType.SELECT: "SELECT {b}",
    ClauseType.FROM: "SELECT 1 FROM {b}",
    ClauseType.WHERE: "SELECT 1 WHERE {b}",
    ClauseType.GROUP_BY: "SELECT 1 GROUP BY {b}",
    ClauseType.HAVING: "SELECT 1 GROUP BY 1 HAVING {b}",
    ClauseType.ORDER_BY: "SELECT 1 ORDER BY {b}",
    ClauseType.LIMIT: "SELECT 1 LIMIT {b}",
}
_AGG_RE = re.compile(r"(?<![\w$])(COUNT|SUM|AVG|MIN|MAX|TOTAL|GROUP_CONCAT)\s*\(", re.IGNORECASE)
_COLUMN_ORDER = [ClauseType.SELECT, ClauseType.WHERE, ClauseType.GROUP_BY,
                 ClauseType.HAVING, ClauseType.ORDER_BY, ClauseType.FROM]
_TABLE_ORDER = [ClauseType.FROM, ClauseType.WHERE, ClauseType.SELECT, ClauseType.GROUP_BY,
                ClauseType.HAVING, ClauseType.ORDER_BY]
_KEYWORD_CLAUSE = {"SELECT": ClauseType.SELECT, "FROM": ClauseType.FROM, "WHERE": ClauseType.WHERE,
                   "GROUP": ClauseType.GROUP_BY, "HAVING": ClauseType.HAVING,
                   "ORDER": ClauseType.ORDER_BY, "LIMIT": ClauseType.LIMIT}


def _ident_re(name: str) -> "re.Pattern[str]":
    return re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])", re.IGNORECASE)


class _AstIndex:
    """Per-clause identifier index built from sqlglot ASTs (used when every clause parses)."""

    def __init__(self, trees: Dict[ClauseType, exp.Expression]) -> None:
        self.columns: Dict[ClauseType, Set[str]] = {}
        self.tables: Dict[ClauseType, Set[str]] = {}
        self.funcs: Dict[ClauseType, Set[str]] = {}
        self.aggs: Dict[ClauseType, bool] = {}
        for ct, tree in trees.items():
            self.columns[ct] = {c.name.lower() for c in tree.find_all(exp.Column)}
            self.tables[ct] = {t.name.lower() for t in tree.find_all(exp.Table)}
            self.funcs[ct] = {
                (f.name if isinstance(f, exp.Anonymous) else f.sql_name()).lower()
                for f in tree.find_all(exp.Func)
            }
            self.aggs[ct] = any(a.find_ancestor(exp.Select) is tree for a in tree.find_all(exp.AggFunc))

    @classmethod
    def build(cls, split: SplitQuery, dialect: str) -> Optional["_AstIndex"]:
        trees: Dict[ClauseType, exp.Expression] = {}
        for ct, body in split.clauses.items():
            try:
                trees[ct] = sqlglot.parse_one(_SKELETON[ct].replace("{b}", body), read=dialect)
            except Exception:
                return None
        return cls(trees)

    def has_column(self, ct: ClauseType, name: str) -> bool:
        return name.lower() in self.columns.get(ct, ())

    def has_table(self, ct: ClauseType, name: str) -> bool:
        return name.lower() in self.tables.get(ct, ())

    def has_func(self, ct: ClauseType, name: str) -> bool:
        return name.lower() in self.funcs.get(ct, ())

    def has_aggregate(self, ct: ClauseType) -> bool:
        return self.aggs.get(ct, False)


class _TextIndex:
    """Regex fallback over literal-masked clause text (works for unparseable queries)."""

    def __init__(self, split: SplitQuery) -> None:
        self.visible = {
            ct: mask_sql(body, hide_parens=False, hide_identifiers=False)
            for ct, body in split.clauses.items()
        }

    def _has(self, ct: ClauseType, pattern: "re.Pattern[str]") -> bool:
        return ct in self.visible and bool(pattern.search(self.visible[ct]))

    def has_column(self, ct: ClauseType, name: str) -> bool:
        return self._has(ct, _ident_re(name))

    has_table = has_column

    def has_func(self, ct: ClauseType, name: str) -> bool:
        return self._has(ct, re.compile(r"(?<![\w$])" + re.escape(name) + r"\s*\(", re.IGNORECASE))

    def has_aggregate(self, ct: ClauseType) -> bool:
        return self._has(ct, _AGG_RE)


def _clause_for_token(split: SplitQuery, token: str) -> Optional[ClauseType]:
    """Map the token named in a ``near "X": syntax error`` message to a clause."""
    tok = token.strip()
    if not tok:
        return None
    kw = _KEYWORD_CLAUSE.get(tok.upper())
    if kw is not None:
        # An unexpected clause keyword means the clause *before* it is malformed.
        previous = [c for c in CLAUSE_ORDER[: CLAUSE_ORDER.index(kw)] if c in split.clauses]
        if previous:
            return previous[-1]
        return kw if kw in split.clauses else None
    pattern = (r"(?<![\w$])" + re.escape(tok) + r"(?![\w$])") if re.fullmatch(r"\w+", tok) else re.escape(tok)
    masked = mask_sql(split.sql, hide_parens=False, hide_identifiers=False)
    match = re.search(pattern, masked, re.IGNORECASE) or re.search(pattern, split.sql, re.IGNORECASE)
    if not match:
        return None
    best: Optional[ClauseType] = None
    for ct, (start, _end) in split.spans.items():
        if start <= match.start() and (best is None or start > split.spans[best][0]):
            best = ct
    return best


def locate_error_clause(sql: str, error_message: str, dialect: str = "sqlite") -> Optional[Localization]:
    """Decide which single clause of ``sql`` caused the SQLite ``error_message``.

    Returns None when the error cannot be tied to one clause (the caller should fall
    back to whole-query repair). Handles: no such column / table / function, ambiguous
    column, misuse of aggregate, JOIN/ON errors, HAVING/GROUP BY/ORDER BY/LIMIT errors,
    sub-select shape errors, and syntax errors (via token position).
    """
    split = split_clauses(sql)
    msg = (error_message or "").strip()
    if not split.supported or not split.clauses or not msg:
        return None
    low = msg.lower()
    index = _AstIndex.build(split, dialect) or _TextIndex(split)
    present = [ct for ct in CLAUSE_ORDER if ct in split.clauses]

    def pick(predicate, order) -> Optional[ClauseType]:
        return next((ct for ct in order if ct in split.clauses and predicate(ct)), None)

    def done(ct: Optional[ClauseType], why: str, conf: float = 0.9) -> Optional[Localization]:
        return Localization(clause=ct, reason=why, confidence=conf) if ct else None

    m = re.search(r"(?:no such column|ambiguous column name):\s*(.+)$", msg, re.IGNORECASE)
    if m:
        col = m.group(1).strip().split(".")[-1].strip("\"`[]' ")
        ct = pick(lambda c: index.has_column(c, col), _COLUMN_ORDER)
        return done(ct, f"column '{col}' is referenced in {ct.value if ct else '?'}")

    m = re.search(r"no such table:\s*(.+)$", msg, re.IGNORECASE)
    if m:
        tbl = m.group(1).strip().split(".")[-1].strip("\"`[]' ")
        ct = pick(lambda c: index.has_table(c, tbl), _TABLE_ORDER)
        return done(ct, f"table '{tbl}' is referenced in {ct.value if ct else '?'}")

    m = re.search(r"(?:no such function:\s*|wrong number of arguments to function\s+)(\w+)", msg, re.IGNORECASE)
    if m:
        fn = m.group(1)
        ct = pick(lambda c: index.has_func(c, fn), CLAUSE_ORDER)
        return done(ct, f"function '{fn}' is used in {ct.value if ct else '?'}")

    if "aggregate functions are not allowed in the group by" in low:
        return done(ClauseType.GROUP_BY if ClauseType.GROUP_BY in split.clauses else None, "aggregate inside GROUP BY")
    if "misuse of" in low and "aggregate" in low:
        ct = pick(index.has_aggregate, [ClauseType.WHERE, ClauseType.GROUP_BY, ClauseType.ORDER_BY,
                                        ClauseType.FROM, ClauseType.SELECT])
        return done(ct, f"aggregate call used illegally in {ct.value if ct else '?'}", 0.85)

    if "having clause on a non-aggregate query" in low:
        return done(ClauseType.HAVING, "HAVING without aggregation")
    if "group by clause is required before having" in low:
        return done(ClauseType.GROUP_BY, "HAVING requires GROUP BY", 0.7)
    if "group by term" in low or "group by clause" in low:
        return done(ClauseType.GROUP_BY if ClauseType.GROUP_BY in split.clauses else None, "GROUP BY error")
    if "order by" in low:
        return done(ClauseType.ORDER_BY if ClauseType.ORDER_BY in split.clauses else None, "ORDER BY error")
    if "datatype mismatch" in low and ClauseType.LIMIT in split.clauses:
        return done(ClauseType.LIMIT, "LIMIT/OFFSET type error", 0.6)
    if re.search(r"\bjoin\b|on clause references|cannot join using", low):
        return done(ClauseType.FROM if ClauseType.FROM in split.clauses else None, "JOIN/ON error")
    if "sub-select returns" in low or "row value misused" in low:
        ct = pick(lambda c: "select" in (split.clauses[c] or "").lower(),
                  [ClauseType.WHERE, ClauseType.HAVING, ClauseType.SELECT, ClauseType.FROM])
        return done(ct, "sub-select shape mismatch", 0.7)

    m = re.search(r'near "(.*?)": syntax error', msg, re.IGNORECASE | re.DOTALL) \
        or re.search(r'unrecognized token:\s*"?(.*?)"?$', msg, re.IGNORECASE)
    if m:
        ct = _clause_for_token(split, m.group(1))
        return done(ct, f"syntax error at token {m.group(1)!r} falls in {ct.value if ct else '?'}", 0.8)
    if "incomplete input" in low and present:
        return done(present[-1], "statement ends prematurely; last clause is incomplete", 0.6)
    return None


# --------------------------------------------------------------------------- #
# Patcher
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a precise SQLite repair engine. A SQL query failed. Exactly ONE clause is broken; "
    "every other clause is FROZEN and will be re-attached verbatim by the system, so never "
    "rewrite, repeat, reorder or 'improve' them. Return ONLY the corrected body of the broken "
    "clause (no leading keyword, no explanation) inside a single ```sql code block. Use only "
    "tables and columns that exist in the schema, SQLite syntax, and keep the original intent "
    f"of the question. If the clause is optional and must be dropped, return {REMOVE_SENTINEL}."
)


class ASTPatcher:
    """Repair a failing query by rewriting only the clause that caused the error.

    Args:
        llm: object with ``generate(messages) -> {"content", "prompt_tokens", ...}``.
        dialect: sqlglot dialect (SQLite for BIRD).
        max_attempts: LLM tries per patch; a rejected output is fed back once per retry.
        validate_syntax: reject patches whose reconstructed query sqlglot cannot parse.
    """

    def __init__(self, llm: LLMLike, dialect: str = "sqlite", max_attempts: int = 2,
                 validate_syntax: bool = True) -> None:
        self.llm = llm
        self.dialect = dialect
        self.max_attempts = max(1, max_attempts)
        self.validate_syntax = validate_syntax

    def localize(self, sql: str, error_message: str) -> Optional[Localization]:
        """Public wrapper around :func:`locate_error_clause`."""
        return locate_error_clause(sql, error_message, self.dialect)

    # -- prompt ----------------------------------------------------------- #
    @staticmethod
    def _user_prompt(question: str, schema: str, evidence: str, split: SplitQuery,
                     target: ClauseType, error_message: str) -> str:
        frozen: List[str] = []
        if split.prefix:
            frozen.append(f"(prefix): {split.prefix}")
        frozen += [f"{CLAUSE_KEYWORD[ct]}: {body}" for ct in CLAUSE_ORDER
                   if ct != target and (body := split.clauses.get(ct))]
        kw = CLAUSE_KEYWORD[target]
        return "\n".join([
            f"Question: {question}",
            f"External knowledge: {evidence}" if evidence else "",
            f"Database schema:\n{schema}",
            "Frozen clauses (DO NOT change):\n" + ("\n".join(frozen) or "(none)"),
            f"Broken clause: {kw}",
            f"Current {kw} body: {split.clauses.get(target, '') or '(absent)'}",
            f"SQLite error: {error_message}",
            f"Return only the corrected body of the {kw} clause (without the word {kw}).",
        ]).replace("\n\n\n", "\n\n")

    @staticmethod
    def _extract_body(content: str, target: ClauseType) -> str:
        text = extract_code_block(content).strip().rstrip(";").strip()
        if text.upper() == REMOVE_SENTINEL:
            return ""
        sp = split_clauses(text)
        if sp.keywords_found and target in sp.clauses:  # model echoed keyword(s) / a full query
            return sp.clauses[target]
        return text

    # -- main entry --------------------------------------------------------- #
    def patch(self, question: str, schema: str, sql: str, error_message: str,
              evidence: str = "") -> PatchResult:
        """Return a patched query that differs from ``sql`` in exactly one clause."""
        usage = TokenUsage()
        base = dict(original_sql=sql, patched_sql=sql)

        def result(status: PatchStatus, **kw) -> PatchResult:
            return PatchResult(status=status, prompt_tokens=usage.prompt_tokens,
                               completion_tokens=usage.completion_tokens, total_calls=usage.calls,
                               **base, **kw)

        split = split_clauses(sql)
        if not split.supported:
            return result(PatchStatus.UNSUPPORTED, error=split.unsupported_reason)
        loc = self.localize(sql, error_message)
        if loc is None:
            return result(PatchStatus.UNLOCALIZED, error="error could not be tied to a single clause")

        target = loc.clause
        old_body = split.clauses.get(target, "")
        frozen_types = [ct for ct in CLAUSE_ORDER if ct in split.clauses and ct != target]
        frozen = {ct.value: split.clauses[ct] for ct in frozen_types}
        info = dict(target_clause=target, localization_reason=loc.reason, broken_content=old_body,
                    frozen_clauses=frozen)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(question, schema, evidence, split, target, error_message)},
        ]
        problem = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            content = call_llm(self.llm, messages, usage)
            new_body = self._extract_body(content, target)
            problem = None
            patched = ""
            if not new_body and target not in _OPTIONAL and content.strip().upper() != REMOVE_SENTINEL:
                problem = "empty clause body"
            elif not new_body and target not in _OPTIONAL:
                problem = f"{CLAUSE_KEYWORD[target]} cannot be removed"
            elif re.sub(r"\s+", " ", new_body).strip() == re.sub(r"\s+", " ", old_body).strip():
                problem = "the clause is identical to the broken one; it must change to fix the error"
            else:
                try:
                    patched = replace_clause_in_sql(sql, target, new_body, self.dialect,
                                                    normalize=False, validate=self.validate_syntax)
                except ReconstructionError as exc:
                    problem = str(exc)
            if problem is None and not frozen_clauses_intact(sql, patched, frozen_types):
                problem = "frozen clauses were modified"  # defensive; cannot happen via splice
            if problem is None:
                try:
                    table: Optional[ClauseTable] = extract_clause_table(patched, dialect=self.dialect)
                except Exception:
                    table = None
                return PatchResult(
                    status=PatchStatus.PATCHED, original_sql=sql, patched_sql=patched,
                    new_content=new_body, frozen_intact=True, clause_table=table, attempts=attempt,
                    prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                    total_calls=usage.calls, **info)
            messages += [
                {"role": "assistant", "content": content},
                {"role": "user", "content": (
                    f"Rejected: {problem}. Return ONLY the corrected body of the "
                    f"{CLAUSE_KEYWORD[target]} clause in a ```sql block.")},
            ]
        return result(PatchStatus.REJECTED, attempts=self.max_attempts, error=problem, **info)
