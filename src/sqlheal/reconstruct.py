"""Clause-level SQL splitting and reconstruction for SQLHeal.

Why text-level splitting instead of re-printing the AST?
    Gap 1 (Repair Drift) is about *frozen clauses staying untouched*. A sqlglot
    round-trip (`expr.sql()`) silently normalises quoting, operators and spacing in
    every clause, which is itself a (small) form of drift, and it cannot even parse
    the broken queries (e.g. ``SELECCT``) that a repairer must handle. So this module
    splits a statement into its *top-level* clauses with a quote/comment/paren-aware
    scanner, swaps exactly one clause body, and re-joins the rest verbatim.
    sqlglot is used for what it is good at: validating the result, optional
    canonical formatting, and clause-level diffing.

Public API
    split_clauses(sql)                       -> SplitQuery
    reconstruct_sql_from_clauses(clauses)    -> str
    replace_clause_in_sql(sql, clause, body) -> str
    diff_clauses(sql_a, sql_b)               -> List[ClauseType]   (drift metric)
    mask_sql(sql)                            -> str                (scanner helper)

Limitations (by design, raised as ``ReconstructionError``): set operations
(UNION/INTERSECT/EXCEPT), WINDOW clauses and multiple top-level SELECTs are not
supported. A leading ``WITH`` prefix and ``OFFSET`` (kept inside the LIMIT body) are.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple, Union

from sqlglot import exp

from sqlheal.parsing import parse_sql
from sqlheal.state import ClauseType

__all__ = [
    "CLAUSE_ORDER",
    "CLAUSE_KEYWORD",
    "ReconstructionError",
    "SplitQuery",
    "mask_sql",
    "split_clauses",
    "reconstruct_sql_from_clauses",
    "replace_clause_in_sql",
    "frozen_clauses_intact",
    "diff_clauses",
]

CLAUSE_ORDER: List[ClauseType] = [
    ClauseType.SELECT,
    ClauseType.FROM,
    ClauseType.WHERE,
    ClauseType.GROUP_BY,
    ClauseType.HAVING,
    ClauseType.ORDER_BY,
    ClauseType.LIMIT,
]

CLAUSE_KEYWORD: Dict[ClauseType, str] = {
    ClauseType.SELECT: "SELECT",
    ClauseType.FROM: "FROM",
    ClauseType.WHERE: "WHERE",
    ClauseType.GROUP_BY: "GROUP BY",
    ClauseType.HAVING: "HAVING",
    ClauseType.ORDER_BY: "ORDER BY",
    ClauseType.LIMIT: "LIMIT",
}
_KEYWORD_TO_CLAUSE = {v: k for k, v in CLAUSE_KEYWORD.items()}
_UNSUPPORTED_KEYWORDS = {"UNION", "INTERSECT", "EXCEPT", "WINDOW"}

_KW_RE = re.compile(
    r"(?<![\w.$@:])"
    r"(SELECT|FROM|WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|UNION|INTERSECT|EXCEPT|WINDOW)"
    r"(?![\w$])",
    re.IGNORECASE,
)
_LEADING_KW_RE = {
    ct: re.compile(r"^\s*" + kw.replace(" ", r"\s+") + r"(?![\w$])\s*", re.IGNORECASE)
    for ct, kw in CLAUSE_KEYWORD.items()
}


class ReconstructionError(ValueError):
    """Raised when a statement cannot be split, or a clause cannot be safely swapped."""


# --------------------------------------------------------------------------- #
# Scanner
# --------------------------------------------------------------------------- #
def mask_sql(sql: str, *, hide_parens: bool = True, hide_identifiers: bool = True) -> str:
    """Return a same-length copy of ``sql`` where non-structural text is blanked.

    * Comments and single-quoted string literals are always blanked.
    * Double-quoted / back-quoted text is blanked when ``hide_identifiers`` is True.
    * Everything nested inside parentheses is blanked when ``hide_parens`` is True
      (the outermost parentheses themselves are kept), so a keyword regex run on the
      result only ever sees *top-level* keywords.

    Because the output has identical length, match offsets map 1:1 onto ``sql``.
    """
    n = len(sql)
    out: List[str] = []
    i = 0
    depth = 0
    while i < n:
        ch = sql[i]
        visible = depth == 0 or not hide_parens
        if ch == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
            continue
        if ch == "/" and sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(" " * (j - i))
            i = j
            continue
        if ch in ("'", '"', "`"):
            j = i + 1
            while j < n:
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:  # doubled quote escape
                        j += 2
                        continue
                    break
                j += 1
            end = min(j + 1, n)
            hide = ch == "'" or hide_identifiers or not visible
            out.append(" " * (end - i) if hide else sql[i:end])
            i = end
            continue
        if ch == "(":
            out.append("(" if visible else " ")
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
            out.append(")" if (depth == 0 or not hide_parens) else " ")
        else:
            out.append(ch if visible else " ")
        i += 1
    return "".join(out)


def _norm_kw(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.upper())


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class SplitQuery:
    """A statement broken into top-level clauses.

    Attributes:
        sql: the statement with surrounding whitespace / trailing ``;`` removed.
        prefix: text before the first SELECT (e.g. a ``WITH ... AS (...)`` block).
        clauses: clause bodies *without* their keyword, verbatim from the source.
        spans: ``(start, end)`` offsets of each body inside ``sql``.
        keywords_found: clause keywords that were actually present in the text.
        unsupported_reason: set when the statement uses a construct we refuse to edit.
    """

    sql: str
    prefix: str = ""
    clauses: Dict[ClauseType, str] = field(default_factory=dict)
    spans: Dict[ClauseType, Tuple[int, int]] = field(default_factory=dict)
    keywords_found: List[ClauseType] = field(default_factory=list)
    unsupported_reason: Optional[str] = None

    @property
    def supported(self) -> bool:
        return self.unsupported_reason is None


def split_clauses(sql: str) -> SplitQuery:
    """Split ``sql`` into its top-level clauses without needing it to be valid SQL.

    A missing/misspelled leading SELECT keyword (``SELECCT id FROM t``) is tolerated:
    the text before the first recognised keyword is treated as the SELECT body.
    """
    text = (sql or "").strip().rstrip(";").strip()
    result = SplitQuery(sql=text)
    masked = mask_sql(text)
    matches = list(_KW_RE.finditer(masked))
    if not matches:
        result.unsupported_reason = "no clause keywords found"
        return result

    first = matches[0]
    preamble = text[: first.start()].strip()
    if _norm_kw(first.group(1)) == "SELECT":
        result.prefix = preamble
    elif preamble:
        result.clauses[ClauseType.SELECT] = preamble
        result.spans[ClauseType.SELECT] = (0, first.start())

    for i, m in enumerate(matches):
        kw = _norm_kw(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if kw in _UNSUPPORTED_KEYWORDS:
            result.unsupported_reason = f"statement contains {kw}"
            continue
        ct = _KEYWORD_TO_CLAUSE[kw]
        if ct in result.clauses:
            result.unsupported_reason = result.unsupported_reason or f"multiple top-level {kw} clauses"
            continue
        result.clauses[ct] = text[start:end].strip()
        result.spans[ct] = (start, end)
        result.keywords_found.append(ct)
    return result


# --------------------------------------------------------------------------- #
# Clause normalisation / assembly
# --------------------------------------------------------------------------- #
def _coerce_clause_type(key: Union[str, ClauseType]) -> Union[ClauseType, str]:
    if isinstance(key, ClauseType):
        return key
    k = re.sub(r"[\s\-]+", "_", str(key).strip().upper())
    if k in ("JOIN", "JOINS"):
        return "JOIN"
    try:
        return ClauseType(k)
    except ValueError as exc:
        raise ReconstructionError(f"Unknown clause key: {key!r}") from exc


def _strip_leading_keyword(ct: ClauseType, body: str) -> str:
    return _LEADING_KW_RE[ct].sub("", body, count=1)


def _check_body(ct: ClauseType, body: str) -> None:
    """A clause body must not smuggle in other top-level clauses."""
    hit = _KW_RE.search(mask_sql(body))
    if hit:
        raise ReconstructionError(
            f"{CLAUSE_KEYWORD[ct]} body contains a top-level '{_norm_kw(hit.group(1))}' keyword; "
            "provide only the body of a single clause."
        )


def _prepare(clauses: Mapping[Union[str, ClauseType], Optional[str]]) -> Dict[ClauseType, str]:
    out: Dict[ClauseType, str] = {}
    joins: List[str] = []
    for key, body in clauses.items():
        if body is None:
            continue
        ct = _coerce_clause_type(key)
        if ct == "JOIN":
            joins.append(str(body).strip())
            continue
        text = _strip_leading_keyword(ct, str(body)).strip()  # type: ignore[arg-type]
        if text:
            out[ct] = text  # type: ignore[index]
    if joins:
        out[ClauseType.FROM] = f"{out.get(ClauseType.FROM, '')} {' '.join(joins)}".strip()
    for ct, body in out.items():
        _check_body(ct, body)
    return out


def _build(prefix: str, clauses: Dict[ClauseType, str]) -> str:
    if not clauses.get(ClauseType.SELECT, "").strip():
        raise ReconstructionError("SELECT clause is required and cannot be empty")
    parts: List[str] = [prefix.strip()] if prefix.strip() else []
    for ct in CLAUSE_ORDER:
        body = clauses.get(ct, "").strip()
        if body:
            parts.append(f"{CLAUSE_KEYWORD[ct]} {body}")
    return " ".join(parts)


def _finalise(sql: str, dialect: str, normalize: bool, validate: bool) -> str:
    expr = parse_sql(sql, dialect=dialect)
    if expr is None or not isinstance(expr, exp.Select):
        if validate:
            raise ReconstructionError(f"Reconstructed SQL is not a valid SELECT statement: {sql!r}")
        return sql
    return expr.sql(dialect=dialect, pretty=False) if normalize else sql


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def reconstruct_sql_from_clauses(
    clauses: Mapping[Union[str, ClauseType], Optional[str]],
    dialect: str = "sqlite",
    *,
    prefix: str = "",
    normalize: bool = True,
    validate: bool = True,
) -> str:
    """Assemble a full statement from a clause dictionary.

    Args:
        clauses: mapping of clause name -> body. Keys may be ``ClauseType`` members or
            strings (``"GROUP_BY"``, ``"group by"``, ...). An extra ``"JOIN"`` key is
            appended to FROM. Bodies may include their leading keyword (so the output of
            ``ClauseTable.to_dict()`` -- whose FROM body starts with ``FROM`` -- works
            unchanged). Empty / None bodies drop the clause.
        dialect: sqlglot dialect used for validation and normalisation.
        prefix: optional text placed before SELECT (e.g. a WITH block).
        normalize: if True, return sqlglot's canonical rendering; if False return the
            clause bodies verbatim, joined in canonical clause order.
        validate: raise ``ReconstructionError`` if the result is not a parseable SELECT.

    Returns:
        The statement without a trailing semicolon.
    """
    prepared = _prepare(clauses)
    return _finalise(_build(prefix, prepared), dialect, normalize, validate)


def replace_clause_in_sql(
    original_sql: str,
    target_clause: Union[ClauseType, str],
    new_content: str,
    dialect: str = "sqlite",
    *,
    normalize: bool = False,
    validate: bool = True,
) -> str:
    """Swap exactly one clause of ``original_sql``; every other clause stays byte-identical.

    ``new_content`` is the clause body, with or without its leading keyword. An empty
    body removes an optional clause (WHERE, GROUP BY, ...); a clause that is absent in
    the original is added in its canonical position.

    Raises:
        ReconstructionError: unsupported statement shape, body smuggling another clause,
            or (when ``validate``) a result sqlglot cannot parse.
    """
    ct = _coerce_clause_type(target_clause)
    if ct == "JOIN":
        raise ReconstructionError("JOIN conditions live inside the FROM clause; target FROM instead")
    split = split_clauses(original_sql)
    if not split.supported:
        raise ReconstructionError(f"Cannot edit statement: {split.unsupported_reason}")
    body = _strip_leading_keyword(ct, new_content or "").strip()  # type: ignore[arg-type]
    _check_body(ct, body)  # type: ignore[arg-type]
    clauses = dict(split.clauses)
    if body:
        clauses[ct] = body  # type: ignore[index]
    else:
        clauses.pop(ct, None)  # type: ignore[arg-type]
    return _finalise(_build(split.prefix, clauses), dialect, normalize, validate)


def frozen_clauses_intact(
    before_sql: str,
    after_sql: str,
    frozen: List[ClauseType],
) -> bool:
    """True iff every clause in ``frozen`` is identical (modulo whitespace) in both queries."""
    a, b = split_clauses(before_sql), split_clauses(after_sql)
    return all(_collapse(a.clauses.get(ct, "")) == _collapse(b.clauses.get(ct, "")) for ct in frozen)


def diff_clauses(sql_a: str, sql_b: str) -> List[ClauseType]:
    """Clauses whose body differs between two queries (whitespace-insensitive).

    This is the *Repair Drift* measurement for experiment E4/Gap 1: run it on
    (broken query, repaired query) and drop the clause that was actually broken; whatever
    remains was changed although it was correct.
    """
    a, b = split_clauses(sql_a), split_clauses(sql_b)
    return [
        ct
        for ct in CLAUSE_ORDER
        if _collapse(a.clauses.get(ct, "")) != _collapse(b.clauses.get(ct, ""))
    ]
