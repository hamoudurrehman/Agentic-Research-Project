"""Gap 2 -- cardinality-aware literal prober (counter to *silent empty results*).

A query that executes but returns 0 rows is often caused by a literal that does not match
how the value is stored (``'usa'`` vs ``'USA'``, ``'New York '`` vs ``'New York'``,
``'Mens'`` vs ``"Men's"``). Given an EMPTY result this module

  1. finds string-literal predicates (``=``, ``LIKE``, ``IN``) in the top-level WHERE / HAVING,
  2. checks, with read-only probes through the sandbox, which literals match *no* stored value,
  3. for each such literal gathers real candidate values from the database
     (low-cardinality columns are enumerated; high-cardinality ones use a probe ladder:
     case/trim-insensitive -> punctuation-squashed -> substring -> token LIKE),
  4. picks a candidate (deterministically if it differs only by case/spacing/punctuation,
     otherwise by asking the LLM, which must answer with a candidate verbatim or null),
  5. rewrites only that literal (WHERE/HAVING body only; other clauses stay verbatim) and
     re-executes. By default the edit is kept only if it yields rows (``require_rows``).

An optional ``GuardedVerifier`` is consulted after the probes and before any edit (Gap 3).
All probe SQL is built from identifiers whitelisted against ``PRAGMA table_info`` and from
escaped literals, and is executed through the read-only sandbox.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import sqlglot
from pydantic import BaseModel
from sqlglot import exp

from sqlheal.modules._common import (
    LLMLike, TokenUsage, call_llm, extract_json_object, like_escape, quote_ident, quote_literal,
)
from sqlheal.modules.verifier import EmptyVerdict, GuardedVerifier
from sqlheal.parsing import parse_sql
from sqlheal.reconstruct import ReconstructionError, replace_clause_in_sql, split_clauses
from sqlheal.sandbox import SQLiteSandbox
from sqlheal.state import ClauseType, ExecutionStatus, SandboxResult

__all__ = ["LiteralProber", "ProbeRecord", "ProbeResult", "PredicateKind"]

_SQUASH_CHARS = [" ", "-", "_", ".", ",", "'"]
_WRAPPERS = {exp.Lower: "LOWER", exp.Upper: "UPPER", exp.Trim: "TRIM"}


class PredicateKind(str, Enum):
    EQ = "EQ"
    LIKE = "LIKE"
    IN = "IN"


class ProbeRecord(BaseModel):
    """Everything the prober learned about one literal (for logs / failure analysis)."""

    clause: ClauseType
    operator: PredicateKind
    table: str
    column: str
    literal: str
    literal_exists: bool
    cardinality: Optional[int] = None
    strategy: str = "exists"          # exists | enumerate | ci_trim | squash | contains | tokens_and | token_longest | none
    probe_sqls: List[str] = []
    candidates: List[str] = []
    chosen: Optional[str] = None
    method: str = "none"              # none | fastpath | llm
    final_literal: Optional[str] = None


class ProbeResult(BaseModel):
    original_sql: str
    healed_sql: str                    # == original_sql when nothing was kept
    healed: bool = False
    candidate_sql: Optional[str] = None  # the edited query even if it was rejected (debugging)
    healed_status: Optional[ExecutionStatus] = None
    reason: str = ""
    probes: List[ProbeRecord] = []
    verdict: Optional[EmptyVerdict] = None
    verdict_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_calls: int = 0
    probe_queries: int = 0


@dataclass
class _Ref:
    """A string literal compared against a column, with handles needed to edit it."""

    clause: ClauseType
    kind: PredicateKind
    node: exp.Expression               # the Literal (or quoted-identifier-as-string) node
    column: exp.Column
    wrappers: List[str]                # innermost first, e.g. ["TRIM", "LOWER"]
    literal: str
    token_text: str                    # how the literal is rendered inside the clause text
    tables: List[Tuple[str, str]] = field(default_factory=list)  # (table, actual column name)
    record: Optional[ProbeRecord] = None


def _squash(text: str) -> str:
    return re.sub(r"[\s\-_.,'’\"/]+", "", text.casefold())


def _tokens(text: str) -> List[str]:
    return sorted({t for t in re.findall(r"[^\W_]{3,}", text.casefold())}, key=len, reverse=True)


class LiteralProber:
    """Heal EMPTY queries whose string literals do not match stored values.

    Args:
        llm: object with ``generate(messages)``.
        sandbox: read-only sandbox used for every probe and re-execution.
        max_candidates: candidates kept per literal for high-cardinality columns.
        low_cardinality_threshold: columns with at most this many distinct values are
            enumerated in full instead of pattern-probed (this is the "cardinality-aware" part).
        max_literals: cap on how many missing literals are processed per query.
        require_rows: keep an edit only if the edited query returns rows.
    """

    def __init__(self, llm: LLMLike, sandbox: Optional[SQLiteSandbox] = None, dialect: str = "sqlite",
                 max_candidates: int = 8, low_cardinality_threshold: int = 25, max_literals: int = 6,
                 require_rows: bool = True) -> None:
        self.llm = llm
        self.sandbox = sandbox or SQLiteSandbox()
        self.dialect = dialect
        self.max_candidates = max_candidates
        self.low_cardinality_threshold = low_cardinality_threshold
        self.max_literals = max_literals
        self.require_rows = require_rows
        self._col_cache: Dict[Tuple[str, str], Dict[str, str]] = {}
        self._probe_count = 0

    # ------------------------------------------------------------------ #
    # sandbox plumbing
    # ------------------------------------------------------------------ #
    def _run(self, db_path: Union[str, Path], sql: str) -> SandboxResult:
        self._probe_count += 1
        return self.sandbox.execute(db_path, sql)

    def _columns(self, db_path: Union[str, Path], table: str) -> Dict[str, str]:
        """lower-case column name -> real column name, via PRAGMA (cached)."""
        key = (str(db_path), table.lower())
        if key not in self._col_cache:
            res = self._run(db_path, f"PRAGMA table_info({quote_ident(table)})")
            self._col_cache[key] = {str(r[1]).lower(): str(r[1]) for r in (res.rows or [])}
        return self._col_cache[key]

    # ------------------------------------------------------------------ #
    # predicate discovery
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unwrap(node: exp.Expression) -> Optional[Tuple[exp.Column, List[str]]]:
        wrappers: List[str] = []
        while True:
            if isinstance(node, exp.Column):
                return node, list(reversed(wrappers))
            if isinstance(node, exp.Paren):
                node = node.this
                continue
            name = next((v for k, v in _WRAPPERS.items() if isinstance(node, k)), None)
            if name is None:
                return None
            wrappers.append(name)
            node = node.this

    def _string_literal(self, node: Optional[exp.Expression], known_cols: set) -> Optional[str]:
        if isinstance(node, exp.Literal) and node.is_string:
            return str(node.this)
        # SQLite quirk: "Male" is a string when no column has that name.
        if (isinstance(node, exp.Column) and not node.table and node.this is not None
                and node.this.args.get("quoted") and node.name.lower() not in known_cols):
            return node.name
        return None

    def _collect_refs(self, db_path, sql: str) -> Tuple[Optional[Dict[ClauseType, exp.Expression]], List[_Ref], str]:
        split = split_clauses(sql)
        if not split.supported:
            return None, [], f"unsupported statement: {split.unsupported_reason}"
        tree = parse_sql(sql, dialect=self.dialect)
        if tree is None:
            return None, [], "query does not parse"
        alias_map: Dict[str, str] = {}
        query_tables: List[str] = []
        for t in tree.find_all(exp.Table):
            if t.name:
                alias_map[(t.alias or t.name).lower()] = t.name
                alias_map.setdefault(t.name.lower(), t.name)
                if t.name not in query_tables:
                    query_tables.append(t.name)
        known_cols = {c for t in query_tables for c in self._columns(db_path, t)}

        conds: Dict[ClauseType, exp.Expression] = {}
        skeletons = {ClauseType.WHERE: "SELECT 1 WHERE {b}", ClauseType.HAVING: "SELECT 1 GROUP BY 1 HAVING {b}"}
        for ct, skel in skeletons.items():
            body = split.clauses.get(ct)
            if not body:
                continue
            try:
                parsed = sqlglot.parse_one(skel.replace("{b}", body), read=self.dialect)
                conds[ct] = parsed.args["where" if ct == ClauseType.WHERE else "having"].this
            except Exception:
                continue

        refs: List[_Ref] = []
        for ct, cond in conds.items():
            for pred in cond.find_all(exp.EQ, exp.Like, exp.ILike, exp.In):
                if isinstance(pred, exp.In):
                    pairs = [(pred.this, lit) for lit in (pred.args.get("expressions") or [])]
                    kind = PredicateKind.IN
                elif isinstance(pred, (exp.Like, exp.ILike)):
                    pairs, kind = [(pred.this, pred.expression)], PredicateKind.LIKE
                else:
                    pairs, kind = [(pred.this, pred.expression), (pred.expression, pred.this)], PredicateKind.EQ
                for col_side, lit_side in pairs:
                    unwrapped = self._unwrap(col_side)
                    text = self._string_literal(lit_side, known_cols)
                    if unwrapped is None or text is None:
                        continue
                    column, wrappers = unwrapped
                    if column.name.lower() not in known_cols and not column.table:
                        continue
                    if column.table:
                        names = [alias_map[column.table.lower()]] if column.table.lower() in alias_map else []
                    else:
                        names = query_tables
                    resolved = [(t, self._columns(db_path, t)[column.name.lower()])
                                for t in names if column.name.lower() in self._columns(db_path, t)]
                    if not resolved:
                        continue
                    refs.append(_Ref(clause=ct, kind=kind, node=lit_side, column=column, wrappers=wrappers,
                                     literal=text, token_text=lit_side.sql(dialect=self.dialect),
                                     tables=resolved))
                    break
        self._conds = conds
        return conds, refs, ""

    # ------------------------------------------------------------------ #
    # probing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wrap(expr_sql: str, wrappers: Sequence[str]) -> str:
        for w in wrappers:
            expr_sql = f"{w}({expr_sql})"
        return expr_sql

    @staticmethod
    def _core(ref: _Ref) -> str:
        text = ref.literal.strip("%") if ref.kind == PredicateKind.LIKE else ref.literal
        return text.strip()

    def _exists(self, db_path, table: str, column: str, ref: _Ref) -> bool:
        col = self._wrap(quote_ident(column), ref.wrappers)
        op = "LIKE" if ref.kind == PredicateKind.LIKE else "="
        res = self._run(db_path, f"SELECT 1 FROM {quote_ident(table)} WHERE {col} {op} "
                                 f"{quote_literal(ref.literal)} LIMIT 1")
        return res.status != ExecutionStatus.EMPTY  # ERROR -> assume it exists: do not touch

    def _gather(self, db_path, table: str, column: str, ref: _Ref) -> Tuple[List[str], str, List[str], Optional[int]]:
        """Return (candidates, strategy, probe_sqls, cardinality) for one literal."""
        t, c = quote_ident(table), quote_ident(column)
        core = self._core(ref)
        probes: List[str] = []

        def fetch(where: str, limit: int) -> List[str]:
            sql = f"SELECT DISTINCT {c} FROM {t} WHERE {where} LIMIT {limit}"
            probes.append(sql)
            res = self._run(db_path, sql)
            return [str(r[0]) for r in (res.rows or []) if r[0] is not None]

        card_sql = f"SELECT COUNT(DISTINCT {c}) FROM {t}"
        probes.append(card_sql)
        card_res = self._run(db_path, card_sql)
        card = int(card_res.rows[0][0]) if card_res.rows else None

        def finish(values: List[str], strategy: str, cap: int):
            uniq = list(dict.fromkeys(values))
            uniq.sort(key=lambda v: difflib.SequenceMatcher(None, _squash(v), _squash(core)).ratio(), reverse=True)
            return uniq[:cap], strategy, probes, card

        if card is not None and 0 < card <= self.low_cardinality_threshold:
            return finish(fetch(f"{c} IS NOT NULL", card), "enumerate", self.low_cardinality_threshold)

        n = self.max_candidates * 3
        found = fetch(f"LOWER(TRIM({c})) = {quote_literal(core.casefold())}", n)
        if found:
            return finish(found, "ci_trim", self.max_candidates)
        squashed = f"LOWER({c})"
        for ch in _SQUASH_CHARS:
            squashed = f"REPLACE({squashed}, {quote_literal(ch)}, '')"
        if _squash(core):
            found = fetch(f"{squashed} = {quote_literal(_squash(core))}", n)
            if found:
                return finish(found, "squash", self.max_candidates)
        if core:
            found = fetch(f"LOWER({c}) LIKE {quote_literal('%' + like_escape(core.casefold()) + '%')} ESCAPE '\\'", n)
            if found:
                return finish(found, "contains", self.max_candidates)
        toks = _tokens(core)
        if len(toks) >= 2:
            where = " AND ".join(f"LOWER({c}) LIKE {quote_literal('%' + like_escape(x) + '%')} ESCAPE '\\'"
                                 for x in toks[:3])
            found = fetch(where, n)
            if found:
                return finish(found, "tokens_and", self.max_candidates)
        if toks:
            found = fetch(f"LOWER({c}) LIKE {quote_literal('%' + like_escape(toks[0]) + '%')} ESCAPE '\\'", n)
            if found:
                return finish(found, "token_longest", self.max_candidates)
        return [], "none", probes, card

    def diagnose(self, db_path: Union[str, Path], sql: str) -> Tuple[List[ProbeRecord], List[_Ref], str]:
        """Probe every string literal; return (records, unmatched refs with candidates, note)."""
        conds, refs, note = self._collect_refs(db_path, sql)
        records: List[ProbeRecord] = []
        actionable: List[_Ref] = []
        for ref in refs:
            table, column = ref.tables[0]
            if any(self._exists(db_path, t, c, ref) for t, c in ref.tables):
                rec = ProbeRecord(clause=ref.clause, operator=ref.kind, table=table, column=column,
                                  literal=ref.literal, literal_exists=True)
                records.append(rec)
                continue
            if len(actionable) >= self.max_literals:
                continue
            rec = ProbeRecord(clause=ref.clause, operator=ref.kind, table=table, column=column,
                              literal=ref.literal, literal_exists=False)
            for t, c in ref.tables[:3]:
                cands, strategy, probes, card = self._gather(db_path, t, c, ref)
                rec.probe_sqls += probes
                if cands:
                    rec.table, rec.column, rec.candidates = t, c, cands
                    rec.strategy, rec.cardinality = strategy, card
                    break
                rec.strategy, rec.cardinality = strategy, card
            ref.record = rec
            records.append(rec)
            if rec.candidates:
                actionable.append(ref)
        return records, actionable, note

    # ------------------------------------------------------------------ #
    # choosing + editing
    # ------------------------------------------------------------------ #
    def _fastpath(self, ref: _Ref) -> Optional[str]:
        rec = ref.record
        assert rec is not None
        same = [c for c in rec.candidates if _squash(c) == _squash(self._core(ref))]
        return same[0] if len(same) == 1 else None

    def _choose_with_llm(self, question: str, schema: str, sql: str, evidence: str,
                         refs: List[_Ref], usage: TokenUsage) -> Dict[int, Optional[str]]:
        blocks = []
        for i, ref in enumerate(refs):
            rec = ref.record
            assert rec is not None
            blocks.append(f"[{i}] {rec.table}.{rec.column} {rec.operator.value} {ref.literal!r}  "
                          f"(distinct values: {rec.cardinality}; candidates: {rec.candidates})")
        user = "\n".join(filter(None, [
            f"Question: {question}",
            f"External knowledge: {evidence}" if evidence else "",
            f"Database schema:\n{schema}",
            f"SQL (returned 0 rows):\n{sql}",
            "These query literals match no stored value. For each, choose the candidate that the "
            "question actually refers to, or null if none fits:\n" + "\n".join(blocks),
            'Reply with JSON only: {"choices": [{"id": 0, "value": "<candidate copied verbatim, or null>"}]}',
        ]))
        content = call_llm(self.llm, [
            {"role": "system", "content": "You map query literals to the values actually stored in a database. "
                                          "Only answer with a candidate copied verbatim, or null. Never invent values."},
            {"role": "user", "content": user}], usage)
        obj = extract_json_object(content) or {}
        out: Dict[int, Optional[str]] = {}
        for item in obj.get("choices", []) if isinstance(obj.get("choices"), list) else []:
            try:
                idx = int(item.get("id"))
            except (TypeError, ValueError, AttributeError):
                continue
            value = item.get("value")
            cands = refs[idx].record.candidates if 0 <= idx < len(refs) and refs[idx].record else []
            if isinstance(value, str):
                exact = next((c for c in cands if c == value), None) \
                    or next((c for c in cands if c.casefold() == value.casefold()), None)
                out[idx] = exact  # ungrounded answers become None
        return out

    @staticmethod
    def _final_literal(ref: _Ref, chosen: str) -> str:
        value = chosen
        for w in ref.wrappers:  # keep the predicate's own LOWER/UPPER/TRIM semantics satisfiable
            value = {"LOWER": value.lower(), "UPPER": value.upper(), "TRIM": value.strip()}[w]
        if ref.kind == PredicateKind.LIKE:
            value = ("%" if ref.literal.startswith("%") else "") + value + ("%" if ref.literal.endswith("%") else "")
        return value

    def _apply(self, sql: str, edits: List[Tuple[_Ref, str]]) -> str:
        split = split_clauses(sql)
        by_clause: Dict[ClauseType, List[Tuple[_Ref, str]]] = {}
        for ref, new in edits:
            by_clause.setdefault(ref.clause, []).append((ref, new))
        for ct, items in by_clause.items():
            body, textual = split.clauses[ct], True
            for ref, new in items:  # minimal-diff text edit when the literal text is unambiguous
                if body.count(ref.token_text) != 1:
                    textual = False
                    break
                body = body.replace(ref.token_text, quote_literal(new))
            if not textual:
                for ref, new in items:
                    ref.node.replace(exp.Literal.string(new))
                body = self._conds[ct].sql(dialect=self.dialect)
            sql = replace_clause_in_sql(sql, ct, body, self.dialect, normalize=False)
        return sql

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #
    def heal(self, question: str, schema: str, sql: str, db_path: Union[str, Path], evidence: str = "",
             sandbox_result: Optional[SandboxResult] = None,
             verifier: Optional[GuardedVerifier] = None) -> ProbeResult:
        """Try to repair an EMPTY query by grounding its literals in the database.

        Args:
            sandbox_result: result of executing ``sql``. If given and its status is not
                EMPTY nothing is done; if omitted the query is executed to confirm it is EMPTY.
            verifier: optional Gap-3 gate, consulted only when an edit is about to be made;
                ``EXPECTED_EMPTY`` leaves the query untouched.
        """
        self._probe_count = 0
        usage = TokenUsage()
        out = ProbeResult(original_sql=sql, healed_sql=sql)

        def finish(reason: str) -> ProbeResult:
            out.reason = reason
            out.prompt_tokens, out.completion_tokens, out.total_calls = (
                usage.prompt_tokens + out.prompt_tokens, usage.completion_tokens + out.completion_tokens,
                usage.calls + out.total_calls)
            out.probe_queries = self._probe_count
            return out

        status = sandbox_result.status if sandbox_result else self._run(db_path, sql).status
        if status != ExecutionStatus.EMPTY:
            return finish(f"not triggered: execution status is {status.value}, not EMPTY")

        records, actionable, note = self.diagnose(db_path, sql)
        out.probes = records
        if note:
            return finish(note)
        if not actionable:
            if records and all(r.literal_exists for r in records):
                return finish("every string literal exists in its column; emptiness is not a literal mismatch")
            return finish("no string literal with groundable candidates found")

        if verifier is not None:
            hints = [f"literal {r.literal!r} not found in {r.table}.{r.column}; stored values include "
                     f"{r.candidates[:5]}" for r in (a.record for a in actionable) if r]
            verdict = verifier.verify(question, schema, sql, db_path=db_path, evidence=evidence, probe_hints=hints)
            out.verdict, out.verdict_reason = verdict.verdict, verdict.reason
            out.prompt_tokens += verdict.prompt_tokens
            out.completion_tokens += verdict.completion_tokens
            out.total_calls += verdict.total_calls
            if not verdict.allow_healing:
                return finish(f"blocked by verifier: {verdict.reason}")

        pending: List[_Ref] = []
        for ref in actionable:
            fast = self._fastpath(ref)
            if fast is not None:
                ref.record.chosen, ref.record.method = fast, "fastpath"  # type: ignore[union-attr]
            else:
                pending.append(ref)
        if pending:
            choices = self._choose_with_llm(question, schema, sql, evidence, pending, usage)
            for i, ref in enumerate(pending):
                if choices.get(i):
                    ref.record.chosen, ref.record.method = choices[i], "llm"  # type: ignore[union-attr]

        edits: List[Tuple[_Ref, str]] = []
        for ref in actionable:
            rec = ref.record
            if rec and rec.chosen:
                rec.final_literal = self._final_literal(ref, rec.chosen)
                if rec.final_literal != ref.literal:
                    edits.append((ref, rec.final_literal))
        if not edits:
            return finish("no grounded replacement was selected")

        try:
            candidate = self._apply(sql, edits)
        except ReconstructionError as exc:
            return finish(f"could not rebuild the query: {exc}")
        out.candidate_sql = candidate
        res = self._run(db_path, candidate)
        out.healed_status = res.status
        if self.require_rows and res.status != ExecutionStatus.SUCCESS:
            return finish(f"edited query was rejected: status {res.status.value}")
        out.healed_sql, out.healed = candidate, True
        return finish(f"replaced {len(edits)} literal(s)")
