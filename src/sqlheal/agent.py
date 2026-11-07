"""SQLHeal agent: a bounded execute -> diagnose -> repair state machine.

Per turn the agent executes the current SQL in the read-only sandbox and branches on the
typed ``ExecutionStatus``:

    SUCCESS            -> accept, stop.
    ERROR / TIMEOUT    -> Gap 1: ``ASTPatcher`` rewrites only the broken clause.
                          If the patcher is disabled (ablation) or cannot localise / produce a
                          valid patch, fall back to whole-query regeneration (the B1 baseline).
    EMPTY              -> Gap 3 + Gap 2: ``LiteralProber.heal`` probes literals; the
                          ``GuardedVerifier`` is consulted as a gate right before an edit.
                          If nothing is healed (or the prober is disabled) the empty result is
                          accepted as final.

``max_turns`` bounds the number of *executions*; every repair produces the next turn's SQL.
Every tool call is logged to a ``TrajectoryLogger`` (see ``save_trajectory``). The "thought"
field is a short controller rationale generated from the observed state, not hidden model
chain-of-thought.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from sqlheal.modules import ASTPatcher, GuardedVerifier, LiteralProber, locate_error_clause
from sqlheal.modules._common import LLMLike, TokenUsage, call_llm, extract_code_block
from sqlheal.parsing import canonicalise_sql, extract_clause_table
from sqlheal.reconstruct import diff_clauses
from sqlheal.sandbox import SQLiteSandbox
from sqlheal.state import (
    ClauseType, ExecutionStatus, QueryEvaluationState, QueryTurn, SandboxResult,
)

__all__ = ["SQLHealAgent", "TrajectoryLogger", "TrajectoryStep", "get_schema_text"]

_RANK = {ExecutionStatus.SUCCESS: 2, ExecutionStatus.EMPTY: 1, ExecutionStatus.ERROR: 0, ExecutionStatus.TIMEOUT: 0}

GENERATE_SYSTEM = (
    "You are an expert SQLite analyst. Write ONE SQLite query that answers the question. "
    "Use only tables and columns from the schema. Return only the query in a ```sql block."
)
REWRITE_SYSTEM = (
    "You are an expert SQLite analyst repairing a failed query. Return the full corrected "
    "SQLite query only, in a ```sql block."
)


# --------------------------------------------------------------------------- #
# Trajectory logging
# --------------------------------------------------------------------------- #
class TrajectoryStep(BaseModel):
    step: int
    turn: int
    thought: str
    action: str                       # generate | execute | patch | rewrite | verify | probe_and_heal | accept | stop
    tool: str                         # LLM | SQLiteSandbox | ASTPatcher | LiteralProber | GuardedVerifier | controller
    input_sql: Optional[str] = None
    output_sql: Optional[str] = None
    status: Optional[ExecutionStatus] = None   # set on execute steps
    execution_time_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = ""


class TrajectoryLogger:
    """Collects ``TrajectoryStep``s, notifies live listeners, and writes JSON."""

    def __init__(self, meta: Dict[str, Any], listener: Optional[Callable[[TrajectoryStep], None]] = None) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.meta = dict(meta)
        self.steps: List[TrajectoryStep] = []
        self.final: Dict[str, Any] = {}
        self._listener = listener

    def log(self, turn: int, thought: str, action: str, tool: str, **kw: Any) -> TrajectoryStep:
        step = TrajectoryStep(step=len(self.steps) + 1, turn=turn, thought=thought, action=action, tool=tool,
                              timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **kw)
        self.steps.append(step)
        if self._listener:
            self._listener(step)
        return step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            **self.meta,
            "steps": [s.model_dump(mode="json") for s in self.steps],
            "final": self.final,
        }

    def save(self, output_path: Union[str, Path]) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        return path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_schema_text(sandbox: SQLiteSandbox, db_path: Union[str, Path]) -> str:
    """Schema prompt text = the CREATE TABLE statements stored in ``sqlite_master`` (read-only)."""
    res = sandbox.execute(db_path, "SELECT name, sql FROM sqlite_master "
                                   "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    return "\n".join(str(r[1]).strip() + ";" for r in (res.rows or []) if r[1])


def _collapse(sql: str) -> str:
    return re.sub(r"\s+", " ", sql or "").strip().rstrip(";").strip()


def _rows_set(result: SandboxResult) -> set:
    return {tuple(r) for r in (result.rows or [])}


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class SQLHealAgent:
    """Bounded self-healing NL-to-SQL agent.

    Args:
        llm: ``LLMClient`` (anything with ``generate(messages)``).
        sandbox: read-only SQLite sandbox; a default one is created if omitted.
        patcher / prober / verifier: gap modules; defaults are built from ``llm``/``sandbox``.
        max_turns: maximum number of SQL executions (default 3).
        enable_patcher: False -> every ERROR/TIMEOUT uses whole-query regeneration (B1).
        enable_prober: False -> EMPTY results are accepted as final.
        enable_verifier: False -> every EMPTY result is probed without the Gap-3 guard.
        on_step: optional live callback receiving each ``TrajectoryStep`` (used by the CLI).
        model_name: recorded in the trajectory metadata.
    """

    def __init__(self, llm: LLMLike, sandbox: Optional[SQLiteSandbox] = None,
                 patcher: Optional[ASTPatcher] = None, prober: Optional[LiteralProber] = None,
                 verifier: Optional[GuardedVerifier] = None, max_turns: int = 3,
                 enable_patcher: bool = True, enable_prober: bool = True, enable_verifier: bool = True,
                 on_step: Optional[Callable[[TrajectoryStep], None]] = None,
                 model_name: Optional[str] = None) -> None:
        self.llm = llm
        self.sandbox = sandbox or SQLiteSandbox()
        self.patcher = patcher or ASTPatcher(llm)
        self.prober = prober or LiteralProber(llm, self.sandbox)
        self.verifier = verifier or GuardedVerifier(llm, self.sandbox)
        self.max_turns = max(1, max_turns)
        self.enable_patcher = enable_patcher
        self.enable_prober = enable_prober
        self.enable_verifier = enable_verifier
        self.on_step = on_step
        self.model_name = model_name or getattr(llm, "model", None)
        self.trajectory: Optional[TrajectoryLogger] = None
        self.last_state: Optional[QueryEvaluationState] = None
        self.final_result: Optional[SandboxResult] = None

    # ------------------------------------------------------------------ #
    # LLM-backed steps
    # ------------------------------------------------------------------ #
    def _generate(self, question: str, schema: str, evidence: str, usage: TokenUsage) -> str:
        user = "\n".join(filter(None, [
            f"Database schema:\n{schema}",
            f"External knowledge: {evidence}" if evidence else "",
            f"Question: {question}",
        ]))
        content = call_llm(self.llm, [{"role": "system", "content": GENERATE_SYSTEM},
                                      {"role": "user", "content": user}], usage)
        return extract_code_block(content).strip().rstrip(";").strip()

    def _rewrite(self, question: str, schema: str, evidence: str, sql: str, error: str,
                 usage: TokenUsage) -> str:
        """B1 baseline: regenerate the whole query from the error message."""
        user = "\n".join(filter(None, [
            f"Database schema:\n{schema}",
            f"External knowledge: {evidence}" if evidence else "",
            f"Question: {question}",
            "The previous query failed.",
            f"Query: {sql}",
            f"Error: {error}",
        ]))
        content = call_llm(self.llm, [{"role": "system", "content": REWRITE_SYSTEM},
                                      {"role": "user", "content": user}], usage)
        return extract_code_block(content).strip().rstrip(";").strip()

    # ------------------------------------------------------------------ #
    # repair dispatch
    # ------------------------------------------------------------------ #
    def _repair_error(self, traj: TrajectoryLogger, turn: int, question: str, schema: str, evidence: str,
                      sql: str, result: SandboxResult) -> Optional[Tuple[str, TokenUsage]]:
        """ERROR/TIMEOUT branch. Returns (new_sql, usage) or None if no repair could be made."""
        usage = TokenUsage()
        err = result.error_message or result.status.value
        why = "patcher disabled (ablation)"
        if self.enable_patcher:
            pr = self.patcher.patch(question, schema, sql, err, evidence)
            usage.prompt_tokens += pr.prompt_tokens
            usage.completion_tokens += pr.completion_tokens
            usage.calls += pr.total_calls
            traj.log(
                turn, f"{result.status.value}: '{err}'. Localised to "
                      f"{pr.target_clause.value if pr.target_clause else 'no single clause'}; "
                      f"all other clauses frozen.", "patch", "ASTPatcher",
                input_sql=sql, output_sql=pr.patched_sql if pr.applied else None,
                prompt_tokens=pr.prompt_tokens, completion_tokens=pr.completion_tokens, error=pr.error,
                details={
                    "status": pr.status.value, "target_clause": pr.target_clause.value if pr.target_clause else None,
                    "localization_reason": pr.localization_reason, "broken_content": pr.broken_content,
                    "new_content": pr.new_content, "frozen_intact": pr.frozen_intact, "attempts": pr.attempts,
                    "changed_clauses": [c.value for c in diff_clauses(sql, pr.patched_sql)] if pr.applied else [],
                    "drift_clauses": [],
                })
            if pr.applied:
                return pr.patched_sql, usage
            why = f"patcher returned {pr.status.value} ({pr.error})"
        try:
            new_sql = self._rewrite(question, schema, evidence, sql, err, usage)
        except Exception as exc:  # noqa: BLE001 - LLM/network failure ends the loop gracefully
            traj.log(turn, f"Whole-query rewrite failed: {exc}", "rewrite", "LLM", input_sql=sql, error=str(exc))
            return None
        loc = locate_error_clause(sql, err)
        target = loc.clause if loc else None
        changed = diff_clauses(sql, new_sql)
        drift = [c for c in changed if c != target] if target else []
        traj.log(
            turn, f"{why}; regenerating the whole query (B1 baseline).", "rewrite", "LLM",
            input_sql=sql, output_sql=new_sql or None, prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            details={"estimated_target": target.value if target else None,
                     "changed_clauses": [c.value for c in changed], "drift_clauses": [c.value for c in drift]})
        return (new_sql, usage) if new_sql else None

    def _repair_empty(self, traj: TrajectoryLogger, turn: int, question: str, schema: str, evidence: str,
                      db_path: str, sql: str, result: SandboxResult) -> Optional[Tuple[str, TokenUsage]]:
        """EMPTY branch. Returns (new_sql, usage) if healed, else None (empty accepted)."""
        if not self.enable_prober:
            traj.log(turn, "0 rows; prober disabled (ablation), so the empty result is accepted.",
                     "accept", "controller", input_sql=sql, details={"reason": "prober_disabled"})
            return None
        hr = self.prober.heal(question, schema, sql, db_path, evidence=evidence, sandbox_result=result,
                              verifier=self.verifier if self.enable_verifier else None)
        usage = TokenUsage(prompt_tokens=hr.prompt_tokens, completion_tokens=hr.completion_tokens,
                           calls=hr.total_calls)
        if hr.verdict is not None:
            traj.log(turn, f"Gate before editing: {hr.verdict.value}. {hr.verdict_reason}", "verify",
                     "GuardedVerifier", input_sql=sql,
                     details={"verdict": hr.verdict.value, "reason": hr.verdict_reason,
                              "tokens_accounted_in": "probe_and_heal"})
        guard = "verifier gate on" if self.enable_verifier else "verifier gate OFF (ablation)"
        traj.log(
            turn, f"0 rows; probing string literals against the database ({guard}). {hr.reason}.",
            "probe_and_heal", "LiteralProber", input_sql=sql, output_sql=hr.healed_sql if hr.healed else None,
            prompt_tokens=hr.prompt_tokens, completion_tokens=hr.completion_tokens,
            details={"healed": hr.healed, "reason": hr.reason, "probe_queries": hr.probe_queries,
                     "probes": [p.model_dump(mode="json") for p in hr.probes],
                     "changed_clauses": [c.value for c in diff_clauses(sql, hr.healed_sql)] if hr.healed else []})
        if hr.healed:
            return hr.healed_sql, usage
        traj.log(turn, f"No repair applied ({hr.reason}); accepting the empty result.", "accept", "controller",
                 input_sql=sql, details={"reason": "empty_accepted"})
        return None

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def run(self, db_path: Union[str, Path], question: str, sql: Optional[str] = None,
            question_id: Union[int, str] = "adhoc", evidence: str = "", gold_sql: Optional[str] = None,
            schema: Optional[str] = None) -> QueryEvaluationState:
        """Answer ``question`` against ``db_path`` (optionally starting from a candidate ``sql``).

        ``gold_sql`` is used only *after* the loop to compute EX / exact match; the agent never
        sees it. Returns a ``QueryEvaluationState`` compatible with ``metrics.py``.
        """
        db = str(db_path)
        state = QueryEvaluationState(question_id=question_id, db_id=Path(db).stem, question=question,
                                     gold_sql=gold_sql)
        traj = self.trajectory = TrajectoryLogger({
            "question_id": question_id, "db_id": state.db_id, "db_path": db, "question": question,
            "model": self.model_name, "max_turns": self.max_turns,
            "config": {"enable_patcher": self.enable_patcher, "enable_prober": self.enable_prober,
                       "enable_verifier": self.enable_verifier},
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, self.on_step)
        self.last_state, self.final_result = state, None
        schema = schema if schema is not None else get_schema_text(self.sandbox, db)

        carry = TokenUsage()
        current = (sql or "").strip().rstrip(";").strip()
        if not current:
            usage = TokenUsage()
            try:
                current = self._generate(question, schema, evidence, usage)
            except Exception as exc:  # noqa: BLE001
                traj.log(1, f"Initial generation failed: {exc}", "generate", "LLM", error=str(exc))
                return self._finish(state, traj, db, [], gold_sql)
            traj.log(1, "No candidate SQL supplied; generating the initial query from question and schema.",
                     "generate", "LLM", output_sql=current or None, prompt_tokens=usage.prompt_tokens,
                     completion_tokens=usage.completion_tokens)
            carry = usage
        else:
            traj.log(1, "Using the supplied candidate SQL as turn 1.", "generate", "controller", output_sql=current)

        seen: set = set()
        results: List[SandboxResult] = []
        for turn in range(1, self.max_turns + 1):
            seen.add(_collapse(current))
            res = self.sandbox.execute(db, current)
            results.append(res)
            state.turns.append(QueryTurn(
                turn_index=turn, sql=current, clause_table=extract_clause_table(current),
                sandbox_result=res, prompt_tokens=carry.prompt_tokens, completion_tokens=carry.completion_tokens))
            state.total_calls += carry.calls
            state.total_prompt_tokens += carry.prompt_tokens
            state.total_completion_tokens += carry.completion_tokens
            nrows = len(res.rows or [])
            traj.log(turn, f"Executing turn {turn} in the read-only sandbox.", "execute", "SQLiteSandbox",
                     input_sql=current, status=res.status, execution_time_ms=res.execution_time_ms,
                     error=res.error_message, details={"row_count": nrows, "columns": res.columns,
                                                       "rows_preview": [list(r) for r in (res.rows or [])[:3]]})
            if res.status == ExecutionStatus.SUCCESS:
                traj.log(turn, f"Returned {nrows} row(s); accepting.", "accept", "controller", input_sql=current,
                         details={"reason": "success"})
                break
            if turn == self.max_turns:
                traj.log(turn, f"Turn budget ({self.max_turns}) exhausted with status {res.status.value}.",
                         "stop", "controller", input_sql=current, details={"reason": "max_turns"})
                break

            if res.status == ExecutionStatus.EMPTY:
                repaired = self._repair_empty(traj, turn, question, schema, evidence, db, current, res)
            else:
                repaired = self._repair_error(traj, turn, question, schema, evidence, current, res)
            if repaired is None:
                if res.status != ExecutionStatus.EMPTY:
                    traj.log(turn, "No repair could be produced; stopping.", "stop", "controller",
                             input_sql=current, details={"reason": "no_repair"})
                break
            new_sql, carry = repaired
            if _collapse(new_sql) in seen:
                traj.log(turn, "Repair reproduced an already-tried query; stopping to avoid a loop.", "stop",
                         "controller", input_sql=current, output_sql=new_sql, details={"reason": "no_progress"})
                break
            current = new_sql.strip().rstrip(";").strip()
        return self._finish(state, traj, db, results, gold_sql)

    # ------------------------------------------------------------------ #
    def _finish(self, state: QueryEvaluationState, traj: TrajectoryLogger, db: str,
                results: List[SandboxResult], gold_sql: Optional[str]) -> QueryEvaluationState:
        if state.turns:
            # prefer SUCCESS > EMPTY > ERROR/TIMEOUT; among equals the latest turn
            best = max(range(len(state.turns)),
                       key=lambda i: (_RANK[state.turns[i].sandbox_result.status], i))  # type: ignore[union-attr]
            final_turn = state.turns[best]
            state.final_sql = final_turn.sql
            state.final_status = final_turn.sandbox_result.status  # type: ignore[union-attr]
            self.final_result = final_turn.sandbox_result
        if gold_sql and state.final_sql:
            gold = self.sandbox.execute(db, gold_sql)
            if self.final_result is not None and gold.status in (ExecutionStatus.SUCCESS, ExecutionStatus.EMPTY):
                state.execution_match = (self.final_result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.EMPTY)
                                         and _rows_set(self.final_result) == _rows_set(gold))
            state.exact_match = canonicalise_sql(state.final_sql).lower() == canonicalise_sql(gold_sql).lower()
        traj.final = {
            "final_sql": state.final_sql, "final_status": state.final_status.value,
            "turns_used": len(state.turns), "total_calls": state.total_calls,
            "total_prompt_tokens": state.total_prompt_tokens,
            "total_completion_tokens": state.total_completion_tokens,
            "total_execution_ms": round(sum(t.sandbox_result.execution_time_ms for t in state.turns
                                            if t.sandbox_result), 3),
            "execution_match": state.execution_match if gold_sql else None,
            "exact_match": state.exact_match if gold_sql else None,
        }
        return state

    def save_trajectory(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Write the last run's trajectory as JSON (default ``logs/<question_id>_<run_id>.json``)."""
        if self.trajectory is None:
            raise RuntimeError("No trajectory yet: call run() first.")
        if output_path is None:
            qid = re.sub(r"[^\w.-]+", "_", str(self.trajectory.meta.get("question_id", "run")))
            output_path = Path("logs") / f"{qid}_{self.trajectory.run_id}.json"
        return self.trajectory.save(output_path)
