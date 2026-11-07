"""Evaluation metrics for SQLHeal (Text-to-SQL repair).

Metrics follow Section B.12.2 of the Phase 1 proposal.

    EX     Execution Accuracy: final result rows equal the gold result rows.
    RDR    Repair Drift Rate (Gap 1): share of clauses that were correct in the initial
           query (equal to the gold clause) but were changed by the repair.
    HRR    Healing Recovery Rate (Gap 1): share of queries whose first execution was an
           ERROR that end in SUCCESS (hrr_ex_pct is the stricter "ends with gold result").
    ERP    Empty-Set Recovery Precision (Gap 2): share of empty-result edits that end with
           the gold result. Also reported: recovery rate and SFPR before / after repair.
    EOCR   Empty-Set Over-Correction Rate (Gap 3): share of genuinely empty gold answers
           (that were empty at first) that end up returning rows.

Notes on how the numbers are produced:
    * "Correct clause" means the clause text equals the gold clause text after
      whitespace collapsing (reconstruct.diff_clauses). This is stricter than semantic
      equality, so RDR is an upper bound on true drift (proposal B.12.1).
    * Gold emptiness is not stored in QueryEvaluationState, so functions that need it take
      a ``gold_empty`` set of question ids (as strings).
    * The old ``compute_rdr`` of this file measured repair success, not drift. It is now
      called ``compute_repair_success_rate``; ``compute_rdr`` is the Repair Drift Rate.
"""
from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from pydantic import BaseModel

from sqlheal.reconstruct import CLAUSE_ORDER, diff_clauses, split_clauses
from sqlheal.state import ExecutionStatus, QueryEvaluationState, SandboxResult

# Default price per 1M tokens (USD). Override with the "pricing" block of the config file.
DEFAULT_PRICES: Dict[str, float] = {"input_per_1m": 0.15, "output_per_1m": 0.60}

_OK = (ExecutionStatus.SUCCESS, ExecutionStatus.EMPTY)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _dump(model: BaseModel) -> Dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _pct(num: int, den: int) -> Optional[float]:
    """Percentage rounded to 2 places, or None when the denominator is 0."""
    if den == 0:
        return None
    return round(100.0 * num / den, 2)


def _collapse(sql: str) -> str:
    return " ".join((sql or "").split()).rstrip(";").strip()


def _ids(gold_empty: Optional[Iterable[Any]]) -> Set[str]:
    return {str(x) for x in (gold_empty or [])}


def initial_status(state: QueryEvaluationState) -> Optional[ExecutionStatus]:
    """Execution status of the first turn (the candidate before any repair)."""
    if not state.turns or state.turns[0].sandbox_result is None:
        return None
    return state.turns[0].sandbox_result.status


def final_result(state: QueryEvaluationState) -> Optional[SandboxResult]:
    """Sandbox result of the turn that produced ``final_sql`` (the latest such turn)."""
    for turn in reversed(state.turns):
        if turn.sql == state.final_sql:
            return turn.sandbox_result
    return state.turns[-1].sandbox_result if state.turns else None


def execution_match(predicted: Optional[SandboxResult], gold: Optional[SandboxResult]) -> bool:
    """EX rule used for every system: both ran (rows or empty) and the row sets are equal.

    Two empty results match, so a correct "no rows" answer is counted as correct.
    """
    if predicted is None or gold is None:
        return False
    if predicted.status not in _OK or gold.status not in _OK:
        return False
    return {tuple(r) for r in (predicted.rows or [])} == {tuple(r) for r in (gold.rows or [])}


# --------------------------------------------------------------------------- #
# EX and first-turn failure rates
# --------------------------------------------------------------------------- #
def compute_ex(states: List[QueryEvaluationState]) -> Optional[float]:
    return _pct(sum(1 for s in states if s.execution_match), len(states))


class InitialRates(BaseModel):
    initial_error_pct: Optional[float] = None
    initial_empty_pct: Optional[float] = None
    initial_success_pct: Optional[float] = None


def compute_initial_rates(states: List[QueryEvaluationState]) -> InitialRates:
    """Natural failure rates of the first candidate (what B0 reports)."""
    first = [initial_status(s) for s in states]
    n = len(states)
    return InitialRates(
        initial_error_pct=_pct(sum(1 for x in first if x in (ExecutionStatus.ERROR, ExecutionStatus.TIMEOUT)), n),
        initial_empty_pct=_pct(sum(1 for x in first if x == ExecutionStatus.EMPTY), n),
        initial_success_pct=_pct(sum(1 for x in first if x == ExecutionStatus.SUCCESS), n),
    )


# --------------------------------------------------------------------------- #
# RDR - Repair Drift Rate (Gap 1)
# --------------------------------------------------------------------------- #
def drift_for_state(state: QueryEvaluationState) -> Optional[Tuple[List[Any], List[Any]]]:
    """Return (correct_clauses_before_repair, drifted_clauses) for one query, or None.

    None means the query is not scored: no gold, no repair happened, or a query
    could not be split into clauses (UNION, WINDOW, ...).
    """
    if not state.gold_sql or len(state.turns) < 2 or not state.final_sql:
        return None
    before = state.turns[0].sql
    after = state.final_sql
    if _collapse(before) == _collapse(after):
        return None
    split_before = split_clauses(before)
    split_gold = split_clauses(state.gold_sql)
    if not (split_before.supported and split_gold.supported and split_clauses(after).supported):
        return None
    slots = [c for c in CLAUSE_ORDER if c in split_before.clauses or c in split_gold.clauses]
    wrong_before = set(diff_clauses(before, state.gold_sql))
    correct = [c for c in slots if c not in wrong_before]
    changed = set(diff_clauses(before, after))
    drifted = [c for c in correct if c in changed]
    return correct, drifted


class RDRResult(BaseModel):
    rdr_pct: Optional[float] = None        # drifted clauses / correct clauses (clause level)
    rdr_query_pct: Optional[float] = None  # repaired queries with at least one drifted clause
    rdr_drifted_clauses: int = 0
    rdr_correct_clauses: int = 0
    rdr_repaired_queries: int = 0
    rdr_excluded_queries: int = 0          # repaired but not splittable into clauses


def compute_rdr(states: List[QueryEvaluationState]) -> RDRResult:
    drifted_total = 0
    correct_total = 0
    repaired = 0
    with_drift = 0
    excluded = 0
    for s in states:
        repair_happened = len(s.turns) > 1 and s.final_sql and _collapse(s.turns[0].sql) != _collapse(s.final_sql)
        scored = drift_for_state(s)
        if scored is None:
            if repair_happened and s.gold_sql:
                excluded += 1
            continue
        correct, drifted = scored
        repaired += 1
        correct_total += len(correct)
        drifted_total += len(drifted)
        if drifted:
            with_drift += 1
    return RDRResult(
        rdr_pct=_pct(drifted_total, correct_total),
        rdr_query_pct=_pct(with_drift, repaired),
        rdr_drifted_clauses=drifted_total,
        rdr_correct_clauses=correct_total,
        rdr_repaired_queries=repaired,
        rdr_excluded_queries=excluded,
    )


# --------------------------------------------------------------------------- #
# HRR - Healing Recovery Rate (Gap 1)
# --------------------------------------------------------------------------- #
class HRRResult(BaseModel):
    hrr_pct: Optional[float] = None      # initial ERROR -> final SUCCESS
    hrr_ex_pct: Optional[float] = None   # initial ERROR -> final result equals gold
    hrr_initial_errors: int = 0
    hrr_recovered: int = 0


def compute_hrr(states: List[QueryEvaluationState]) -> HRRResult:
    failed = [s for s in states if initial_status(s) == ExecutionStatus.ERROR]
    recovered = sum(1 for s in failed if s.final_status == ExecutionStatus.SUCCESS)
    recovered_ex = sum(1 for s in failed if s.execution_match)
    return HRRResult(
        hrr_pct=_pct(recovered, len(failed)),
        hrr_ex_pct=_pct(recovered_ex, len(failed)),
        hrr_initial_errors=len(failed),
        hrr_recovered=recovered,
    )


# --------------------------------------------------------------------------- #
# ERP - Empty-Set Recovery Precision (Gap 2), plus recovery rate and SFPR
# --------------------------------------------------------------------------- #
def _has_empty_edit(state: QueryEvaluationState) -> bool:
    """True if some EMPTY turn was followed by another turn (an edit was applied)."""
    for i, turn in enumerate(state.turns[:-1]):
        if turn.sandbox_result is not None and turn.sandbox_result.status == ExecutionStatus.EMPTY:
            return True
    return False


class ERPResult(BaseModel):
    erp_pct: Optional[float] = None            # edits that end with the gold result / all edits
    erp_edits: int = 0
    erp_recovered: int = 0
    recovery_pct: Optional[float] = None       # gold non-empty, initially empty, finally correct
    recovery_candidates: int = 0
    sfpr_before_pct: Optional[float] = None    # gold non-empty and initial result empty
    sfpr_after_pct: Optional[float] = None     # gold non-empty and final result empty
    sfpr_rel_reduction_pct: Optional[float] = None


def compute_erp(states: List[QueryEvaluationState], gold_empty: Optional[Iterable[Any]] = None) -> ERPResult:
    """``gold_empty``: ids of questions whose gold answer is empty (default: none)."""
    empty_ids = _ids(gold_empty)
    edited = [s for s in states if _has_empty_edit(s)]
    recovered = sum(1 for s in edited if s.execution_match)

    nonempty_gold = [s for s in states if str(s.question_id) not in empty_ids]
    start_empty = [s for s in nonempty_gold if initial_status(s) == ExecutionStatus.EMPTY]
    end_empty = [s for s in nonempty_gold if s.final_status == ExecutionStatus.EMPTY]
    fixed = sum(1 for s in start_empty if s.execution_match)

    before = _pct(len(start_empty), len(nonempty_gold))
    after = _pct(len(end_empty), len(nonempty_gold))
    reduction = None
    if before:
        reduction = round(100.0 * (before - (after or 0.0)) / before, 2)
    return ERPResult(
        erp_pct=_pct(recovered, len(edited)),
        erp_edits=len(edited),
        erp_recovered=recovered,
        recovery_pct=_pct(fixed, len(start_empty)),
        recovery_candidates=len(start_empty),
        sfpr_before_pct=before,
        sfpr_after_pct=after,
        sfpr_rel_reduction_pct=reduction,
    )


# --------------------------------------------------------------------------- #
# EOCR - Empty-Set Over-Correction Rate (Gap 3)
# --------------------------------------------------------------------------- #
class EOCRResult(BaseModel):
    eocr_pct: Optional[float] = None
    eocr_population: int = 0     # E*: gold empty and initial result empty
    eocr_overcorrected: int = 0  # E* queries that end up returning rows


def compute_eocr(states: List[QueryEvaluationState], gold_empty: Optional[Iterable[Any]] = None) -> EOCRResult:
    empty_ids = _ids(gold_empty)
    population = [s for s in states
                  if str(s.question_id) in empty_ids and initial_status(s) == ExecutionStatus.EMPTY]
    over = sum(1 for s in population if s.final_status == ExecutionStatus.SUCCESS)
    return EOCRResult(eocr_pct=_pct(over, len(population)), eocr_population=len(population),
                      eocr_overcorrected=over)


# --------------------------------------------------------------------------- #
# Cost and latency
# --------------------------------------------------------------------------- #
class CostLatencySummary(BaseModel):
    avg_latency_ms: Optional[float] = None    # wall clock per query (LLM + sandbox)
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    avg_sandbox_ms: float = 0.0               # SQL execution time per query
    avg_calls: float = 0.0                    # LLM calls per query
    avg_turns: float = 0.0                    # SQL executions per query
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_cost_usd: float = 0.0
    retry_distribution: Dict[str, int] = {}   # turns used -> number of queries


def _percentile(values: List[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[index]


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, prices: Optional[Dict[str, float]] = None) -> float:
    p = dict(DEFAULT_PRICES)
    p.update(prices or {})
    return (prompt_tokens * p["input_per_1m"] + completion_tokens * p["output_per_1m"]) / 1_000_000.0


def summarize_cost_latency(
    states: List[QueryEvaluationState],
    latencies_ms: Optional[List[float]] = None,
    prices: Optional[Dict[str, float]] = None,
) -> CostLatencySummary:
    n = len(states)
    if n == 0:
        return CostLatencySummary()
    prompt = sum(s.total_prompt_tokens for s in states)
    completion = sum(s.total_completion_tokens for s in states)
    sandbox_ms = sum(t.sandbox_result.execution_time_ms for s in states for t in s.turns if t.sandbox_result)
    cost = estimate_cost_usd(prompt, completion, prices)
    hist: Dict[str, int] = {}
    for s in states:
        key = str(len(s.turns))
        hist[key] = hist.get(key, 0) + 1
    lat = [x for x in (latencies_ms or []) if x is not None]
    return CostLatencySummary(
        avg_latency_ms=round(sum(lat) / len(lat), 2) if lat else None,
        p50_latency_ms=round(_percentile(lat, 0.5), 2) if lat else None,
        p95_latency_ms=round(_percentile(lat, 0.95), 2) if lat else None,
        avg_sandbox_ms=round(sandbox_ms / n, 3),
        avg_calls=round(sum(s.total_calls for s in states) / n, 3),
        avg_turns=round(sum(len(s.turns) for s in states) / n, 3),
        total_prompt_tokens=prompt,
        total_completion_tokens=completion,
        total_cost_usd=round(cost, 6),
        avg_cost_usd=round(cost / n, 6),
        retry_distribution=dict(sorted(hist.items())),
    )


# --------------------------------------------------------------------------- #
# Backwards-compatible aggregate (used by eval/run_bird.py)
# --------------------------------------------------------------------------- #
class BenchmarkMetrics(BaseModel):
    total_questions: int
    error_count: int
    empty_count: int
    success_count: int
    timeout_count: int
    error_rate_pct: float
    empty_rate_pct: float
    ex_pct: float
    total_calls: int
    avg_prompt_tokens: float
    avg_completion_tokens: float
    rdr_pct: Optional[float] = None


def compute_benchmark_metrics(states: List[QueryEvaluationState]) -> BenchmarkMetrics:
    total = len(states)
    if total == 0:
        return BenchmarkMetrics(
            total_questions=0, error_count=0, empty_count=0, success_count=0, timeout_count=0,
            error_rate_pct=0.0, empty_rate_pct=0.0, ex_pct=0.0, total_calls=0,
            avg_prompt_tokens=0.0, avg_completion_tokens=0.0,
        )
    error_count = sum(1 for s in states if s.final_status == ExecutionStatus.ERROR)
    empty_count = sum(1 for s in states if s.final_status == ExecutionStatus.EMPTY)
    timeout_count = sum(1 for s in states if s.final_status == ExecutionStatus.TIMEOUT)
    success_count = sum(1 for s in states if s.final_status == ExecutionStatus.SUCCESS)
    ex_count = sum(1 for s in states if s.execution_match)
    return BenchmarkMetrics(
        total_questions=total,
        error_count=error_count,
        empty_count=empty_count,
        success_count=success_count,
        timeout_count=timeout_count,
        error_rate_pct=round(100.0 * error_count / total, 2),
        empty_rate_pct=round(100.0 * empty_count / total, 2),
        ex_pct=round(100.0 * ex_count / total, 2),
        total_calls=sum(s.total_calls for s in states),
        avg_prompt_tokens=round(sum(s.total_prompt_tokens for s in states) / total, 1),
        avg_completion_tokens=round(sum(s.total_completion_tokens for s in states) / total, 1),
    )


def compute_repair_success_rate(repaired_states: List[QueryEvaluationState], initial_failures: int = 0) -> float:
    """Old ``compute_rdr``: successfully repaired queries / initial failures * 100."""
    if initial_failures == 0:
        return 0.0
    ok = sum(1 for s in repaired_states if s.execution_match)
    return round(100.0 * ok / initial_failures, 2)


# Backwards compatibility alias for Phase 1 tests
def compute_legacy_rdr(repaired_states: List[QueryEvaluationState], initial_failures: int = 0) -> float:
    return compute_repair_success_rate(repaired_states, initial_failures=initial_failures)


# --------------------------------------------------------------------------- #
# Everything at once, and mean / std over repeated runs
# --------------------------------------------------------------------------- #
def compute_all_metrics(
    states: List[QueryEvaluationState],
    gold_empty: Optional[Iterable[Any]] = None,
    latencies_ms: Optional[List[float]] = None,
    prices: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """One flat dictionary with EX, error rates, RDR, HRR, ERP, EOCR, cost and latency."""
    out = _dump(compute_benchmark_metrics(states))
    out.pop("rdr_pct", None)  # replaced by the real Repair Drift Rate below
    for part in (
        compute_initial_rates(states),
        compute_rdr(states),
        compute_hrr(states),
        compute_erp(states, gold_empty),
        compute_eocr(states, gold_empty),
        summarize_cost_latency(states, latencies_ms, prices),
    ):
        out.update(_dump(part))
    return out


def mean_std_summary(metric_dicts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Mean and sample standard deviation of every numeric metric over repeated runs.

    None values are skipped, so a rate that is undefined in one run only uses the other runs.
    """
    keys: List[str] = []
    for d in metric_dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in keys:
                keys.append(k)
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for k in keys:
        vals = [d[k] for d in metric_dicts
                if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool)]
        if not vals:
            continue
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[k] = {"mean": round(statistics.fmean(vals), 4), "std": round(std, 4), "n": len(vals)}
    return summary
