"""Evaluation metrics for Text-to-SQL and Repair Baselines.
Includes Execution Accuracy (EX), Error Rate, Empty Rate, and
Relative Degradation/Repair Rate (RDR) across fault injection sets.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel
from sqlheal.state import ExecutionStatus, QueryEvaluationState


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
            total_questions=0,
            error_count=0,
            empty_count=0,
            success_count=0,
            timeout_count=0,
            error_rate_pct=0.0,
            empty_rate_pct=0.0,
            ex_pct=0.0,
            total_calls=0,
            avg_prompt_tokens=0.0,
            avg_completion_tokens=0.0,
        )

    error_count = sum(1 for s in states if s.final_status == ExecutionStatus.ERROR)
    empty_count = sum(1 for s in states if s.final_status == ExecutionStatus.EMPTY)
    timeout_count = sum(1 for s in states if s.final_status == ExecutionStatus.TIMEOUT)
    success_count = sum(1 for s in states if s.final_status == ExecutionStatus.SUCCESS)

    ex_count = sum(1 for s in states if s.execution_match)

    total_calls = sum(s.total_calls for s in states)
    total_prompt_tok = sum(s.total_prompt_tokens for s in states)
    total_comp_tok = sum(s.total_completion_tokens for s in states)

    return BenchmarkMetrics(
        total_questions=total,
        error_count=error_count,
        empty_count=empty_count,
        success_count=success_count,
        timeout_count=timeout_count,
        error_rate_pct=round((error_count / total) * 100.0, 2),
        empty_rate_pct=round((empty_count / total) * 100.0, 2),
        ex_pct=round((ex_count / total) * 100.0, 2),
        total_calls=total_calls,
        avg_prompt_tokens=round(total_prompt_tok / total, 1),
        avg_completion_tokens=round(total_comp_tok / total, 1),
    )


def compute_rdr(repaired_states: List[QueryEvaluationState], initial_failures: int) -> float:
    """Compute Relative Degradation / Repair Rate (RDR):
    RDR = (Successfully Repaired Queries) / (Total Initial Faults or Failures) * 100.0
    """
    if initial_failures == 0:
        return 0.0
    successful_repairs = sum(1 for s in repaired_states if s.execution_match)
    return round((successful_repairs / initial_failures) * 100.0, 2)
