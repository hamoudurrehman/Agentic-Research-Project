from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    EMPTY = "EMPTY"
    TIMEOUT = "TIMEOUT"


class ClauseType(str, Enum):
    SELECT = "SELECT"
    FROM = "FROM"
    WHERE = "WHERE"
    GROUP_BY = "GROUP_BY"
    HAVING = "HAVING"
    ORDER_BY = "ORDER_BY"
    LIMIT = "LIMIT"


class ClauseInfo(BaseModel):
    clause_type: ClauseType
    content: str
    canonical: str
    ast_type: Optional[str] = None


class ClauseTable(BaseModel):
    clauses: List[ClauseInfo] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, str]:
        return {c.clause_type.value: c.content for c in self.clauses}


class SandboxResult(BaseModel):
    status: ExecutionStatus
    rows: Optional[List[tuple]] = None
    columns: Optional[List[str]] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


class QueryTurn(BaseModel):
    turn_index: int
    sql: str
    clause_table: Optional[ClauseTable] = None
    sandbox_result: Optional[SandboxResult] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class QueryEvaluationState(BaseModel):
    question_id: int | str
    db_id: str
    question: str
    gold_sql: Optional[str] = None
    turns: List[QueryTurn] = Field(default_factory=list)
    final_sql: Optional[str] = None
    final_status: ExecutionStatus = ExecutionStatus.EMPTY
    exact_match: bool = False
    execution_match: bool = Field(default=False, alias="EX")
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    model_config = {"populate_by_name": True}
