"""B1 Baseline: Iterative whole-query regeneration fixer baseline.
When B0 query execution fails or yields empty results, passes execution feedback
and re-generates the entire query up to max_repair_turns.
"""
from __future__ import annotations

from typing import Optional
from sqlheal.baselines.b0_single import B0Baseline, extract_sql_from_response
from sqlheal.llm import LLMClient
from sqlheal.parsing import extract_clause_table
from sqlheal.sandbox import SQLiteSandbox
from sqlheal.state import ExecutionStatus, QueryEvaluationState, QueryTurn


class B1FixerBaseline:
    def __init__(
        self,
        llm_client: LLMClient,
        sandbox: Optional[SQLiteSandbox] = None,
        max_repair_turns: int = 2,
    ):
        self.llm = llm_client
        self.sandbox = sandbox or SQLiteSandbox()
        self.max_repair_turns = max_repair_turns
        self.b0 = B0Baseline(llm_client=llm_client, sandbox=self.sandbox)

    def generate_repair_prompt(
        self,
        question: str,
        schema_ddl: str,
        failed_sql: str,
        status: ExecutionStatus,
        error_message: Optional[str] = None,
    ) -> str:
        prompt = (
            "You are a database expert fixing an invalid or incorrect SQLite query.\n\n"
            f"Database Schema:\n{schema_ddl}\n\n"
            f"Question:\n{question}\n\n"
            f"The previous query failed execution with status: {status.value}.\n"
            f"Failed SQL:\n```sql\n{failed_sql}\n```\n"
        )
        if error_message:
            prompt += f"Execution Error Message:\n{error_message}\n\n"
        elif status == ExecutionStatus.EMPTY:
            prompt += "Execution returned 0 rows (empty result set), but the question expects data.\n\n"

        prompt += "Regenerate the complete corrected SQL query. Return ONLY the new SQL in a ```sql block.\n\nCorrected SQL:"
        return prompt

    def run(
        self,
        question_id: int | str,
        question: str,
        db_path: str,
        db_id: str,
        schema_ddl: str = "",
        evidence: str = "",
        gold_sql: Optional[str] = None,
    ) -> QueryEvaluationState:
        # Run turn 1 (B0)
        state = self.b0.run(
            question_id=question_id,
            question=question,
            db_path=db_path,
            db_id=db_id,
            schema_ddl=schema_ddl,
            evidence=evidence,
            gold_sql=gold_sql,
        )

        current_turn = 1
        current_status = state.final_status
        current_sql = state.final_sql or ""
        last_error = state.turns[-1].sandbox_result.error_message if state.turns[-1].sandbox_result else None

        # If execution succeeded and (gold is matched or no error), return early
        while current_status in (ExecutionStatus.ERROR, ExecutionStatus.EMPTY, ExecutionStatus.TIMEOUT) and current_turn <= self.max_repair_turns:
            current_turn += 1
            repair_prompt = self.generate_repair_prompt(
                question=question,
                schema_ddl=schema_ddl,
                failed_sql=current_sql,
                status=current_status,
                error_message=last_error,
            )
            messages = [{"role": "user", "content": repair_prompt}]
            response = self.llm.generate(messages)
            raw_content = response["content"]
            repaired_sql = extract_sql_from_response(raw_content)

            clause_table = extract_clause_table(repaired_sql)
            sandbox_res = self.sandbox.execute(db_path, repaired_sql)

            turn = QueryTurn(
                turn_index=current_turn,
                sql=repaired_sql,
                clause_table=clause_table,
                sandbox_result=sandbox_res,
                prompt_tokens=response.get("prompt_tokens", 0),
                completion_tokens=response.get("completion_tokens", 0),
            )
            state.turns.append(turn)
            state.total_calls += 1
            state.total_prompt_tokens += response.get("prompt_tokens", 0)
            state.total_completion_tokens += response.get("completion_tokens", 0)
            state.final_sql = repaired_sql
            state.final_status = sandbox_res.status

            current_sql = repaired_sql
            current_status = sandbox_res.status
            last_error = sandbox_res.error_message

            if current_status == ExecutionStatus.SUCCESS:
                break

        # Check final execution match against gold SQL
        if gold_sql:
            gold_res = self.sandbox.execute(db_path, gold_sql)
            final_res = self.sandbox.execute(db_path, state.final_sql or "")
            if gold_res.status == ExecutionStatus.SUCCESS and final_res.status == ExecutionStatus.SUCCESS:
                try:
                    state.execution_match = set(gold_res.rows or []) == set(final_res.rows or [])
                except Exception:
                    state.execution_match = gold_res.rows == final_res.rows
            else:
                state.execution_match = False

        state.exact_match = (state.final_sql or "").strip().rstrip(";") == (gold_sql or "").strip().rstrip(";")
        return state
