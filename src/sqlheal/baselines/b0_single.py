"""B0 Baseline: Single-turn zero-shot text-to-SQL generator."""
from __future__ import annotations

import re
from typing import Optional
from sqlheal.llm import LLMClient
from sqlheal.parsing import extract_clause_table
from sqlheal.sandbox import SQLiteSandbox
from sqlheal.state import ExecutionStatus, QueryEvaluationState, QueryTurn


def extract_sql_from_response(text: str) -> str:
    """Extract SQL query from markdown block or raw text."""
    match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


class B0Baseline:
    def __init__(self, llm_client: LLMClient, sandbox: Optional[SQLiteSandbox] = None):
        self.llm = llm_client
        self.sandbox = sandbox or SQLiteSandbox()

    def generate_prompt(self, question: str, schema_ddl: str, evidence: str = "") -> str:
        prompt = (
            "You are a database expert. Given the SQLite schema, evidence and question, "
            "write a SQLite SQL query that answers the question. Return ONLY the SQL query in a ```sql block.\n\n"
            f"Database Schema:\n{schema_ddl}\n\n"
        )
        if evidence:
            prompt += f"Evidence / Hints:\n{evidence}\n\n"
        prompt += f"Question: {question}\n\nSQL Query:"
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
        prompt = self.generate_prompt(question, schema_ddl, evidence)
        messages = [{"role": "user", "content": prompt}]

        response = self.llm.generate(messages)
        raw_content = response["content"]
        predicted_sql = extract_sql_from_response(raw_content)

        # Parse clause table
        clause_table = extract_clause_table(predicted_sql)

        # Sandbox execution
        sandbox_res = self.sandbox.execute(db_path, predicted_sql)

        # Evaluate execution match if gold_sql is available
        ex_match = False
        if gold_sql:
            gold_res = self.sandbox.execute(db_path, gold_sql)
            if gold_res.status == ExecutionStatus.SUCCESS and sandbox_res.status == ExecutionStatus.SUCCESS:
                # Compare unordered set of row tuples
                try:
                    ex_match = set(gold_res.rows or []) == set(sandbox_res.rows or [])
                except Exception:
                    ex_match = gold_res.rows == sandbox_res.rows

        turn = QueryTurn(
            turn_index=1,
            sql=predicted_sql,
            clause_table=clause_table,
            sandbox_result=sandbox_res,
            prompt_tokens=response.get("prompt_tokens", 0),
            completion_tokens=response.get("completion_tokens", 0),
        )

        return QueryEvaluationState(
            question_id=question_id,
            db_id=db_id,
            question=question,
            gold_sql=gold_sql,
            turns=[turn],
            final_sql=predicted_sql,
            final_status=sandbox_res.status,
            exact_match=(predicted_sql.strip().rstrip(";") == (gold_sql or "").strip().rstrip(";")),
            EX=ex_match,
            total_calls=1,
            total_prompt_tokens=response.get("prompt_tokens", 0),
            total_completion_tokens=response.get("completion_tokens", 0),
        )
