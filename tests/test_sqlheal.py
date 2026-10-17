"""Unit tests covering parsing, clause canonicalisation, sandbox, fault injector F1-F8, and metrics."""
import pytest
import sqlite3
import tempfile
from pathlib import Path

from sqlheal.parsing import parse_sql, canonicalise_sql, extract_clause_table
from sqlheal.state import ClauseType, ExecutionStatus, QueryEvaluationState, QueryTurn
from sqlheal.sandbox import SQLiteSandbox
from sqlheal.faults import FaultInjector, FaultType
from eval.metrics import compute_benchmark_metrics, compute_rdr


def test_clause_parsing_and_canonicalisation():
    sql = "SELECT id,   name, balance FROM  users  WHERE balance > 100 ORDER BY  name ASC LIMIT 5;"
    clause_table = extract_clause_table(sql)
    clauses = {c.clause_type: c.content for c in clause_table.clauses}

    assert ClauseType.SELECT in clauses
    assert "name" in clauses[ClauseType.SELECT]
    assert ClauseType.FROM in clauses
    assert "users" in clauses[ClauseType.FROM]
    assert ClauseType.WHERE in clauses
    assert "balance > 100" in clauses[ClauseType.WHERE]
    assert ClauseType.ORDER_BY in clauses
    assert ClauseType.LIMIT in clauses

    # Canonicalisation
    canonical = canonicalise_sql(sql)
    assert "SELECT" in canonical
    assert "LIMIT 5" in canonical


def test_sandbox_read_only_and_typed_states():
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        conn = sqlite3.connect(tmp.name)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_data (id INT, val TEXT);")
        cursor.execute("INSERT INTO test_data VALUES (1, 'apple'), (2, 'banana');")
        conn.commit()
        conn.close()

        sandbox = SQLiteSandbox()

        # Success case
        res_ok = sandbox.execute(tmp.name, "SELECT * FROM test_data WHERE id = 1;")
        assert res_ok.status == ExecutionStatus.SUCCESS
        assert len(res_ok.rows) == 1
        assert res_ok.rows[0] == (1, "apple")

        # Empty case
        res_empty = sandbox.execute(tmp.name, "SELECT * FROM test_data WHERE id = 999;")
        assert res_empty.status == ExecutionStatus.EMPTY
        assert res_empty.rows == []

        # Error case
        res_err = sandbox.execute(tmp.name, "SELECT nonexistent_col FROM test_data;")
        assert res_err.status == ExecutionStatus.ERROR
        assert res_err.error_message is not None


def test_fault_injector_f1_to_f8():
    injector = FaultInjector()
    sql = "SELECT name, balance FROM accounts WHERE balance > 500 GROUP BY name HAVING count(*) > 1 ORDER BY balance DESC LIMIT 5;"

    # F1: Syntax Error
    f1 = injector.inject(sql, FaultType.F1_SYNTAX_ERROR)
    assert f1.faulty_clause == ClauseType.SELECT
    assert "SELECCT" in f1.corrupted_sql or ";;" in f1.corrupted_sql

    # F2: Column Hallucination
    f2 = injector.inject(sql, FaultType.F2_COLUMN_HALLUCINATION)
    assert "non_existent_column_xyz" in f2.corrupted_sql

    # F3: Table Hallucination
    f3 = injector.inject(sql, FaultType.F3_TABLE_HALLUCINATION)
    assert "non_existent_table_abc" in f3.corrupted_sql

    # F4: Wrong Operator
    f4 = injector.inject(sql, FaultType.F4_WRONG_OPERATOR)
    assert "<" in f4.corrupted_sql or "!=" in f4.corrupted_sql or "1 = 0" in f4.corrupted_sql

    # F6: Wrong Aggregate
    f6 = injector.inject(sql, FaultType.F6_WRONG_AGGREGATE)
    assert "SUM" in f6.corrupted_sql or "AVG" in f6.corrupted_sql

    # F7: Incorrect Order / Limit
    f7 = injector.inject(sql, FaultType.F7_INCORRECT_ORDER_LIMIT)
    assert "ASC" in f7.corrupted_sql or "LIMIT" in f7.corrupted_sql

    # F8: Missing Group By
    f8 = injector.inject(sql, FaultType.F8_MISSING_GROUP_BY)
    assert "GROUP BY" not in f8.corrupted_sql


def test_metrics_computation_and_rdr():
    s1 = QueryEvaluationState(
        question_id=1,
        db_id="test",
        question="q1",
        final_status=ExecutionStatus.SUCCESS,
        EX=True,
        total_calls=1,
        total_prompt_tokens=100,
        total_completion_tokens=20,
    )
    s2 = QueryEvaluationState(
        question_id=2,
        db_id="test",
        question="q2",
        final_status=ExecutionStatus.ERROR,
        EX=False,
        total_calls=2,
        total_prompt_tokens=200,
        total_completion_tokens=40,
    )
    s3 = QueryEvaluationState(
        question_id=3,
        db_id="test",
        question="q3",
        final_status=ExecutionStatus.EMPTY,
        EX=False,
        total_calls=1,
        total_prompt_tokens=100,
        total_completion_tokens=20,
    )

    metrics = compute_benchmark_metrics([s1, s2, s3])
    assert metrics.total_questions == 3
    assert metrics.error_count == 1
    assert metrics.empty_count == 1
    assert metrics.success_count == 1
    assert metrics.ex_pct == pytest.approx(33.33, rel=1e-2)
    assert metrics.error_rate_pct == pytest.approx(33.33, rel=1e-2)
    assert metrics.empty_rate_pct == pytest.approx(33.33, rel=1e-2)

    # Test RDR
    rdr = compute_rdr([s1, s2], initial_failures=2)
    assert rdr == 50.0
