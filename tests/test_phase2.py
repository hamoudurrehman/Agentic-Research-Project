"""Phase 2 tests: metrics, SQLHealAgent on synthetic cases, and the experiment pipeline.

Run from the repository root:   pytest tests/test_phase2.py -q

* Metric tests build QueryEvaluationState objects by hand, so they need no model and no database.
* Agent tests use the offline MockLLM from experiments/run_experiments.py and a tiny SQLite file.
  They check outcomes (final status, EX), not which module did the repair.
* Pipeline tests run the whole CLI with --mock, which must never crash (CI check).
"""
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.metrics import (  # noqa: E402
    compute_all_metrics,
    compute_eocr,
    compute_erp,
    compute_ex,
    compute_hrr,
    compute_rdr,
    estimate_cost_usd,
    execution_match,
    mean_std_summary,
    summarize_cost_latency,
)
from experiments import run_experiments as rx  # noqa: E402
from sqlheal.agent import SQLHealAgent  # noqa: E402
from sqlheal.sandbox import SQLiteSandbox  # noqa: E402
from sqlheal.state import ExecutionStatus, QueryEvaluationState, QueryTurn, SandboxResult  # noqa: E402

S, E, X = ExecutionStatus.SUCCESS, ExecutionStatus.EMPTY, ExecutionStatus.ERROR


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_state(qid: str, turns: List[Tuple[str, ExecutionStatus]], gold: Optional[str], ex: bool = False,
               tokens: Tuple[int, int] = (0, 0)) -> QueryEvaluationState:
    """Build a finished state. ``turns`` is a list of (sql, status); the last turn is the final one."""
    built = [
        QueryTurn(turn_index=i + 1, sql=sql, sandbox_result=SandboxResult(status=st, rows=[(1,)] if st == S else []))
        for i, (sql, st) in enumerate(turns)
    ]
    return QueryEvaluationState(
        question_id=qid, db_id="d", question="q", gold_sql=gold, turns=built, final_sql=turns[-1][0],
        final_status=turns[-1][1], execution_match=ex, total_calls=len(turns),
        total_prompt_tokens=tokens[0], total_completion_tokens=tokens[1],
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_execution_match_counts_two_empty_results_as_equal():
    ok = SandboxResult(status=S, rows=[(1,), (2,)])
    same = SandboxResult(status=S, rows=[(2,), (1,)])
    empty = SandboxResult(status=E, rows=[])
    err = SandboxResult(status=X, error_message="boom")
    assert execution_match(ok, same)
    assert execution_match(empty, SandboxResult(status=E, rows=[]))
    assert not execution_match(ok, empty)
    assert not execution_match(err, err)
    assert not execution_match(None, ok)


def test_ex_percentage():
    states = [make_state("1", [("SELECT 1", S)], None, ex=True), make_state("2", [("SELECT 2", S)], None, ex=False)]
    assert compute_ex(states) == 50.0
    assert compute_ex([]) is None


def test_rdr_counts_only_clauses_that_were_correct_before_repair():
    gold = "SELECT name FROM t WHERE x = 2 ORDER BY name"
    before = "SELECT name FROM t WHERE x = 1 ORDER BY name"
    drifted = make_state("a", [(before, S), ("SELECT name FROM t WHERE x = 2 ORDER BY name DESC", S)], gold)
    clean = make_state("b", [(before, S), (gold, S)], gold)
    r = compute_rdr([drifted, clean])
    # correct before repair: SELECT, FROM, ORDER_BY in each query (WHERE was the broken clause)
    assert r.rdr_correct_clauses == 6
    assert r.rdr_drifted_clauses == 1          # only ORDER BY changed in "drifted"
    assert r.rdr_pct == pytest.approx(16.67)
    assert r.rdr_query_pct == 50.0
    assert r.rdr_repaired_queries == 2


def test_rdr_skips_unrepaired_and_unscored_queries():
    gold = "SELECT a FROM t WHERE x = 1"
    no_repair = make_state("a", [(gold, S)], gold)
    no_gold = make_state("b", [("SELECT a FROM t", X), ("SELECT a FROM t WHERE x = 1", S)], None)
    union = make_state("c", [("SELECT a FROM t UNION SELECT b FROM t", X), (gold, S)], gold)
    r = compute_rdr([no_repair, no_gold, union])
    assert r.rdr_repaired_queries == 0
    assert r.rdr_pct is None
    assert r.rdr_excluded_queries == 1  # the UNION query was repaired but cannot be split


def test_rdr_removed_correct_clause_counts_as_drift():
    gold = "SELECT a FROM t WHERE x = 1 LIMIT 5"
    broken = "SELECT b FROM t WHERE x = 1 LIMIT 5"
    repaired = "SELECT a FROM t WHERE x = 1"  # LIMIT was correct and got dropped
    r = compute_rdr([make_state("a", [(broken, X), (repaired, S)], gold)])
    assert r.rdr_drifted_clauses == 1
    assert r.rdr_correct_clauses == 3


def test_hrr_only_looks_at_initial_errors():
    recovered = make_state("1", [("bad", X), ("good", S)], None, ex=True)
    wrong_result = make_state("2", [("bad", X), ("other", S)], None, ex=False)
    failed = make_state("3", [("bad", X), ("bad2", X)], None)
    fine = make_state("4", [("ok", S)], None, ex=True)
    r = compute_hrr([recovered, wrong_result, failed, fine])
    assert r.hrr_initial_errors == 3
    assert r.hrr_recovered == 2
    assert r.hrr_pct == pytest.approx(66.67)
    assert r.hrr_ex_pct == pytest.approx(33.33)
    assert compute_hrr([fine]).hrr_pct is None


def test_erp_and_sfpr():
    healed = make_state("1", [("q", E), ("q2", S)], None, ex=True)       # edit, recovered
    bad_edit = make_state("2", [("q", E), ("q3", S)], None, ex=False)    # edit, wrong result
    untouched = make_state("3", [("q", E)], None, ex=False)              # empty accepted, no edit
    gold_empty_edit = make_state("4", [("q", E), ("q4", S)], None, ex=False)  # edit on a true empty
    r = compute_erp([healed, bad_edit, untouched, gold_empty_edit], gold_empty={"4"})
    assert r.erp_edits == 3
    assert r.erp_recovered == 1
    assert r.erp_pct == pytest.approx(33.33)
    # non-empty gold: ids 1, 2, 3 all start empty; only id 1 ends correct
    assert r.recovery_candidates == 3
    assert r.recovery_pct == pytest.approx(33.33)
    assert r.sfpr_before_pct == 100.0
    assert r.sfpr_after_pct == pytest.approx(33.33)  # only id 3 still ends empty
    assert r.sfpr_rel_reduction_pct == pytest.approx(66.67)


def test_erp_without_edits_is_undefined():
    r = compute_erp([make_state("1", [("q", S)], None)])
    assert r.erp_pct is None and r.sfpr_before_pct == 0.0 and r.sfpr_rel_reduction_pct is None


def test_eocr_counts_only_true_empties_that_started_empty():
    kept = make_state("1", [("q", E)], None, ex=True)                 # guard kept the empty answer
    broken = make_state("2", [("q", E), ("q2", S)], None, ex=False)   # over-corrected
    not_empty_start = make_state("3", [("q", S)], None, ex=True)      # not in E*
    other = make_state("4", [("q", E), ("q2", S)], None, ex=True)     # gold is non-empty, so ignored
    r = compute_eocr([kept, broken, not_empty_start, other], gold_empty={"1", "2", "3"})
    assert r.eocr_population == 2
    assert r.eocr_overcorrected == 1
    assert r.eocr_pct == 50.0
    assert compute_eocr([other], gold_empty=set()).eocr_pct is None


def test_cost_and_latency_summary():
    states = [make_state("1", [("a", S)], None, tokens=(1_000_000, 1_000_000)),
              make_state("2", [("a", X), ("b", S)], None)]
    summary = summarize_cost_latency(states, latencies_ms=[100.0, 300.0])
    assert summary.avg_latency_ms == 200.0
    assert summary.total_prompt_tokens == 1_000_000
    assert summary.total_cost_usd == pytest.approx(0.75)  # default prices 0.15 in / 0.60 out per 1M
    assert summary.retry_distribution == {"1": 1, "2": 1}
    assert estimate_cost_usd(1_000_000, 0, {"input_per_1m": 1.0}) == pytest.approx(1.0)
    assert summarize_cost_latency([]).avg_calls == 0.0


def test_compute_all_metrics_handles_empty_and_normal_input():
    assert compute_all_metrics([])["total_questions"] == 0
    full = compute_all_metrics([make_state("1", [("SELECT 1", S)], None, ex=True)], latencies_ms=[5.0])
    for key in ("ex_pct", "rdr_pct", "hrr_pct", "erp_pct", "eocr_pct", "avg_latency_ms", "total_cost_usd"):
        assert key in full
    assert full["ex_pct"] == 100.0


def test_mean_std_summary_skips_none_and_non_numbers():
    s = mean_std_summary([{"a": 10.0, "b": None, "h": {"1": 2}}, {"a": 20.0, "b": 4.0}])
    assert s["a"]["mean"] == 15.0 and s["a"]["n"] == 2
    assert s["b"] == {"mean": 4.0, "std": 0.0, "n": 1}
    assert "h" not in s


# --------------------------------------------------------------------------- #
# Gap-table judgement and dataset helpers
# --------------------------------------------------------------------------- #
def test_gap_judgement_thresholds():
    assert rx.judge_gap1(40.0, 15.0, 60.0, 59.0) == "Fully"
    assert rx.judge_gap1(40.0, 30.0, 60.0, 60.0) == "Partially"
    assert rx.judge_gap1(40.0, 45.0, 60.0, 60.0) == "Not"
    assert rx.judge_gap1(0.0, 0.0, 60.0, 60.0).startswith("Not")
    assert rx.judge_gap1(None, 1.0, 1.0, 1.0) == "Not evaluated"
    assert rx.judge_gap2(10.0, 20.0, 75.0, 85.0) == "Fully"
    assert rx.judge_gap2(10.0, 20.0, 50.0, 60.0) == "Partially"
    assert rx.judge_gap3(5.0, 60.0, 80.0, 82.0) == "Fully"
    assert rx.judge_gap3(30.0, 60.0, 80.0, 82.0) == "Partially"
    assert rx.judge_gap3(70.0, 60.0, 80.0, 82.0) == "Not"


def test_latex_escaping_and_tables():
    assert rx.latex_escape("50% & a_b #1") == "50\\% \\& a\\_b \\#1"
    md = rx.markdown_table(["A", "B"], [["1", "2"]])
    assert md.splitlines()[0] == "| A | B |" and md.splitlines()[2] == "| 1 | 2 |"


def test_literal_and_absent_value_builders():
    gold = "SELECT name FROM t WHERE city = 'Austin' AND year = 2021"
    assert rx.make_literal_candidate(gold, "case") == "SELECT name FROM t WHERE city = 'aUSTIN' AND year = 2021"
    assert rx.make_literal_candidate(gold, "space") == "SELECT name FROM t WHERE city = 'Austin ' AND year = 2021"
    assert rx.make_literal_candidate("SELECT 1 FROM t WHERE a = 5", "case") is None
    new_gold, absent = rx.make_absent_gold(gold, "7")
    assert absent == "zz_absent_7" and "'zz_absent_7'" in new_gold


def test_scripted_first_call_llm_returns_candidate_once():
    mock = rx.MockLLM()
    wrapped = rx.ScriptedFirstCallLLM(mock, "SELECT 1")
    first = wrapped.generate([{"role": "user", "content": "anything"}])
    assert "SELECT 1" in first["content"] and first["prompt_tokens"] == 0
    wrapped.generate([{"role": "user", "content": "anything"}])
    assert mock.calls == 1  # only the second call reached the real model


# --------------------------------------------------------------------------- #
# SQLHealAgent on synthetic cases
# --------------------------------------------------------------------------- #
@pytest.fixture()
def world(tmp_path):
    items = rx.build_mock_items(tmp_path)
    sandbox = SQLiteSandbox()
    mock = rx.MockLLM()
    by_name = {}
    for item in items:
        item["gold_empty"] = sandbox.execute(item["db_path"], item["gold_sql"]).status == E
        mock.register(item)
        by_name[item["question_id"]] = item
    return mock, sandbox, by_name


def run_agent(world, qid: int, tmp_path: Path, **flags):
    mock, sandbox, items = world
    item = items[qid]
    agent = SQLHealAgent(mock, sandbox=sandbox, max_turns=3, **flags)
    state = agent.run(item["db_path"], item["question"], question_id=qid, gold_sql=item["gold_sql"])
    return agent, state


def test_agent_repairs_syntax_error(world, tmp_path):
    _, state = run_agent(world, 2, tmp_path)  # weak query starts with SELECCT
    assert state.turns[0].sandbox_result.status == X
    assert state.final_status == S and state.execution_match
    assert len(state.turns) >= 2


def test_agent_repairs_column_error(world, tmp_path):
    _, state = run_agent(world, 3, tmp_path)  # amt instead of amount
    assert state.turns[0].sandbox_result.status == X
    assert state.final_status == S and state.execution_match


def test_agent_repairs_case_mismatch_literal(world, tmp_path):
    """'austin' vs 'Austin': a case-only edit must not be rejected as an already-tried query."""
    _, state = run_agent(world, 4, tmp_path)
    assert state.turns[0].sandbox_result.status == E
    assert state.final_status == S and state.execution_match
    assert "'Austin'" in state.final_sql


def test_agent_keeps_expected_empty_result(world, tmp_path):
    _, state = run_agent(world, 5, tmp_path)  # no orders in 1999: the empty answer is correct
    assert state.turns[0].sandbox_result.status == E
    assert state.final_status == E and state.execution_match
    assert len(state.turns) == 1


def test_agent_without_prober_accepts_empty_result(world, tmp_path):
    _, state = run_agent(world, 4, tmp_path, enable_prober=False, enable_verifier=False)
    assert state.final_status == E and not state.execution_match


def test_agent_success_query_is_not_touched(world, tmp_path):
    _, state = run_agent(world, 1, tmp_path)
    assert len(state.turns) == 1 and state.execution_match


def test_agent_saves_trajectory(world, tmp_path):
    agent, _ = run_agent(world, 2, tmp_path)
    path = agent.save_trajectory(tmp_path / "traj" / "case2.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    actions = [s["action"] for s in data["steps"]]
    assert "execute" in actions and "accept" in actions
    assert data["final"]["final_status"] == "SUCCESS"


def test_agent_metrics_end_to_end(world, tmp_path):
    states = [run_agent(world, q, tmp_path)[1] for q in (2, 3, 4, 5)]
    gold_empty = {"5"}
    m = compute_all_metrics(states, gold_empty)
    assert m["ex_pct"] == 100.0
    assert m["hrr_pct"] == 100.0 and m["hrr_initial_errors"] == 2
    assert m["eocr_pct"] == 0.0 and m["eocr_population"] == 1
    assert m["erp_pct"] == 100.0  # the one empty-result edit (Austin) recovered the gold result


# --------------------------------------------------------------------------- #
# Experiment pipeline (--mock must never crash)
# --------------------------------------------------------------------------- #
def run_cli(tmp_path: Path, *extra: str) -> Path:
    out = tmp_path / "results"
    code = rx.main(["--mock", "--output-dir", str(out), "--log-dir", str(tmp_path / "logs"), *extra])
    assert code == 0
    return out


def test_all_experiments_run_with_mock(tmp_path):
    out = run_cli(tmp_path, "--experiment", "all")
    for exp_id in rx.EXPERIMENT_IDS:
        data = json.loads((out / exp_id / "results.json").read_text(encoding="utf-8"))
        assert data["experiment"] == exp_id
        assert data["config"]["mock"] is True and data["config"]["seed"] == 42
    e3 = json.loads((out / "E3" / "results.json").read_text(encoding="utf-8"))
    assert e3["arms"]["A3"]["natural"]["runs"][0]["crash_rate_pct"] == 0.0
    assert len(e3["queries"]) == 6
    e6 = json.loads((out / "E6" / "results.json").read_text(encoding="utf-8"))
    assert [r[0] for r in e6["efficiency_table"]["rows"]] == ["B0", "B1", "A3"]
    e5 = json.loads((out / "E5" / "results.json").read_text(encoding="utf-8"))
    assert [r[0] for r in e5["ablation_table"]["rows"]] == ["A1", "A2", "A3"]


def test_gap_resolution_table_files(tmp_path):
    out = run_cli(tmp_path, "--experiment", "E4")
    md = (out / "gap_resolution_table.md").read_text(encoding="utf-8")
    tex = (out / "gap_resolution_table.tex").read_text(encoding="utf-8")
    for gap in ("Gap 1", "Gap 2", "Gap 3"):
        assert gap in md and gap in tex
    assert "\\begin{tabular}" in tex and "Addressed?" in md
    e4 = json.loads((out / "E4" / "results.json").read_text(encoding="utf-8"))
    assert len(e4["gap_table"]["rows"]) == 3
    assert set(e4["gap1"]["arms"]) == {"B1", "A1"}
    assert set(e4["gap3"]["stress"]) == {"B1", "A2", "A3"}


def test_single_experiment_and_limit(tmp_path):
    out = run_cli(tmp_path, "--experiment", "e1", "--limit", "3")
    data = json.loads((out / "E1" / "results.json").read_text(encoding="utf-8"))
    assert data["arms"]["B0"]["natural"]["n_items"] == 3
    assert not (out / "E2").exists()


def test_repeated_runs_report_mean_and_std(tmp_path):
    out = run_cli(tmp_path, "--experiment", "E3", "--runs", "2")
    block = json.loads((out / "E3" / "results.json").read_text(encoding="utf-8"))["arms"]["A3"]["natural"]
    assert len(block["runs"]) == 2
    assert block["summary"]["ex_pct"]["n"] == 2 and "std" in block["summary"]["ex_pct"]


def test_trajectories_are_written(tmp_path):
    run_cli(tmp_path, "--experiment", "E3", "--traj-sample", "2")
    files = list((tmp_path / "logs" / "A3_natural").glob("*.json"))
    assert len(files) == 2


def test_missing_data_file_fails_cleanly(tmp_path):
    code = rx.main(["--experiment", "E1", "--data", str(tmp_path / "nope.json"), "--db-dir", str(tmp_path),
                    "--output-dir", str(tmp_path / "r"), "--log-dir", str(tmp_path / "l")])
    assert code == 1
