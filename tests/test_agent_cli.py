import json
from sqlheal.agent import SQLHealAgent
from sqlheal.cli import main
from sqlheal.state import ExecutionStatus as S


def j(**kw):
    return json.dumps(kw)


def test_error_is_patched_with_zero_drift(db, stub, tmp_path):
    llm = stub("```sql\ngender = 'M'\n```")
    agent = SQLHealAgent(llm)
    st = agent.run(db, "male clients", sql="SELECT name FROM client WHERE gendr = 'M' ORDER BY name",
                   gold_sql="SELECT name FROM client WHERE gender = 'M' ORDER BY name")
    assert st.final_status == S.SUCCESS and [t.sql for t in st.turns][-1].endswith("gender = 'M' ORDER BY name")
    assert len(st.turns) == 2 and st.execution_match and st.exact_match and st.total_calls == 1
    patch = next(s for s in agent.trajectory.steps if s.action == "patch")
    assert patch.details["changed_clauses"] == ["WHERE"] and patch.prompt_tokens == 10
    path = agent.save_trajectory(tmp_path / "t.json")
    data = json.loads(path.read_text())
    assert [s["action"] for s in data["steps"]][:3] == ["generate", "execute", "patch"]
    assert data["final"]["final_status"] == "SUCCESS"


def test_patcher_ablation_uses_whole_query_rewrite_and_measures_drift(db, stub):
    llm = stub("```sql\nSELECT name FROM client WHERE gender = 'M' ORDER BY id\n```")
    agent = SQLHealAgent(llm, enable_patcher=False)
    st = agent.run(db, "q", sql="SELECT name FROM client WHERE gendr = 'M' ORDER BY name")
    rw = next(s for s in agent.trajectory.steps if s.action == "rewrite")
    assert st.final_status == S.SUCCESS and set(rw.details["changed_clauses"]) == {"WHERE", "ORDER_BY"}
    assert rw.details["drift_clauses"] == ["ORDER_BY"]  # correct clause was mutated = Repair Drift


def test_patcher_failure_falls_back_to_rewrite(db, stub):
    llm = stub("gendr = 'M'", "gendr = 'M'", "SELECT name FROM client WHERE gender = 'M'")
    st = SQLHealAgent(llm).run(db, "q", sql="SELECT name FROM client WHERE gendr = 'M'")
    assert st.final_status == S.SUCCESS and st.total_calls == 3


def test_empty_is_healed_by_prober_with_verifier_gate(db, stub):
    llm = stub(j(verdict="MISMATCH", confidence=0.9, reason="encoding"), j(choices=[{"id": 0, "value": "M"}]))
    agent = SQLHealAgent(llm)
    st = agent.run(db, "male clients", sql="SELECT name FROM client WHERE gender = 'Male'")
    acts = [s.action for s in agent.trajectory.steps]
    assert st.final_status == S.SUCCESS and acts.index("verify") < acts.index("probe_and_heal")
    assert st.final_sql.endswith("gender = 'M'")


def test_expected_empty_is_left_alone_and_ablations(db, stub):
    sql = "SELECT name FROM client WHERE gender = 'Male'"
    guarded = SQLHealAgent(stub(j(verdict="EXPECTED_EMPTY", confidence=0.9, reason="no such")))
    st = guarded.run(db, "q", sql=sql)
    assert st.final_status == S.EMPTY and st.final_sql == sql and len(st.turns) == 1
    off = SQLHealAgent(stub(), enable_prober=False).run(db, "q", sql=sql)
    assert off.final_status == S.EMPTY and off.total_calls == 0
    unguarded = SQLHealAgent(stub(j(choices=[{"id": 0, "value": "F"}])), enable_verifier=False)
    assert unguarded.run(db, "q", sql=sql).final_status == S.SUCCESS  # no gate: heals


def test_generates_initial_sql_and_respects_max_turns(db, stub):
    st = SQLHealAgent(stub("```sql\nSELECT name FROM client\n```")).run(db, "all names")
    assert st.final_status == S.SUCCESS and st.turns[0].prompt_tokens == 10
    bad = SQLHealAgent(stub("```sql\nSELECT x FROM nope\n```", "```sql\nSELECT y FROM nope2\n```"), max_turns=1)
    assert len(bad.run(db, "q", sql="SELECT z FROM nope3").turns) == 1


def test_cli_smoke(db, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(["--db", db, "--question", "names", "--sql", "SELECT name FROM client", "--no-color", "--no-cache"])
    out = capsys.readouterr().out
    assert code == 0 and "Final status: SUCCESS" in out and "Ann" in out and "Clause tables per turn" in out
    assert list((tmp_path / "logs").glob("*.json"))
    assert main(["--db", "/nope.sqlite", "--question", "q"]) == 2
