import pytest
from sqlheal.modules import ASTPatcher, PatchStatus, locate_error_clause
from sqlheal.reconstruct import diff_clauses
from sqlheal.state import ClauseType as C

SQ = "SELECT name FROM client WHERE gendr = 'M' ORDER BY name LIMIT 2"


@pytest.mark.parametrize("sql,err,clause", [
    (SQ, "no such column: gendr", C.WHERE),
    ("SELECT nme FROM client WHERE gender = 'M'", "no such column: client.nme", C.SELECT),
    ("SELECT name FROM clients WHERE gender = 'M'", "no such table: clients", C.FROM),
    ("SELECT name FROM client WHERE COUNT(*) > 1", "misuse of aggregate function COUNT()", C.WHERE),
    ("SELECCT name FROM client", 'near "SELECCT": syntax error', C.SELECT),
    ("SELECT name, FROM client", 'near "FROM": syntax error', C.SELECT),
    ("SELECT name FROM client WHERE gender = 'M' ORDER BY FOO(name)", "no such function: FOO", C.ORDER_BY),
])
def test_localisation(sql, err, clause):
    loc = locate_error_clause(sql, err)
    assert loc is not None and loc.clause == clause


def test_patch_rewrites_only_target_clause(stub):
    llm = stub("```sql\ngender = 'M'\n```")
    r = ASTPatcher(llm).patch("Names of male clients, 2 only", "client(id,name,gender,country)", SQ, "no such column: gendr")
    assert r.status == PatchStatus.PATCHED and r.target_clause == C.WHERE
    assert r.patched_sql == "SELECT name FROM client WHERE gender = 'M' ORDER BY name LIMIT 2"
    assert diff_clauses(SQ, r.patched_sql) == [C.WHERE] and r.frozen_intact
    assert "Frozen clauses" in llm.calls[0][1]["content"] and r.total_calls == 1 and r.prompt_tokens == 10


def test_patch_accepts_echoed_full_query_and_fixes_keyword_typo(stub):
    r = ASTPatcher(stub("SELECT name FROM client")).patch("q", "s", "SELECCT name FROM client", 'near "SELECCT": syntax error')
    assert r.applied and r.patched_sql == "SELECT name FROM client"


def test_patch_retries_then_rejects(stub):
    llm = stub("gendr = 'M'", "gendr = 'M'")  # identical to the broken clause twice
    r = ASTPatcher(llm, max_attempts=2).patch("q", "s", SQ, "no such column: gendr")
    assert r.status == PatchStatus.REJECTED and r.patched_sql == SQ and len(llm.calls) == 2


def test_unsupported_and_unlocalized(stub):
    p = ASTPatcher(stub())
    assert p.patch("q", "s", "SELECT a FROM t UNION SELECT b FROM u", "x").status == PatchStatus.UNSUPPORTED
    assert p.patch("q", "s", SQ, "disk I/O error").status == PatchStatus.UNLOCALIZED
