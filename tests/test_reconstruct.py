import pytest
from sqlheal.reconstruct import (ReconstructionError, diff_clauses, reconstruct_sql_from_clauses,
                                 replace_clause_in_sql, split_clauses)
from sqlheal.state import ClauseType as C

Q = "SELECT DISTINCT a, COUNT(*) FROM t JOIN u ON t.id = u.id WHERE x = 'ORDER BY' AND y IN (SELECT y FROM z WHERE q > 1) GROUP BY a HAVING COUNT(*) > 1 ORDER BY a DESC LIMIT 5 OFFSET 2;"


def test_split_ignores_keywords_in_strings_and_subqueries():
    s = split_clauses(Q)
    assert list(s.clauses) == [C.SELECT, C.FROM, C.WHERE, C.GROUP_BY, C.HAVING, C.ORDER_BY, C.LIMIT]
    assert s.clauses[C.SELECT] == "DISTINCT a, COUNT(*)"
    assert s.clauses[C.WHERE].startswith("x = 'ORDER BY'") and "(SELECT y FROM z" in s.clauses[C.WHERE]
    assert s.clauses[C.LIMIT] == "5 OFFSET 2"


def test_replace_keeps_every_other_clause_verbatim():
    out = replace_clause_in_sql(Q, C.HAVING, "HAVING COUNT(*) > 3")
    assert diff_clauses(Q, out) == [C.HAVING]
    assert out.endswith("LIMIT 5 OFFSET 2") and "DISTINCT" in out


def test_replace_can_add_and_remove_optional_clause():
    base = "SELECT a FROM t"
    added = replace_clause_in_sql(base, "where", "a = 1")
    assert added == "SELECT a FROM t WHERE a = 1"
    assert replace_clause_in_sql(added, C.WHERE, "") == base


def test_rejects_smuggled_clause_and_invalid_result():
    with pytest.raises(ReconstructionError):
        replace_clause_in_sql(Q, C.WHERE, "x = 1 ORDER BY y")
    with pytest.raises(ReconstructionError):
        replace_clause_in_sql("SELECT a FROM t", C.SELECT, "")
    with pytest.raises(ReconstructionError):
        replace_clause_in_sql("SELECT a FROM t UNION SELECT b FROM u", C.WHERE, "a=1")


def test_misspelled_select_is_tolerated_by_split():
    s = split_clauses("SELECCT id FROM t WHERE x = 1")
    assert s.clauses[C.SELECT] == "SELECCT id" and s.clauses[C.FROM] == "t"


def test_reconstruct_from_clause_dict_orders_and_strips_keywords():
    sql = reconstruct_sql_from_clauses({"LIMIT": "3", "FROM": "FROM t", "group by": "a", "SELECT": "a, COUNT(*)",
                                        "WHERE": "WHERE a > 1"})
    assert sql == "SELECT a, COUNT(*) FROM t WHERE a > 1 GROUP BY a LIMIT 3"
    assert reconstruct_sql_from_clauses({"SELECT": "a", "FROM": "t"}) == "SELECT a FROM t"
