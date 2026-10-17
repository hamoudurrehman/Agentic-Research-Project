"""Fault injector supporting fault types F1 through F8.
Injects controlled syntax, schema, logic, join, aggregate, and clause-level faults
into valid SQL queries and tracks the target faulty clause.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple
from pydantic import BaseModel

import sqlglot
from sqlglot import exp

from sqlheal.parsing import parse_sql
from sqlheal.state import ClauseType


class FaultType(str, Enum):
    F1_SYNTAX_ERROR = "F1_SYNTAX_ERROR"                  # Mismatched quotes, typo in keyword
    F2_COLUMN_HALLUCINATION = "F2_COLUMN_HALLUCINATION"  # Non-existent column in SELECT or WHERE
    F3_TABLE_HALLUCINATION = "F3_TABLE_HALLUCINATION"    # Non-existent table in FROM
    F4_WRONG_OPERATOR = "F4_WRONG_OPERATOR"              # > changed to <, = to !=, LIKE to NOT LIKE
    F5_MISSING_JOIN_ON = "F5_MISSING_JOIN_ON"            # Cross join or drop join condition
    F6_WRONG_AGGREGATE = "F6_WRONG_AGGREGATE"            # COUNT swapped with SUM or AVG
    F7_INCORRECT_ORDER_LIMIT = "F7_INCORRECT_ORDER_LIMIT"# ASC/DESC flip or altered limit
    F8_MISSING_GROUP_BY = "F8_MISSING_GROUP_BY"          # Missing GROUP BY when aggregating non-agg cols


class FaultInjectionResult(BaseModel):
    original_sql: str
    corrupted_sql: str
    fault_type: FaultType
    faulty_clause: ClauseType
    description: str


class FaultInjector:
    def inject(self, sql: str, fault_type: FaultType) -> FaultInjectionResult:
        """Inject specified fault type into query."""
        parsed = parse_sql(sql)
        clean = sql.strip().rstrip(";")

        if fault_type == FaultType.F1_SYNTAX_ERROR:
            # Corrupt SELECT keyword or punctuation
            corrupted = clean.replace("SELECT", "SELECCT", 1) if "SELECT" in clean else clean + " WHERE ;;"
            return FaultInjectionResult(
                original_sql=clean,
                corrupted_sql=corrupted,
                fault_type=fault_type,
                faulty_clause=ClauseType.SELECT,
                description="Keyword typo SELECCT injected into SELECT clause.",
            )

        if fault_type == FaultType.F2_COLUMN_HALLUCINATION:
            # Change column or add non-existent column
            if parsed and parsed.expressions:
                # modify first select expr
                corrupted = clean.replace(parsed.expressions[0].sql(), "non_existent_column_xyz", 1)
                clause = ClauseType.SELECT
            else:
                corrupted = clean.replace("SELECT ", "SELECT non_existent_column_xyz, ", 1)
                clause = ClauseType.SELECT
            return FaultInjectionResult(
                original_sql=clean,
                corrupted_sql=corrupted,
                fault_type=fault_type,
                faulty_clause=clause,
                description="Replaced column with non-existent column in SELECT clause.",
            )

        if fault_type == FaultType.F3_TABLE_HALLUCINATION:
            # Replace table name in FROM
            if parsed and parsed.args.get("from"):
                from_table = parsed.args["from"].this.sql()
                corrupted = clean.replace(from_table, "non_existent_table_abc", 1)
            else:
                corrupted = clean + " FROM non_existent_table_abc"
            return FaultInjectionResult(
                original_sql=clean,
                corrupted_sql=corrupted,
                fault_type=fault_type,
                faulty_clause=ClauseType.FROM,
                description="Replaced table with non_existent_table_abc in FROM clause.",
            )

        if fault_type == FaultType.F4_WRONG_OPERATOR:
            # Flip comparison operators
            corrupted = clean
            replaced = False
            for op, replacement in [(" > ", " < "), (" = ", " != "), (" LIKE ", " NOT LIKE "), (">=", "<=")]:
                if op in corrupted:
                    corrupted = corrupted.replace(op, replacement, 1)
                    replaced = True
                    break
            if not replaced:
                corrupted = clean + " WHERE 1 = 0"
            return FaultInjectionResult(
                original_sql=clean,
                corrupted_sql=corrupted,
                fault_type=fault_type,
                faulty_clause=ClauseType.WHERE,
                description="Swapped comparison operator in WHERE condition.",
            )

        if fault_type == FaultType.F5_MISSING_JOIN_ON:
            # Corrupt JOIN condition
            if " ON " in clean:
                parts = clean.split(" ON ", 1)
                # Drop condition
                corrupted = parts[0] + " ON 1 = 1"
            else:
                corrupted = clean.replace(" FROM ", " FROM dummy_join_tbl CROSS JOIN ", 1)
            return FaultInjectionResult(
                original_sql=clean,
                corrupted_sql=corrupted,
                fault_type=fault_type,
                faulty_clause=ClauseType.FROM,
                description="Corrupted JOIN condition into trivial join.",
            )

        if fault_type == FaultType.F6_WRONG_AGGREGATE:
            # Swap COUNT -> SUM or SUM -> AVG
            corrupted = clean
            for agg, repl in [("COUNT(", "SUM("), ("AVG(", "COUNT("), ("SUM(", "AVG("), ("MAX(", "MIN(")]:
                if agg in corrupted.upper():
                    # Case-insensitive replace
                    idx = corrupted.upper().find(agg)
                    corrupted = corrupted[:idx] + repl + corrupted[idx + len(agg):]
                    break
            if corrupted == clean:
                corrupted = clean.replace("SELECT ", "SELECT COUNT(", 1).replace(" FROM", ") FROM", 1)
            return FaultInjectionResult(
                original_sql=clean,
                corrupted_sql=corrupted,
                fault_type=fault_type,
                faulty_clause=ClauseType.SELECT,
                description="Swapped aggregation function in SELECT.",
            )

        if fault_type == FaultType.F7_INCORRECT_ORDER_LIMIT:
            # Swap DESC with ASC, or flip LIMIT
            if "DESC" in clean.upper():
                idx = clean.upper().find("DESC")
                corrupted = clean[:idx] + "ASC" + clean[idx + 4:]
                clause = ClauseType.ORDER_BY
            elif "ASC" in clean.upper():
                idx = clean.upper().find("ASC")
                corrupted = clean[:idx] + "DESC" + clean[idx + 3:]
                clause = ClauseType.ORDER_BY
            elif "LIMIT 1" in clean.upper():
                corrupted = clean.replace("LIMIT 1", "LIMIT 100").replace("limit 1", "limit 100")
                clause = ClauseType.LIMIT
            else:
                corrupted = clean + " ORDER BY 1 DESC LIMIT 1"
                clause = ClauseType.ORDER_BY
            return FaultInjectionResult(
                original_sql=clean,
                corrupted_sql=corrupted,
                fault_type=fault_type,
                faulty_clause=clause,
                description="Inverted ordering direction or changed LIMIT.",
            )

        if fault_type == FaultType.F8_MISSING_GROUP_BY:
            # Drop GROUP BY clause
            if "GROUP BY" in clean.upper():
                idx = clean.upper().find("GROUP BY")
                # find end of group by (before order by, limit or end)
                rest = clean[idx:]
                end_keywords = ["ORDER BY", "HAVING", "LIMIT"]
                earliest = len(rest)
                for kw in end_keywords:
                    k_idx = rest.upper().find(kw)
                    if k_idx != -1 and k_idx < earliest:
                        earliest = k_idx
                corrupted = clean[:idx].strip() + (" " + rest[earliest:].strip() if earliest < len(rest) else "")
            else:
                corrupted = clean.replace("SELECT ", "SELECT MAX(1), ", 1)
            return FaultInjectionResult(
                original_sql=clean,
                corrupted_sql=corrupted,
                fault_type=fault_type,
                faulty_clause=ClauseType.GROUP_BY,
                description="Removed GROUP BY clause causing improper aggregation grouping.",
            )

        return FaultInjectionResult(
            original_sql=clean,
            corrupted_sql=clean,
            fault_type=fault_type,
            faulty_clause=ClauseType.SELECT,
            description="No mutation applied.",
        )
