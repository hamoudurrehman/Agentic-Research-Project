"""SQL parsing, clause extraction, clause table generation, and canonicalisation using sqlglot."""
from __future__ import annotations

from typing import Dict, List, Optional
import sqlglot
from sqlglot import exp

from sqlheal.state import ClauseInfo, ClauseTable, ClauseType


def parse_sql(sql: str, dialect: str = "sqlite") -> Optional[exp.Expression]:
    """Parse a SQL string into a sqlglot AST expression."""
    clean_sql = sql.strip().rstrip(";")
    try:
        return sqlglot.parse_one(clean_sql, read=dialect)
    except Exception:
        return None


def canonicalise_sql(sql: str, dialect: str = "sqlite") -> str:
    """Format and canonicalise SQL query using sqlglot."""
    expr = parse_sql(sql, dialect=dialect)
    if expr is None:
        return sql.strip()
    return expr.sql(dialect=dialect, pretty=False).strip()


def extract_clause_table(sql: str, dialect: str = "sqlite") -> ClauseTable:
    """Extract individual clauses (SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT)
    into a structured ClauseTable.
    """
    expr = parse_sql(sql, dialect=dialect)
    if expr is None:
        return ClauseTable(clauses=[])

    clauses: List[ClauseInfo] = []

    # SELECT
    select_exprs = expr.expressions
    if select_exprs:
        select_content = ", ".join(e.sql(dialect=dialect) for e in select_exprs)
        clauses.append(
            ClauseInfo(
                clause_type=ClauseType.SELECT,
                content=select_content,
                canonical=canonicalise_sql(f"SELECT {select_content}", dialect=dialect).replace("SELECT ", "", 1),
                ast_type="Select",
            )
        )

    # FROM & JOINs
    from_expr = expr.args.get("from_") or expr.args.get("from") or expr.find(exp.From)
    joins = expr.args.get("joins", [])
    if from_expr is not None:
        from_str = from_expr.sql(dialect=dialect)
        if joins:
            join_str = " ".join(j.sql(dialect=dialect) for j in joins)
            from_content = f"{from_str} {join_str}"
        else:
            from_content = from_str
        # Strip redundant leading "FROM " if from_str has it
        from_content = from_content.strip()
        clauses.append(
            ClauseInfo(
                clause_type=ClauseType.FROM,
                content=from_content,
                canonical=from_content,
                ast_type="From",
            )
        )

    # WHERE
    where_expr = expr.args.get("where")
    if where_expr is not None:
        where_content = where_expr.this.sql(dialect=dialect)
        clauses.append(
            ClauseInfo(
                clause_type=ClauseType.WHERE,
                content=where_content,
                canonical=where_content.strip(),
                ast_type="Where",
            )
        )

    # GROUP BY
    group_expr = expr.args.get("group")
    if group_expr is not None:
        group_content = ", ".join(e.sql(dialect=dialect) for e in group_expr.expressions)
        clauses.append(
            ClauseInfo(
                clause_type=ClauseType.GROUP_BY,
                content=group_content,
                canonical=group_content.strip(),
                ast_type="Group",
            )
        )

    # HAVING
    having_expr = expr.args.get("having")
    if having_expr is not None:
        having_content = having_expr.this.sql(dialect=dialect)
        clauses.append(
            ClauseInfo(
                clause_type=ClauseType.HAVING,
                content=having_content,
                canonical=having_content.strip(),
                ast_type="Having",
            )
        )

    # ORDER BY
    order_expr = expr.args.get("order")
    if order_expr is not None:
        order_content = ", ".join(e.sql(dialect=dialect) for e in order_expr.expressions)
        clauses.append(
            ClauseInfo(
                clause_type=ClauseType.ORDER_BY,
                content=order_content,
                canonical=order_content.strip(),
                ast_type="Order",
            )
        )

    # LIMIT
    limit_expr = expr.args.get("limit")
    if limit_expr is not None:
        limit_content = limit_expr.expression.sql(dialect=dialect)
        clauses.append(
            ClauseInfo(
                clause_type=ClauseType.LIMIT,
                content=limit_content,
                canonical=limit_content.strip(),
                ast_type="Limit",
            )
        )

    return ClauseTable(clauses=clauses)


def print_clause_table(clause_table: ClauseTable) -> str:
    """Format clause table for printing and logging."""
    lines = ["+------------+------------------------------------------------------+",
             "| Clause     | Content                                              |",
             "+------------+------------------------------------------------------+"]
    for c in clause_table.clauses:
        name = c.clause_type.value.ljust(10)
        content = c.content.replace("\n", " ")
        if len(content) > 52:
            content = content[:49] + "..."
        lines.append(f"| {name} | {content.ljust(52)} |")
    lines.append("+------------+------------------------------------------------------+")
    return "\n".join(lines)
