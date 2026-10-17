"""SQLite read-only execution sandbox returning typed state."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional, Union

from sqlheal.state import ExecutionStatus, SandboxResult


class SQLiteSandbox:
    def __init__(self, timeout_sec: float = 30.0, max_rows: int = 1000):
        self.timeout_sec = timeout_sec
        self.max_rows = max_rows

    def execute(self, db_path: Union[str, Path], sql: str) -> SandboxResult:
        """Execute SQL query against a SQLite database in read-only mode."""
        clean_sql = sql.strip().rstrip(";")
        if not clean_sql:
            return SandboxResult(
                status=ExecutionStatus.ERROR,
                error_message="Empty SQL statement",
                execution_time_ms=0.0,
            )

        # Check file exists if not in-memory
        if db_path != ":memory:" and not Path(db_path).exists():
            return SandboxResult(
                status=ExecutionStatus.ERROR,
                error_message=f"Database file not found: {db_path}",
                execution_time_ms=0.0,
            )

        start_time = time.perf_counter()
        uri_path = db_path
        if db_path != ":memory:":
            abs_path = Path(db_path).resolve().as_posix()
            uri_path = f"file:{abs_path}?mode=ro"

        conn = None
        try:
            conn = sqlite3.connect(
                uri_path,
                uri=True,
                timeout=self.timeout_sec,
                check_same_thread=False,
            )
            cursor = conn.cursor()

            # Execute query
            cursor.execute(clean_sql)
            rows = cursor.fetchmany(self.max_rows)
            columns = [d[0] for d in cursor.description] if cursor.description else []

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if not rows:
                return SandboxResult(
                    status=ExecutionStatus.EMPTY,
                    rows=[],
                    columns=columns,
                    execution_time_ms=elapsed_ms,
                )

            return SandboxResult(
                status=ExecutionStatus.SUCCESS,
                rows=rows,
                columns=columns,
                execution_time_ms=elapsed_ms,
            )

        except sqlite3.OperationalError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = str(e)
            if "interrupted" in err_msg.lower() or "timeout" in err_msg.lower():
                return SandboxResult(
                    status=ExecutionStatus.TIMEOUT,
                    error_message=err_msg,
                    execution_time_ms=elapsed_ms,
                )
            return SandboxResult(
                status=ExecutionStatus.ERROR,
                error_message=err_msg,
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SandboxResult(
                status=ExecutionStatus.ERROR,
                error_message=str(e),
                execution_time_ms=elapsed_ms,
            )
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
