"""Command-line interface for SQLHeal.

    python -m sqlheal.cli --db path/to/db.sqlite --question "..." [--sql "..."] [--model gpt-4o-mini]
                          [--disable-patcher] [--disable-prober] [--disable-verifier]

Prints the colored step-by-step healing process, the clause table of every turn (changed
clauses highlighted), the final SQL and result rows, and saves a JSON trajectory to ``logs/``.
Exit code: 0 = final SUCCESS, 1 = final EMPTY, 2 = final ERROR/TIMEOUT or setup failure.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from sqlheal.agent import SQLHealAgent, TrajectoryStep
from sqlheal.parsing import print_clause_table
from sqlheal.reconstruct import diff_clauses
from sqlheal.state import ExecutionStatus

_CODES = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36",
          "gray": "90", "bold": "1"}
_ACTION_COLOR = {"generate": "blue", "execute": "cyan", "patch": "magenta", "rewrite": "yellow",
                 "verify": "magenta", "probe_and_heal": "magenta", "accept": "green", "stop": "red"}
_STATUS_COLOR = {ExecutionStatus.SUCCESS: "green", ExecutionStatus.EMPTY: "yellow",
                 ExecutionStatus.ERROR: "red", ExecutionStatus.TIMEOUT: "red"}


class Painter:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled:
            return text
        return "".join(f"\033[{_CODES[s]}m" for s in styles) + text + "\033[0m"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sqlheal.cli",
                                description="SQLHeal: agentic NL-to-SQL with localized self-healing.")
    p.add_argument("--db", required=True, help="path to the SQLite database (opened read-only)")
    p.add_argument("--question", required=True, help="natural-language question")
    p.add_argument("--sql", help="candidate SQL to start from (otherwise the LLM generates it)")
    p.add_argument("--evidence", default="", help="optional external knowledge / hint (BIRD 'evidence')")
    p.add_argument("--model", default=os.getenv("SQLHEAL_MODEL", "gpt-4o-mini"), help="LLM model name")
    p.add_argument("--max-turns", type=int, default=3, help="maximum SQL executions (default 3)")
    p.add_argument("--disable-patcher", action="store_true", help="ablation: whole-query regeneration on errors")
    p.add_argument("--disable-prober", action="store_true", help="ablation: accept EMPTY results as final")
    p.add_argument("--disable-verifier", action="store_true", help="ablation: probe every EMPTY result unguarded")
    p.add_argument("--log-dir", default="logs", help="where to write the JSON trajectory (default logs/)")
    p.add_argument("--no-log", action="store_true", help="do not write a trajectory file")
    p.add_argument("--no-cache", action="store_true", help="disable the on-disk LLM response cache")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    p.add_argument("--max-rows", type=int, default=20, help="rows to print (default 20)")
    p.add_argument("--mock", action="store_true", help="offline mock LLM (sets SQLHEAL_MOCK=1; for smoke tests only)")
    return p


def _clip(text: Optional[str], n: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def make_listener(paint: Painter):
    def on_step(s: TrajectoryStep) -> None:
        color = _ACTION_COLOR.get(s.action, "gray")
        print(paint(f"[Turn {s.turn}] step {s.step}: {s.action.upper()}", "bold", color)
              + paint(f"  via {s.tool}", "gray"))
        print(f"    thought : {s.thought}")
        if s.input_sql and s.action not in ("accept", "stop"):
            print(f"    input   : {_clip(s.input_sql)}")
        if s.output_sql and s.output_sql != s.input_sql:
            print(paint(f"    output  : {_clip(s.output_sql)}", "green"))
        if s.status is not None:
            rows = s.details.get("row_count", 0)
            extra = f", {rows} row(s)" if s.status == ExecutionStatus.SUCCESS else ""
            print("    status  : " + paint(s.status.value, "bold", _STATUS_COLOR[s.status])
                  + f"  ({s.execution_time_ms:.1f} ms{extra})")
        if s.error:
            print(paint(f"    error   : {_clip(s.error)}", "red"))
        changed = s.details.get("changed_clauses")
        if changed:
            print(paint(f"    changed : {', '.join(changed)}", "magenta")
                  + (paint(f"   DRIFT: {', '.join(s.details['drift_clauses'])}", "red")
                     if s.details.get("drift_clauses") else ""))
        if s.prompt_tokens or s.completion_tokens:
            print(paint(f"    tokens  : {s.prompt_tokens} prompt / {s.completion_tokens} completion", "gray"))
    return on_step


def print_clause_tables(state, paint: Painter) -> None:
    print(paint("\nClause tables per turn", "bold"))
    prev = None
    for t in state.turns:
        status = t.sandbox_result.status if t.sandbox_result else None
        changed = [c.value for c in diff_clauses(prev, t.sql)] if prev else []
        label = f"Turn {t.turn_index}" + (f"  [{status.value}]" if status else "")
        print(paint(label, "bold", _STATUS_COLOR.get(status, "gray")) +
              (paint(f"  changed vs previous turn: {', '.join(changed) or 'nothing'}", "magenta") if prev else ""))
        table = t.clause_table
        if table and table.clauses:
            for line in print_clause_table(table).splitlines():
                hit = any(f"| {c} " in line[:15] for c in changed)
                print(paint(line, "magenta") if hit else line)
        else:
            print(paint("  (query could not be parsed into clauses)", "gray"))
        prev = t.sql


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    paint = Painter(sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ)
    if not Path(args.db).is_file():
        print(paint(f"error: database not found: {args.db}", "red"), file=sys.stderr)
        return 2
    if args.mock:
        os.environ["SQLHEAL_MOCK"] = "1"

    from sqlheal.llm import LLMClient  # imported lazily so --help works without extras installed
    llm = LLMClient(model=args.model, cache_dir=None if args.no_cache else ".cache/llm")
    agent = SQLHealAgent(
        llm, max_turns=args.max_turns, model_name=args.model,
        enable_patcher=not args.disable_patcher, enable_prober=not args.disable_prober,
        enable_verifier=not args.disable_verifier, on_step=make_listener(paint),
    )

    flags = {"patcher": not args.disable_patcher, "prober": not args.disable_prober,
             "verifier": not args.disable_verifier}
    print(paint("SQLHeal", "bold", "cyan") + f"  db={args.db}  model={args.model}  max_turns={args.max_turns}")
    print("modules: " + "  ".join(paint(k + (" ON" if v else " OFF"), "green" if v else "red")
                                  for k, v in flags.items()))
    print(f"question: {args.question}\n")

    try:
        state = agent.run(args.db, args.question, sql=args.sql, evidence=args.evidence)
    except Exception as exc:  # noqa: BLE001
        print(paint(f"\nerror: {exc}", "red"), file=sys.stderr)
        return 2

    print_clause_tables(state, paint)
    print(paint("\nFinal SQL", "bold"))
    print(f"  {state.final_sql or '(none)'}")
    print(paint("Final status: ", "bold") + paint(state.final_status.value, "bold", _STATUS_COLOR[state.final_status]))
    res = agent.final_result
    if res is not None:
        if res.status == ExecutionStatus.SUCCESS:
            print(paint(f"\nResult rows ({len(res.rows or [])}):", "bold"))
            print("  " + " | ".join(res.columns or []))
            for row in (res.rows or [])[: args.max_rows]:
                print("  " + " | ".join(str(v) for v in row))
            if len(res.rows or []) > args.max_rows:
                print(paint(f"  ... {len(res.rows) - args.max_rows} more row(s)", "gray"))
        elif res.error_message:
            print(paint(f"Last error: {res.error_message}", "red"))
    print(paint(f"\nTurns: {len(state.turns)}  LLM calls: {state.total_calls}  tokens: "
                f"{state.total_prompt_tokens} prompt / {state.total_completion_tokens} completion", "gray"))
    if not args.no_log:
        path = agent.save_trajectory(Path(args.log_dir) / f"{agent.trajectory.run_id}.json")  # type: ignore[union-attr]
        print(paint(f"Trajectory saved to {path}", "gray"))
    return {ExecutionStatus.SUCCESS: 0, ExecutionStatus.EMPTY: 1}.get(state.final_status, 2)


if __name__ == "__main__":
    sys.exit(main())
