"""BIRD dev runner: Executes B0 and B1 benchmarks against SQLite databases."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from typing import Any, Dict, List

from sqlheal.baselines.b0_single import B0Baseline
from sqlheal.baselines.b1_fixer import B1FixerBaseline
from sqlheal.llm import LLMClient
from sqlheal.parsing import print_clause_table
from sqlheal.sandbox import SQLiteSandbox
from eval.metrics import compute_benchmark_metrics


def run_bird_evaluation(
    data_path: str,
    db_root: str,
    system: str = "b0",
    config_path: str = "configs/default.json",
    limit: int = 0,
    output_dir: str = "results",
) -> Dict[str, Any]:
    # Load config
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)

    model = cfg.get("model", "gpt-4o-mini")
    temperature = cfg.get("temperature", 0.0)
    max_repair_turns = cfg.get("max_repair_turns", 2)
    timeout_sec = cfg.get("timeout_sec", 30.0)

    llm = LLMClient(model=model, temperature=temperature, timeout=timeout_sec)
    sandbox = SQLiteSandbox(timeout_sec=timeout_sec)

    runner = (
        B0Baseline(llm_client=llm, sandbox=sandbox)
        if system.lower() == "b0"
        else B1FixerBaseline(llm_client=llm, sandbox=sandbox, max_repair_turns=max_repair_turns)
    )

    # Load questions
    if not os.path.exists(data_path):
        print(f"Dataset file not found at {data_path}. Creating sample evaluation suite.")
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        # Generate representative sample
        samples = [
            {
                "question_id": 1,
                "db_id": "financial",
                "question": "What is the average transaction amount for active accounts?",
                "SQL": "SELECT AVG(amount) FROM transactions WHERE status = 'active';",
                "evidence": "Filter where status is active.",
            },
            {
                "question_id": 2,
                "db_id": "financial",
                "question": "Find the account owner with the highest balance.",
                "SQL": "SELECT owner_name FROM accounts ORDER BY balance DESC LIMIT 1;",
                "evidence": "Use accounts table.",
            },
        ]
        with open(data_path, "w") as f:
            json.dump(samples, f, indent=2)

    with open(data_path, "r") as f:
        questions = json.load(f)

    if limit > 0:
        questions = questions[:limit]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states = []
    print(f"Running {system.upper()} on {len(questions)} items using model {model}...")

    for i, item in enumerate(questions):
        qid = item.get("question_id", i)
        db_id = item.get("db_id", "sample")
        question_text = item.get("question", "")
        gold_sql = item.get("SQL", "")
        evidence = item.get("evidence", "")

        db_file = Path(db_root) / db_id / f"{db_id}.sqlite"
        if not db_file.exists():
            # In-memory database with standard mock schema if db file doesn't exist
            db_path = ":memory:"
        else:
            db_path = str(db_file)

        state = runner.run(
            question_id=qid,
            question=question_text,
            db_path=db_path,
            db_id=db_id,
            evidence=evidence,
            gold_sql=gold_sql,
        )
        states.append(state)

        # Print clause table for first turn
        if state.turns and state.turns[0].clause_table:
            print(f"\n--- Item {qid} ({db_id}) ---")
            print(f"Generated SQL: {state.turns[0].sql}")
            print(print_clause_table(state.turns[0].clause_table))

    metrics = compute_benchmark_metrics(states)

    results_file = out_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "system": system,
                "model": model,
                "metrics": metrics.model_dump() if hasattr(metrics, "model_dump") else metrics.dict(),
                "states": [s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in states],
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {results_file}")
    print(f"Total: {metrics.total_questions} | EX: {metrics.ex_pct}% | ERROR: {metrics.error_rate_pct}% | EMPTY: {metrics.empty_rate_pct}%")
    return {"metrics": metrics.model_dump() if hasattr(metrics, "model_dump") else metrics.dict(), "states": states}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BIRD benchmark on B0 or B1")
    parser.add_argument("--data", default="data/bird/dev.json", help="Path to dev.json")
    parser.add_argument("--db_root", default="data/bird/dev_databases", help="Path to database root directory")
    parser.add_argument("--system", default="b0", choices=["b0", "b1"], help="System baseline to run")
    parser.add_argument("--config", default="configs/default.json", help="Config file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of evaluation samples")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    args = parser.parse_args()

    run_bird_evaluation(
        data_path=args.data,
        db_root=args.db_root,
        system=args.system,
        config_path=args.config,
        limit=args.limit,
        output_dir=args.output_dir,
    )
