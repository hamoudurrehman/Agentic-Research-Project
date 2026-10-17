"""Generate comparative reports and summary tables for B0 vs B1 runs."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def generate_report(b0_path: str, b1_path: str) -> str:
    with open(b0_path, "r") as f:
        b0_data = json.load(f)
    with open(b1_path, "r") as f:
        b1_data = json.load(f)

    b0_m = b0_data["metrics"]
    b1_m = b1_data["metrics"]
    model = b0_data.get("model", "Frontier LLM")

    table = [
        "| Model | System | ERROR rate (%) | EMPTY rate (%) | EX (%) | Questions |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| {model} | B0 | {b0_m['error_rate_pct']:.1f} | {b0_m['empty_rate_pct']:.1f} | {b0_m['ex_pct']:.1f} | {b0_m['total_questions']:,} |",
        f"| {model} | B1 | {b1_m['error_rate_pct']:.1f} | {b1_m['empty_rate_pct']:.1f} | {b1_m['ex_pct']:.1f} | {b1_m['total_questions']:,} |",
    ]
    report_text = "\n".join(table)
    print("\n=== Phase 1 Baseline Results ===")
    print(report_text)
    return report_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comparison report between B0 and B1")
    parser.add_argument("--b0", required=True, help="Path to B0 results.json")
    parser.add_argument("--b1", required=True, help="Path to B1 results.json")
    args = parser.parse_args()

    generate_report(args.b0, args.b1)
