"""Master experiment runner for SQLHeal (Phase 2, experiments E1-E6).

Usage (from the repository root):

    python experiments/run_experiments.py --experiment all --limit 50 --mock
    python experiments/run_experiments.py --experiment E3 --data data/bird/dev.json \
        --db-dir data/bird/dev_databases --config configs/default.json --seed 42

Experiments (results are written to <output-dir>/<ID>/results.json):

    E1  B0 single-turn zero-shot baseline                      (natural BIRD set)
    E2  B1 CHASE-SQL-style whole-query fixer (base paper)      (natural BIRD set)
    E3  Full SQLHeal agent: patcher + prober + verifier         (natural BIRD set)
    E4  Gap-wise experiments
          Gap 1  RDR / HRR on fault-injected queries: ASTPatcher (A1) vs whole-query rewrite (B1)
          Gap 2  ERP / recovery on natural silent-empty queries and on injected literal faults
          Gap 3  EOCR on the empty-set stress set: unguarded (A2) vs guarded (A3) vs B1
    E5  Ablations A1 (patcher only), A2 (patcher + unguarded prober), A3 (full)
    E6  Efficiency and reliability across E1, E2, E3

Systems are compared on identical items. For the fault, literal and stress sets every system
starts from the same candidate SQL (the injected or gold query), so only the repair differs.
For B1 this is done by ScriptedFirstCallLLM, which returns the candidate as the "first
generation" and passes every later call to the real model.

Every system is scored with the same EX rule (eval.metrics.execution_match), so a correct
empty answer counts as correct for B0, B1 and the agent alike.

--mock replaces the model with a rule-based MockLLM and the BIRD data with a tiny synthetic
database. It exercises the whole pipeline without an API key; its numbers mean nothing.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.metrics import (  # noqa: E402
    DEFAULT_PRICES,
    compute_all_metrics,
    compute_ex,
    compute_hrr,
    compute_rdr,
    execution_match,
    final_result,
    mean_std_summary,
)
from sqlheal.agent import SQLHealAgent, get_schema_text  # noqa: E402
from sqlheal.baselines.b0_single import B0Baseline  # noqa: E402
from sqlheal.baselines.b1_fixer import B1FixerBaseline  # noqa: E402
from sqlheal.faults import FaultInjector, FaultType  # noqa: E402
from sqlheal.llm import LLMClient  # noqa: E402
from sqlheal.sandbox import SQLiteSandbox  # noqa: E402
from sqlheal.splits import ensure_fixed_splits  # noqa: E402
from sqlheal.state import ExecutionStatus, QueryEvaluationState  # noqa: E402

EXPERIMENT_IDS = ["E1", "E2", "E3", "E4", "E5", "E6"]
MAX_CONSECUTIVE_CRASHES = 10  # abort one run after this many crashes in a row

# System definitions. "agent" arms are SQLHealAgent with module toggles.
ARMS: Dict[str, Dict[str, Any]] = {
    "B0": {"kind": "b0", "label": "B0 single-turn zero-shot"},
    "B1": {"kind": "b1", "label": "B1 whole-query regeneration (CHASE-SQL fixer, re-implemented)"},
    "A1": {"kind": "agent", "patcher": True, "prober": False, "verifier": False,
           "label": "A1 patcher only"},
    "A2": {"kind": "agent", "patcher": True, "prober": True, "verifier": False,
           "label": "A2 patcher + unguarded prober"},
    "A3": {"kind": "agent", "patcher": True, "prober": True, "verifier": True,
           "label": "A3 full SQLHeal"},
}

# Success thresholds from proposal section B.6 (provisional, as stated there).
GAP1_RDR_RATIO = 0.5       # RDR(patcher) <= 0.5 * RDR(B1)
GAP1_HRR_MARGIN = 2.0      # HRR(patcher) >= HRR(B1) - 2 points
GAP2_MIN_RECOVERY = 80.0   # injected literal faults
GAP2_MIN_ERP = 70.0
GAP3_MAX_EOCR = 10.0
GAP3_MAX_RECOVERY_LOSS = 5.0


# =========================================================================== #
# Mock model and mock data (used by --mock and by the tests)
# =========================================================================== #
class MockLLM:
    """Rule-based stand-in for the model. Deterministic and offline.

    It finds which registered question a prompt is about and answers:
      * verifier prompts (contain EXPECTED_EMPTY or MISMATCH): a verdict from the gold emptiness
      * repair prompts (contain words like "failed" or "repair"): the gold SQL
      * everything else (initial generation): the item's weak SQL
    If the real modules use different prompt wording, adjust REPAIR_WORDS here.
    """

    REPAIR_WORDS = re.compile(r"previous query|failed|execution error|corrected|repair|fixing|patch|frozen|error:")

    def __init__(self) -> None:
        self.model = "mock-llm"
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.calls = 0

    def register(self, item: Dict[str, Any]) -> None:
        key = item["question"].strip().lower()
        self.registry.setdefault(key, item)

    def _find(self, prompt: str) -> Optional[Dict[str, Any]]:
        low = prompt.lower()
        hits = [k for k in self.registry if k in low]
        if not hits:
            return None
        return self.registry[max(hits, key=len)]

    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        self.calls += 1
        prompt = "\n".join(m.get("content", "") for m in messages)
        item = self._find(prompt)
        if item is None:
            text = "```sql\nSELECT 1\n```"
        elif "EXPECTED_EMPTY" in prompt or "MISMATCH" in prompt:
            verdict = "EXPECTED_EMPTY" if item.get("gold_empty") else "MISMATCH"
            text = json.dumps({"verdict": verdict, "reason": "mock verdict"})
        elif self.REPAIR_WORDS.search(prompt.lower()):
            text = f"```sql\n{item['gold_sql']}\n```"
        else:
            text = f"```sql\n{item.get('mock_weak_sql') or item['gold_sql']}\n```"
        return {"content": text, "prompt_tokens": max(1, len(prompt) // 4),
                "completion_tokens": max(1, len(text) // 4)}


class ScriptedFirstCallLLM:
    """Wraps a model so the first generate() call returns a fixed candidate SQL."""

    def __init__(self, real: Any, first_sql: str) -> None:
        self.real = real
        self.first_sql = first_sql
        self.used = False
        self.model = getattr(real, "model", "unknown")

    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        if not self.used:
            self.used = True
            return {"content": f"```sql\n{self.first_sql}\n```", "prompt_tokens": 0, "completion_tokens": 0}
        return self.real.generate(messages)


MOCK_QUESTIONS: List[Tuple[str, str, str]] = [
    # (question, gold SQL, weak SQL = what the "model" writes first)
    ("How many customers are in California?",
     "SELECT COUNT(*) FROM customers WHERE state = 'California'",
     "SELECT COUNT(*) FROM customers WHERE state = 'California'"),
    ("List the names of all customers in Texas.",
     "SELECT name FROM customers WHERE state = 'Texas'",
     "SELECCT name FROM customers WHERE state = 'Texas'"),
    ("What is the total order amount in 2021?",
     "SELECT SUM(amount) FROM orders WHERE year = 2021",
     "SELECT SUM(amt) FROM orders WHERE year = 2021"),
    ("List the names of customers who live in Austin.",
     "SELECT name FROM customers WHERE city = 'Austin'",
     "SELECT name FROM customers WHERE city = 'austin'"),
    ("Which customers placed an order in 1999?",
     "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id WHERE o.year = 1999",
     "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id WHERE o.year = 1999"),
    ("Which customers have an order above 500?",
     "SELECT DISTINCT c.name FROM customers c JOIN orders o ON o.customer_id = c.id "
     "WHERE o.amount > 500 ORDER BY c.name",
     "SELECT DISTINCT c.name FROM customers c JOIN orders o ON o.customer_id = c.id "
     "WHERE o.amount > 500 ORDER BY c.name"),
]


def create_mock_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    try:
        con.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, state TEXT, city TEXT)")
        con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL, year INTEGER)")
        people = [
            (1, "Ann", "California", "Fresno"), (2, "Bob", "Texas", "Austin"),
            (3, "Cara", "Texas", "Dallas"), (4, "Dev", "California", "Oakland"),
            (5, "Eli", "Ohio", "Akron"), (6, "Fay", "Texas", "Austin"),
        ]
        con.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", people)
        orders = [
            (1, 1, 120.0, 2021), (2, 2, 640.0, 2021), (3, 3, 80.0, 2020),
            (4, 4, 900.0, 2022), (5, 5, 45.0, 2021), (6, 6, 510.0, 2022),
        ]
        con.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders)
        con.commit()
    finally:
        con.close()


def build_mock_items(root: Path) -> List[Dict[str, Any]]:
    db_path = root / "mockdb" / "mockdb.sqlite"
    create_mock_database(db_path)
    items = []
    for i, (question, gold, weak) in enumerate(MOCK_QUESTIONS, start=1):
        items.append({"question_id": i, "db_id": "mockdb", "db_path": str(db_path), "question": question,
                      "evidence": "", "gold_sql": gold, "candidate_sql": None, "mock_weak_sql": weak, "meta": {}})
    return items


# =========================================================================== #
# Data loading and test-set builders
# =========================================================================== #
def load_bird_items(data_path: str, db_dir: str, limit: int, seed: int) -> Tuple[List[Dict[str, Any]], int]:
    """Read BIRD dev json. If data_path points to raw dev.json and splits exist/can be made,
    automatically uses the fixed evaluation split (split_eval.json) to respect Phase 2 Section 4.1.
    """
    path_obj = Path(data_path)
    # If the user passed dev.json, auto-check/generate fixed splits
    if path_obj.name == "dev.json":
        try:
            _, eval_path = ensure_fixed_splits(dev_json_path=str(path_obj), output_dir=str(path_obj.parent), seed=seed)
            if eval_path.exists():
                data_path = str(eval_path)
        except Exception:
            pass  # Fall back to raw dev.json if splitting cannot run

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if limit > 0 and len(raw) > limit:
        keep = sorted(random.Random(seed).sample(range(len(raw)), limit))
        raw = [raw[i] for i in keep]
    items: List[Dict[str, Any]] = []
    missing = 0
    for i, row in enumerate(raw):
        db_id = row.get("db_id", "")
        db_file = Path(db_dir) / db_id / f"{db_id}.sqlite"
        if not db_file.exists():
            missing += 1
            continue
        items.append({"question_id": row.get("question_id", i), "db_id": db_id, "db_path": str(db_file),
                      "question": row.get("question", ""), "evidence": row.get("evidence", ""),
                      "gold_sql": row.get("SQL", ""), "candidate_sql": None, "meta": {}})
    return items, missing


_LITERAL_RE = re.compile(r"=\s*'([^']*[A-Za-z][^']*)'")


def make_literal_candidate(gold_sql: str, kind: str) -> Optional[str]:
    """Corrupt the first string literal of an equality filter (case swap or trailing space)."""
    m = _LITERAL_RE.search(gold_sql)
    if not m:
        return None
    literal = m.group(1)
    new = literal.swapcase() if kind == "case" else literal + " "
    if new == literal:
        return None
    return gold_sql[: m.start(1)] + new + gold_sql[m.end(1):]


def make_absent_gold(gold_sql: str, tag: str) -> Optional[Tuple[str, str]]:
    """Replace the first equality literal with a value that does not exist in the column."""
    m = _LITERAL_RE.search(gold_sql)
    if not m:
        return None
    absent = f"zz_absent_{tag}"
    return gold_sql[: m.start(1)] + absent + gold_sql[m.end(1):], absent


def derive_item(base: Dict[str, Any], question_id: str, gold_sql: str, candidate_sql: Optional[str],
                meta: Dict[str, Any], question: Optional[str] = None, evidence: Optional[str] = None) -> Dict[str, Any]:
    item = dict(base)
    item.update({"question_id": question_id, "gold_sql": gold_sql, "candidate_sql": candidate_sql,
                 "meta": meta, "mock_weak_sql": None})
    if question is not None:
        item["question"] = question
    if evidence is not None:
        item["evidence"] = evidence
    return item


# =========================================================================== #
# Run bookkeeping
# =========================================================================== #
class RunData:
    """Everything measured in one pass of one system over one dataset."""

    def __init__(self, arm: str, dataset: str, run_idx: int) -> None:
        self.arm = arm
        self.dataset = dataset
        self.run_idx = run_idx
        self.states: List[QueryEvaluationState] = []
        self.latencies_ms: List[float] = []
        self.gold_empty: set = set()
        self.attempted = 0
        self.crashes = 0
        self.crash_messages: List[str] = []
        self.aborted = False


def pct(num: int, den: int) -> Optional[float]:
    return None if den == 0 else round(100.0 * num / den, 2)


class Context:
    """Shared state for all experiments: model, sandbox, datasets and cached runs."""

    def __init__(self, args: argparse.Namespace, cfg: Dict[str, Any], llm: Any, sandbox: SQLiteSandbox,
                 natural: List[Dict[str, Any]], mock_llm: Optional[MockLLM]) -> None:
        self.args = args
        self.cfg = cfg
        self.llm = llm
        self.mock_llm = mock_llm
        self.sandbox = sandbox
        self.max_repair_turns = int(cfg.get("max_repair_turns", 2))
        self.prices = dict(DEFAULT_PRICES)
        self.prices.update(cfg.get("pricing", {}))
        self.output_dir = Path(args.output_dir)
        self.log_dir = Path(args.log_dir)
        self.gold_cache: Dict[Tuple[str, str], Any] = {}
        self.schema_cache: Dict[str, str] = {}
        self.runs_cache: Dict[Tuple[str, str], List[RunData]] = {}
        self.block_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.traj_saved: Counter = Counter()
        self.dropped_gold = 0
        self.datasets: Dict[str, List[Dict[str, Any]]] = {}
        self.datasets["natural"] = self._prepare(natural, "any")

    # ----- gold handling ----------------------------------------------------
    def gold(self, db_path: str, sql: str) -> Any:
        key = (db_path, sql)
        if key not in self.gold_cache:
            self.gold_cache[key] = self.sandbox.execute(db_path, sql)
        return self.gold_cache[key]

    def _prepare(self, items: List[Dict[str, Any]], want: str) -> List[Dict[str, Any]]:
        """Execute gold SQL; drop items whose gold fails (or has the wrong emptiness for the set)."""
        kept = []
        for item in items:
            res = self.gold(item["db_path"], item["gold_sql"])
            item["gold_status"] = res.status.value
            item["gold_empty"] = res.status == ExecutionStatus.EMPTY
            wrong = (res.status not in (ExecutionStatus.SUCCESS, ExecutionStatus.EMPTY)
                     or (want == "success" and res.status != ExecutionStatus.SUCCESS)
                     or (want == "empty" and res.status != ExecutionStatus.EMPTY))
            if wrong:
                self.dropped_gold += 1
                continue
            if self.mock_llm is not None:
                self.mock_llm.register(item)
            kept.append(item)
        return kept

    # ----- dataset builders -------------------------------------------------
    def items(self, dataset: str) -> List[Dict[str, Any]]:
        if dataset not in self.datasets:
            builders = {"fault": self._build_fault, "literal": self._build_literal, "stress": self._build_stress}
            self.datasets[dataset] = builders[dataset]()
        return self.datasets[dataset]

    def _cap(self, default: int) -> int:
        return min(default, self.args.limit) if self.args.limit > 0 else default

    def _sample_bases(self, n: int, salt: int) -> List[Dict[str, Any]]:
        bases = [i for i in self.datasets["natural"] if i["gold_status"] == "SUCCESS"]
        rng = random.Random(self.args.seed + salt)
        return rng.sample(bases, min(n, len(bases)))

    def _build_fault(self) -> List[Dict[str, Any]]:
        """One corrupted candidate per (gold query, fault type) using faults.FaultInjector."""
        injector = FaultInjector()
        items: List[Dict[str, Any]] = []
        for base in self._sample_bases(self._cap(self.args.fault_sample), 1):
            for fault in FaultType:
                try:
                    result = injector.inject(base["gold_sql"], fault)
                except Exception:  # noqa: BLE001 - parse failures just skip this fault
                    continue
                if result.corrupted_sql.strip() == base["gold_sql"].strip().rstrip(";"):
                    continue
                meta = {"fault_type": fault.value, "faulty_clause": result.faulty_clause.value}
                items.append(derive_item(base, f"{base['question_id']}:{fault.name[:2]}", base["gold_sql"],
                                         result.corrupted_sql, meta))
        return self._prepare(items, "success")

    def _build_literal(self) -> List[Dict[str, Any]]:
        """Silent-empty faults: case swap or trailing space in an equality literal."""
        items: List[Dict[str, Any]] = []
        for k, base in enumerate(self._sample_bases(self._cap(self.args.fault_sample), 2)):
            order = ["case", "space"] if k % 2 == 0 else ["space", "case"]
            for kind in order:
                cand = make_literal_candidate(base["gold_sql"], kind)
                if cand is None:
                    continue
                if self.gold(base["db_path"], cand).status != ExecutionStatus.EMPTY:
                    continue  # not a silent-empty fault on this database
                items.append(derive_item(base, f"{base['question_id']}:L{kind[0]}", base["gold_sql"], cand,
                                         {"fault_type": f"LITERAL_{kind.upper()}"}))
                break
        return self._prepare(items, "success")

    def _build_stress(self) -> List[Dict[str, Any]]:
        """Gold-empty set: natural empty questions plus synthetic absent-value questions.

        The candidate is the gold query itself, so the first result is empty by construction
        and EOCR measures pure over-correction.
        """
        size = self._cap(self.args.stress_size)
        items: List[Dict[str, Any]] = []
        for base in self.datasets["natural"]:
            if base["gold_status"] == "EMPTY":
                items.append(derive_item(base, f"{base['question_id']}:N", base["gold_sql"], base["gold_sql"],
                                         {"source": "natural"}))
        items = items[:size]
        pool = self._sample_bases(len(self.datasets["natural"]), 3)
        for base in pool:
            if len(items) >= size:
                break
            made = make_absent_gold(base["gold_sql"], str(base["question_id"]))
            if made is None:
                continue
            gold, absent = made
            question = f"{base['question']} Use the value '{absent}' for this condition."
            items.append(derive_item(base, f"{base['question_id']}:S", gold, gold, {"source": "synthetic"},
                                     question=question, evidence=""))
        return self._prepare(items, "empty")

    # ----- misc -------------------------------------------------------------
    def schema(self, db_path: str) -> str:
        if db_path not in self.schema_cache:
            self.schema_cache[db_path] = get_schema_text(self.sandbox, db_path)
        return self.schema_cache[db_path]

    def config_record(self) -> Dict[str, Any]:
        return {
            "model": getattr(self.llm, "model", self.cfg.get("model")),
            "temperature": self.cfg.get("temperature", 0.0),
            "max_repair_turns": self.max_repair_turns,
            "max_turns": self.max_repair_turns + 1,
            "seed": self.args.seed,
            "limit": self.args.limit,
            "runs": self.args.runs,
            "mock": bool(self.args.mock),
            "prices_per_1m_tokens": self.prices,
            "fault_sample": self.args.fault_sample,
            "stress_size": self.args.stress_size,
            "dataset_sizes": {k: len(v) for k, v in self.datasets.items()},
            "gold_items_dropped": self.dropped_gold,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


# =========================================================================== #
# Running one system on one dataset
# =========================================================================== #
def make_agent(ctx: Context, spec: Dict[str, Any]) -> SQLHealAgent:
    return SQLHealAgent(
        ctx.llm, sandbox=ctx.sandbox, max_turns=ctx.max_repair_turns + 1,
        enable_patcher=spec["patcher"], enable_prober=spec["prober"], enable_verifier=spec["verifier"],
        model_name=getattr(ctx.llm, "model", None),
    )


def run_one(ctx: Context, spec: Dict[str, Any], agent: Optional[SQLHealAgent],
            item: Dict[str, Any]) -> QueryEvaluationState:
    db = item["db_path"]
    candidate = item.get("candidate_sql")
    schema = ctx.schema(db)
    qid = item["question_id"]
    if spec["kind"] == "b0":
        runner0 = B0Baseline(llm_client=ctx.llm, sandbox=ctx.sandbox)
        return runner0.run(question_id=qid, question=item["question"], db_path=db, db_id=item["db_id"],
                           schema_ddl=schema, evidence=item.get("evidence", ""), gold_sql=item["gold_sql"])
    if spec["kind"] == "b1":
        llm = ScriptedFirstCallLLM(ctx.llm, candidate) if candidate else ctx.llm
        runner1 = B1FixerBaseline(llm_client=llm, sandbox=ctx.sandbox, max_repair_turns=ctx.max_repair_turns)
        state = runner1.run(question_id=qid, question=item["question"], db_path=db, db_id=item["db_id"],
                            schema_ddl=schema, evidence=item.get("evidence", ""), gold_sql=item["gold_sql"])
        if candidate:
            state.total_calls = max(0, state.total_calls - 1)  # the scripted call was not a model call
        return state
    assert agent is not None
    return agent.run(db, item["question"], sql=candidate, question_id=qid, evidence=item.get("evidence", ""),
                     gold_sql=item["gold_sql"], schema=schema)


def run_arm(ctx: Context, arm: str, dataset: str, run_idx: int) -> RunData:
    spec = ARMS[arm]
    if spec["kind"] == "b0" and dataset != "natural":
        raise ValueError("B0 generates its own query, so it only runs on the natural set")
    items = ctx.items(dataset)
    run = RunData(arm, dataset, run_idx)
    agent = make_agent(ctx, spec) if spec["kind"] == "agent" else None
    in_a_row = 0
    for item in items:
        run.attempted += 1
        start = time.perf_counter()
        try:
            state = run_one(ctx, spec, agent, item)
            in_a_row = 0
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop the experiment
            run.crashes += 1
            in_a_row += 1
            if len(run.crash_messages) < 20:
                run.crash_messages.append(f"{item['question_id']}: {type(exc).__name__}: {exc}"[:300])
            if in_a_row >= MAX_CONSECUTIVE_CRASHES:
                run.aborted = True
                print(f"  [{arm}/{dataset}] aborted after {in_a_row} crashes in a row")
                break
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # Same EX rule for every system.
        gold_res = ctx.gold(item["db_path"], item["gold_sql"])
        state.execution_match = execution_match(final_result(state), gold_res)
        state.question_id = item["question_id"]
        run.states.append(state)
        run.latencies_ms.append(elapsed_ms)
        if item.get("gold_empty"):
            run.gold_empty.add(str(item["question_id"]))
        if agent is not None and run_idx == 0 and ctx.traj_saved[(arm, dataset)] < ctx.args.traj_sample:
            ctx.traj_saved[(arm, dataset)] += 1
            safe = re.sub(r"[^\w.-]+", "_", str(item["question_id"]))
            try:
                agent.save_trajectory(ctx.log_dir / f"{arm}_{dataset}" / f"{safe}.json")
            except Exception as exc:  # noqa: BLE001
                print(f"  could not save trajectory: {exc}")
    return run


def run_metrics(ctx: Context, run: RunData) -> Dict[str, Any]:
    m = compute_all_metrics(run.states, run.gold_empty, run.latencies_ms, ctx.prices)
    turns = [t for s in run.states for t in s.turns]
    timeouts = sum(1 for t in turns if t.sandbox_result and t.sandbox_result.status == ExecutionStatus.TIMEOUT)
    m["attempted"] = run.attempted
    m["crashes"] = run.crashes
    m["crash_rate_pct"] = pct(run.crashes, run.attempted)
    m["timeout_rate_pct"] = pct(timeouts, len(turns))
    m["aborted"] = 1 if run.aborted else 0
    return m


def get_block(ctx: Context, arm: str, dataset: str) -> Dict[str, Any]:
    """Run (or fetch from cache) ``arm`` on ``dataset`` for ctx.args.runs repetitions."""
    key = (arm, dataset)
    if key in ctx.block_cache:
        return ctx.block_cache[key]
    print(f"  running {arm} on {dataset} ({len(ctx.items(dataset))} items, {ctx.args.runs} run(s))")
    runs = [run_arm(ctx, arm, dataset, i) for i in range(ctx.args.runs)]
    ctx.runs_cache[key] = runs
    per_run = [run_metrics(ctx, r) for r in runs]
    crash_info = [m for r in runs for m in r.crash_messages][:20]
    block = {
        "label": ARMS[arm]["label"],
        "n_items": len(ctx.items(dataset)),
        "runs": per_run,
        "summary": mean_std_summary(per_run),
        "crash_messages": crash_info,
    }
    ctx.block_cache[key] = block
    return block


def mean(block: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not block:
        return None
    entry = block["summary"].get(key)
    return None if entry is None else entry["mean"]


def query_records(ctx: Context, arm: str, dataset: str) -> List[Dict[str, Any]]:
    """Compact per-query records of run 0 (kept in results.json for the failure analysis)."""
    runs = ctx.runs_cache.get((arm, dataset), [])
    if not runs:
        return []
    out = []
    for s in runs[0].states:
        first = s.turns[0] if s.turns else None
        out.append({
            "question_id": str(s.question_id), "db_id": s.db_id,
            "initial_sql": first.sql if first else None,
            "initial_status": first.sandbox_result.status.value if first and first.sandbox_result else None,
            "final_sql": s.final_sql, "final_status": s.final_status.value,
            "turns": len(s.turns), "ex": bool(s.execution_match), "llm_calls": s.total_calls,
            "prompt_tokens": s.total_prompt_tokens, "completion_tokens": s.total_completion_tokens,
        })
    return out


# =========================================================================== #
# Tables
# =========================================================================== #
def fmt(value: Optional[float], digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_ms(block: Optional[Dict[str, Any]], key: str, digits: int = 1) -> str:
    """'mean' for one run, 'mean +/- std' for several."""
    if not block or key not in block["summary"]:
        return "n/a"
    e = block["summary"][key]
    if e["n"] > 1:
        return f"{e['mean']:.{digits}f} +/- {e['std']:.{digits}f}"
    return f"{e['mean']:.{digits}f}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def latex_escape(text: str) -> str:
    for old, new in (("\\", "\\textbackslash{}"), ("%", "\\%"), ("&", "\\&"), ("_", "\\_"), ("#", "\\#")):
        text = text.replace(old, new)
    return text


def latex_table(headers: List[str], rows: List[List[str]], caption: str, label: str, widths: List[float]) -> str:
    spec = "".join(f"p{{{w:.2f}\\textwidth}}" for w in widths)
    lines = ["\\begin{table*}[t]", "\\centering", "\\small", f"\\caption{{{latex_escape(caption)}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{spec}}}", "\\hline",
             " & ".join(f"\\textbf{{{latex_escape(h)}}}" for h in headers) + " \\\\", "\\hline"]
    for r in rows:
        lines.append(" & ".join(latex_escape(c) for c in r) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


GAP_HEADERS = ["Gap", "Method / component", "Metric", "Base paper result", "Our result", "Addressed? (Fully / Partially / Not)"]


def judge_gap1(b1_rdr: Optional[float], our_rdr: Optional[float], b1_hrr: Optional[float],
               our_hrr: Optional[float]) -> str:
    if None in (b1_rdr, our_rdr, b1_hrr, our_hrr):
        return "Not evaluated"
    if b1_rdr == 0:
        return "Not (baseline shows no drift)"
    if our_rdr <= GAP1_RDR_RATIO * b1_rdr and our_hrr >= b1_hrr - GAP1_HRR_MARGIN:
        return "Fully"
    if our_rdr < b1_rdr:
        return "Partially"
    return "Not"


def judge_gap2(b1_erp: Optional[float], b1_rec: Optional[float], erp: Optional[float],
               rec: Optional[float]) -> str:
    if erp is None or rec is None:
        return "Not evaluated"
    if erp >= GAP2_MIN_ERP and rec >= GAP2_MIN_RECOVERY:
        return "Fully"
    if (b1_rec is None or rec > b1_rec) or (b1_erp is None or erp > b1_erp):
        return "Partially"
    return "Not"


def judge_gap3(eocr_guarded: Optional[float], eocr_unguarded: Optional[float], rec_guarded: Optional[float],
               rec_unguarded: Optional[float]) -> str:
    if eocr_guarded is None:
        return "Not evaluated"
    recovery_ok = rec_guarded is None or rec_unguarded is None or rec_guarded >= rec_unguarded - GAP3_MAX_RECOVERY_LOSS
    if eocr_guarded <= GAP3_MAX_EOCR and recovery_ok:
        return "Fully"
    if eocr_unguarded is not None and eocr_guarded < eocr_unguarded:
        return "Partially"
    return "Not"


def build_gap_rows(e4: Dict[str, Any]) -> List[List[str]]:
    """Gap-resolution table (manual section 4.4) from an E4 results dictionary."""
    g1 = e4.get("gap1", {}).get("arms", {})
    g2n = e4.get("gap2", {}).get("natural", {})
    g2l = e4.get("gap2", {}).get("literal", {})
    g3 = e4.get("gap3", {}).get("stress", {})
    b1f, a1f = g1.get("B1"), g1.get("A1")
    rows = [
        ["Gap 1: repair drift", "ASTPatcher (repair one clause, freeze the rest)",
         "RDR (lower is better) / HRR, fault-injected set",
         f"RDR {fmt_ms(b1f, 'rdr_pct')} / HRR {fmt_ms(b1f, 'hrr_pct')} (B1, our re-implementation)",
         f"RDR {fmt_ms(a1f, 'rdr_pct')} / HRR {fmt_ms(a1f, 'hrr_pct')}",
         judge_gap1(mean(b1f, "rdr_pct"), mean(a1f, "rdr_pct"), mean(b1f, "hrr_pct"), mean(a1f, "hrr_pct"))],
        ["Gap 2: silent empty results", "LiteralProber (cardinality-aware probing)",
         "ERP / recovery, injected literal faults; SFPR reduction, natural set",
         f"ERP {fmt_ms(g2l.get('B1'), 'erp_pct')} / Rec. {fmt_ms(g2l.get('B1'), 'recovery_pct')} (B1)",
         f"ERP {fmt_ms(g2l.get('A3'), 'erp_pct')} / Rec. {fmt_ms(g2l.get('A3'), 'recovery_pct')}; "
         f"SFPR red. {fmt_ms(g2n.get('A3'), 'sfpr_rel_reduction_pct')}",
         judge_gap2(mean(g2l.get("B1"), "erp_pct"), mean(g2l.get("B1"), "recovery_pct"),
                    mean(g2l.get("A3"), "erp_pct"), mean(g2l.get("A3"), "recovery_pct"))],
        ["Gap 3: over-correction of true empties", "GuardedVerifier (check before editing)",
         "EOCR (lower is better), empty-set stress set",
         f"EOCR {fmt_ms(g3.get('B1'), 'eocr_pct')} (B1); {fmt_ms(g3.get('A2'), 'eocr_pct')} (unguarded A2)",
         f"EOCR {fmt_ms(g3.get('A3'), 'eocr_pct')} (guarded A3)",
         judge_gap3(mean(g3.get("A3"), "eocr_pct"), mean(g3.get("A2"), "eocr_pct"),
                    mean(g2l.get("A3"), "recovery_pct"), mean(g2l.get("A2"), "recovery_pct"))],
    ]
    return rows


GAP_NOTE = ("Base-paper columns are measured on our re-implementation of the CHASE-SQL fixer (B1) with the same "
            "data, model and budget; CHASE-SQL itself reports only EX (73.01% on BIRD dev), so no published "
            "RDR, ERP or EOCR exists. Values are percentages (mean, plus std when several runs were made).")


def save_gap_table(ctx_output: Path, rows: List[List[str]]) -> Tuple[str, str]:
    md = markdown_table(GAP_HEADERS, rows) + "\n\n" + GAP_NOTE + "\n"
    tex = latex_table(GAP_HEADERS, rows, "Gap-resolution summary.", "tab:gap-resolution",
                      [0.10, 0.17, 0.19, 0.19, 0.19, 0.10])
    ctx_output.mkdir(parents=True, exist_ok=True)
    (ctx_output / "gap_resolution_table.md").write_text(md, encoding="utf-8")
    (ctx_output / "gap_resolution_table.tex").write_text(tex + "\n", encoding="utf-8")
    return md, tex


# =========================================================================== #
# Experiments
# =========================================================================== #
def new_payload(ctx: Context, exp_id: str, description: str) -> Dict[str, Any]:
    return {"experiment": exp_id, "description": description, "config": ctx.config_record(), "arms": {}}


def exp_single(ctx: Context, exp_id: str, arm: str, description: str) -> Dict[str, Any]:
    payload = new_payload(ctx, exp_id, description)
    payload["arms"][arm] = {"natural": get_block(ctx, arm, "natural")}
    payload["queries"] = query_records(ctx, arm, "natural")
    payload["config"] = ctx.config_record()
    return payload


def run_e1(ctx: Context) -> Dict[str, Any]:
    return exp_single(ctx, "E1", "B0", "Baseline: single-turn zero-shot generation, no feedback.")


def run_e2(ctx: Context) -> Dict[str, Any]:
    return exp_single(ctx, "E2", "B1", "Base paper: CHASE-SQL-style whole-query regeneration fixer (re-implemented).")


def run_e3(ctx: Context) -> Dict[str, Any]:
    return exp_single(ctx, "E3", "A3", "Proposed system: SQLHeal with patcher, prober and guarded verifier.")


def breakdown_by_fault(ctx: Context, arm: str, dataset: str) -> Dict[str, Any]:
    """Per fault type: items, initial error count, HRR, RDR and EX (run 0)."""
    runs = ctx.runs_cache.get((arm, dataset), [])
    if not runs:
        return {}
    fault_of = {str(i["question_id"]): i["meta"].get("fault_type", "?") for i in ctx.items(dataset)}
    groups: Dict[str, List[QueryEvaluationState]] = {}
    for s in runs[0].states:
        groups.setdefault(fault_of.get(str(s.question_id), "?"), []).append(s)
    out = {}
    for name, states in sorted(groups.items()):
        hrr, rdr = compute_hrr(states), compute_rdr(states)
        out[name] = {"n": len(states), "initial_errors": hrr.hrr_initial_errors, "hrr_pct": hrr.hrr_pct,
                     "rdr_pct": rdr.rdr_pct, "ex_pct": compute_ex(states)}
    return out


def run_e4(ctx: Context) -> Dict[str, Any]:
    payload = new_payload(ctx, "E4", "Gap-wise experiments (Gap 1: drift, Gap 2: silent empties, Gap 3: over-correction).")
    del payload["arms"]
    # Gap 1: same corrupted candidates, whole-query rewrite (B1) vs ASTPatcher (A1).
    g1 = {"dataset": "fault", "arms": {a: get_block(ctx, a, "fault") for a in ("B1", "A1")},
          "by_fault": {a: breakdown_by_fault(ctx, a, "fault") for a in ("B1", "A1")}}
    # Gap 2: natural silent empties (B1 vs full system) and injected literal faults.
    g2 = {"natural": {a: get_block(ctx, a, "natural") for a in ("B1", "A3")},
          "literal": {a: get_block(ctx, a, "literal") for a in ("B1", "A2", "A3")}}
    # Gap 3: empty-set stress set.
    g3 = {"stress": {a: get_block(ctx, a, "stress") for a in ("B1", "A2", "A3")}}
    payload.update({"gap1": g1, "gap2": g2, "gap3": g3})
    payload["gap_table"] = {"headers": GAP_HEADERS, "rows": build_gap_rows(payload), "note": GAP_NOTE}
    payload["config"] = ctx.config_record()
    return payload


def run_e5(ctx: Context) -> Dict[str, Any]:
    payload = new_payload(ctx, "E5", "Ablation: A1 patcher only, A2 + unguarded prober, A3 full system.")
    datasets = ["natural", "fault", "literal", "stress"]
    for arm in ("A1", "A2", "A3"):
        payload["arms"][arm] = {d: get_block(ctx, arm, d) for d in datasets}
    headers = ["System", "EX natural (%)", "HRR fault (%)", "RDR fault (%)", "Recovery literal (%)",
               "ERP literal (%)", "EOCR stress (%)"]
    rows = []
    for arm in ("A1", "A2", "A3"):
        a = payload["arms"][arm]
        rows.append([arm, fmt_ms(a["natural"], "ex_pct"), fmt_ms(a["fault"], "hrr_pct"), fmt_ms(a["fault"], "rdr_pct"),
                     fmt_ms(a["literal"], "recovery_pct"), fmt_ms(a["literal"], "erp_pct"),
                     fmt_ms(a["stress"], "eocr_pct")])
    payload["ablation_table"] = {"headers": headers, "rows": rows}
    payload["config"] = ctx.config_record()
    return payload


def run_e6(ctx: Context) -> Dict[str, Any]:
    payload = new_payload(ctx, "E6", "Efficiency and reliability of B0, B1 and the full system on the natural set.")
    headers = ["System", "Latency (s/query)", "LLM calls", "Prompt tok.", "Completion tok.", "Cost (USD/query)",
               "Crash rate (%)", "Timeout rate (%)", "Turns used (distribution)"]
    rows = []
    for arm in ("B0", "B1", "A3"):
        block = get_block(ctx, arm, "natural")
        payload["arms"][arm] = {"natural": block}
        runs = ctx.runs_cache[(arm, "natural")]
        hist: Counter = Counter()
        for r in runs:
            for s in r.states:
                hist[len(s.turns)] += 1
        block["retry_distribution_all_runs"] = {str(k): v for k, v in sorted(hist.items())}
        sec = block["summary"].get("avg_latency_ms")
        n = max(1, len(ctx.items("natural")))
        rows.append([
            arm,
            "n/a" if sec is None else f"{sec['mean'] / 1000.0:.3f}",
            fmt_ms(block, "avg_calls", 2),
            f"{(mean(block, 'total_prompt_tokens') or 0) / n:.0f}",
            f"{(mean(block, 'total_completion_tokens') or 0) / n:.0f}",
            fmt_ms(block, "avg_cost_usd", 5),
            fmt_ms(block, "crash_rate_pct", 2),
            fmt_ms(block, "timeout_rate_pct", 2),
            ", ".join(f"{k}:{v}" for k, v in block["retry_distribution_all_runs"].items()) or "n/a",
        ])
    payload["efficiency_table"] = {"headers": headers, "rows": rows}
    payload["config"] = ctx.config_record()
    return payload


EXPERIMENTS = {"E1": run_e1, "E2": run_e2, "E3": run_e3, "E4": run_e4, "E5": run_e5, "E6": run_e6}


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce SQLHeal experiments E1-E6")
    p.add_argument("--experiment", type=str.upper, default="ALL", choices=EXPERIMENT_IDS + ["ALL"],
                   help="Experiment to run (default: all)")
    p.add_argument("--limit", type=int, default=0, help="Max natural questions (0 = all); also caps the derived sets")
    p.add_argument("--data", default="data/bird/dev.json", help="Path to BIRD dev.json")
    p.add_argument("--db-dir", default="data/bird/dev_databases", help="Directory with <db_id>/<db_id>.sqlite")
    p.add_argument("--config", default="configs/default.json", help="Model and sandbox config")
    p.add_argument("--seed", type=int, default=42, help="Seed for sampling the question subsets")
    p.add_argument("--output-dir", default="results", help="Results go to <output-dir>/<ID>/results.json")
    p.add_argument("--log-dir", default="logs", help="Agent trajectories go to <log-dir>/<system>_<set>/")
    p.add_argument("--mock", action="store_true", help="Use the offline MockLLM and a synthetic database")
    p.add_argument("--runs", type=int, default=1, help="Repetitions of every configuration (mean/std over runs)")
    p.add_argument("--fault-sample", type=int, default=50, help="Gold queries sampled for the fault sets (x8 faults)")
    p.add_argument("--stress-size", type=int, default=100, help="Target size of the empty-set stress set")
    p.add_argument("--traj-sample", type=int, default=25, help="Agent trajectories saved per system and set")
    args = p.parse_args(argv)
    args.runs = max(1, args.runs)
    return args


def load_config(path: str) -> Dict[str, Any]:
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_context(args: argparse.Namespace, workdir: Path) -> Context:
    cfg = load_config(args.config)
    sandbox = SQLiteSandbox(timeout_sec=cfg.get("timeout_sec", 30.0))
    if args.mock:
        natural = build_mock_items(workdir)
        if args.limit > 0:
            natural = natural[: args.limit]
        mock = MockLLM()
        return Context(args, cfg, mock, sandbox, natural, mock)
    llm = LLMClient(
        model=cfg.get("model", "gpt-4o-mini"), base_url=cfg.get("base_url"), api_key=cfg.get("api_key"),
        temperature=cfg.get("temperature", 0.0), timeout=cfg.get("timeout_sec", 30.0),
        min_interval=cfg.get("min_interval_sec", 0.0),
    )
    natural, missing = load_bird_items(args.data, args.db_dir, args.limit, args.seed)
    if missing:
        print(f"Warning: {missing} question(s) skipped because their database file was not found")
    return Context(args, cfg, llm, sandbox, natural, None)


def print_experiment_tables(result: Dict[str, Any]) -> None:
    for key in ("ablation_table", "efficiency_table"):
        if key in result:
            print(f"\n{result['experiment']} - {key.replace('_', ' ')}")
            print(markdown_table(result[key]["headers"], result[key]["rows"]))
    if result["experiment"] in ("E1", "E2", "E3"):
        for arm, sets in result["arms"].items():
            block = sets["natural"]
            print(f"\n{result['experiment']} ({arm}) on {block['n_items']} items: EX {fmt_ms(block, 'ex_pct')}%, "
                  f"error {fmt_ms(block, 'error_rate_pct')}%, empty {fmt_ms(block, 'empty_rate_pct')}%")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            ctx = build_context(args, Path(tmp))
        except FileNotFoundError as exc:
            print(f"Data file not found: {exc.filename or exc}. Check --data and --db-dir (or use --mock).")
            return 1
        if not ctx.datasets["natural"]:
            print("No usable questions found; check --data and --db-dir (or use --mock).")
            return 1
        wanted = EXPERIMENT_IDS if args.experiment == "ALL" else [args.experiment]
        results: Dict[str, Dict[str, Any]] = {}
        for exp_id in wanted:
            print(f"\n=== {exp_id} ===")
            result = EXPERIMENTS[exp_id](ctx)
            results[exp_id] = result
            out = output_dir / exp_id / "results.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            print(f"saved {out}")
            print_experiment_tables(result)
        # Gap-resolution table: from this run's E4, otherwise from a saved E4 file.
        e4 = results.get("E4")
        if e4 is None:
            saved = output_dir / "E4" / "results.json"
            if saved.exists():
                with open(saved, "r", encoding="utf-8") as f:
                    e4 = json.load(f)
        if e4 is not None:
            md, tex = save_gap_table(output_dir, build_gap_rows(e4))
            print("\nGap-resolution table (Markdown)\n")
            print(md)
            print("Gap-resolution table (LaTeX)\n")
            print(tex)
        else:
            print("\nRun E4 to produce the gap-resolution table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
