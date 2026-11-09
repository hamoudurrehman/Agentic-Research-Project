# SQLHeal Development Log

Required by Section 3.2 of the Phase 2 manual. Every entry records the **date**, the **failure or
limitation observed**, the **change made**, and its **measured effect**.

Rules for this log
- A number goes in only if it comes from a file produced by `experiments/run_experiments.py`
  (`results/<ID>/results.json`) or from a test. The JSON key is written next to each `FILL` slot.
- Numbers marked **(mock)** come from `--mock` (6 synthetic questions, rule-based model). They show that
  a fix works and that the pipeline is wired correctly. They are **not** research results and must not
  appear in the paper.
- Entries marked `FILL` are waiting for a real BIRD run. Replace each `FILL` with the measured value and
  the date of that run; do not estimate.

How to fill a slot (from the repository root):

```bash
python experiments/run_experiments.py --experiment all --data data/bird/dev.json \
    --db-dir data/bird/dev_databases --config configs/default.json --seed 42 --runs 3
# then read the values named in each entry, for example:
python -c "import json; d=json.load(open('results/E4/results.json')); \
print(d['gap1']['arms']['B1']['summary']['rdr_pct'])"
```

## Summary

| # | Date | Observed failure / limitation | Change | Measured effect | Status |
|---|------|-------------------------------|--------|-----------------|--------|
| 1 | FILL | Unknown natural failure rates of the baseline | B0 and B1 baselines, shared sandbox | FILL (E1, E2) | Baseline recorded after first real run |
| 2 | FILL | Whole-query regeneration can change correct clauses | Text-level clause splicer + ASTPatcher (Gap 1) | FILL (E4 Gap 1) | Implemented, awaiting real numbers |
| 3 | FILL | Queries that run but return 0 rows because of a literal mismatch | LiteralProber (Gap 2) | FILL (E4 Gap 2) | Implemented, awaiting real numbers |
| 4 | FILL | Probing every empty result corrupts true empty answers | GuardedVerifier (Gap 3) | FILL (E4 Gap 3, E5) | Implemented, awaiting real numbers |
| 5 | 2026-11-07 | Case-only literal repairs rejected as "already tried" | Loop guard compares SQL case-sensitively | (mock) EX 83.3% to 100.0%; literal recovery 50.0% to 100.0% | Fixed |
| 6 | 2026-11-07 | Baselines scored a correct empty answer as wrong | One shared EX rule for all systems | (mock) B0 EX 33.3% to 50.0% | Fixed |
| 7 | 2026-11-07 | Fault injector F3/F7 produce duplicate clauses | Version-safe `from_` lookup, pinned sqlglot (proposed) | (mock) 6 of 22 repairs unscorable | Open |
| 8 | 2026-11-09 | MSYS2 Python environment was incompatible with the installed Pydantic version; Pydantic v1 lacked the required serialization API | Recreated `.venv` using official Windows CPython 3.12 and reinstalled the project dependencies | Pydantic 2.14.3; trajectory serialization test passed; `pip check` reported no broken requirements | Fixed |
| 9 | 2026-11-09 | Existing environment and platform-specific SQLite temporary-file behavior prevented a completely clean test-suite run | Rebuilt the environment, installed pytest and python-dotenv, and ran the full test suite | 57 passed, 1 failed; remaining failure was `test_sandbox_read_only_and_typed_states` at `sqlite3.connect(tmp.name)` with `unable to open database file` | Environment repaired; one platform-specific test failure remains |
| 10 | 2026-11-09 | Real BIRD baseline and system measurements were needed beyond the earlier small smoke tests | Ran E1 with 50 questions and E2/E3 with 40 questions each using `llama3.1:8b`, seed 42, one run | Results saved in `results/final50/E1/results.json` and `results/final40/E2/` and `E3/`; sample sizes differ, so these runs are not a matched comparison | Preliminary real BIRD results recorded |
| 11 | 2026-11-09 | Gap analysis, ablation and efficiency pipelines needed end-to-end validation with the repaired environment | Ran E4, E5 and E6 smoke experiments using local Llama 3.1 8B | Results saved under `results/smoke_ollama/`; E4 exercised fault, natural, literal and stress subsets; E5 exercised A1/A2/A3; E6 measured B0/B1/A3 | Smoke validation complete; not final research results |
| 12 | 2026-11-09 | Gemini API requests need to remain below the provider's request-per-minute limit | Increased `min_interval_sec` in `configs/gemini.json` from 5 to 6 seconds | Configured request spacing is now 6 seconds; no new Gemini experiment result is claimed from this configuration change | Updated |

---

## Entry 1 - Baselines and natural failure rates

- **Date:** FILL
- **Observed failure / limitation:** Supervisor note 2 asks how often generated queries really fail on
  BIRD. Without a baseline there is nothing to repair and nothing to compare against.
- **Change:** `B0Baseline` (single zero-shot generation) and `B1FixerBaseline` (CHASE-SQL-style
  whole-query regeneration, up to `max_repair_turns`), both running through the read-only
  `SQLiteSandbox`.
- **Measured effect:** FILL
  - B0 first-turn rates: `E1 arms.B0.natural.summary.initial_error_pct`, `initial_empty_pct`
  - B0 EX: `ex_pct`; B1 EX: `E2 arms.B1.natural.summary.ex_pct`
  - Run size and seed: `config.dataset_sizes`, `config.seed`

## Entry 2 - Repair drift: clause-level repair (Gap 1)

- **Date:** FILL
- **Observed failure / limitation:** B1 regenerates the whole query, so a clause that was correct can
  change while another is being fixed. Re-printing an sqlglot AST normalises quoting, operators and
  spacing in every clause (a small drift of its own), and sqlglot cannot parse broken queries such as
  `SELECCT`, which a repairer must handle.
- **Change:** `reconstruct.py` splits a statement into top-level clauses with a quote-, comment- and
  parenthesis-aware scanner, swaps exactly one clause body and re-joins the others verbatim.
  `ASTPatcher` localises the failing clause, asks the model for that clause only, and checks that frozen
  clauses are unchanged (`frozen_clauses_intact`). `diff_clauses` makes the drift measurable.
- **Measured effect:** FILL (hypothesis H1: RDR(A1) at most half of RDR(B1), HRR within 2 points)
  - `E4 gap1.arms.B1.summary.rdr_pct` vs `E4 gap1.arms.A1.summary.rdr_pct`
  - `E4 gap1.arms.B1.summary.hrr_pct` vs `E4 gap1.arms.A1.summary.hrr_pct`
  - per fault type: `E4 gap1.by_fault.<arm>.<fault>`

## Entry 3 - Silent empty results: literal probing (Gap 2)

- **Date:** FILL
- **Observed failure / limitation:** A filter such as `WHERE state = 'california'` runs without error
  and returns 0 rows because the stored value is `California`. B1 re-prompts the model without telling
  it which literal is wrong.
- **Change:** `LiteralProber` finds the culprit predicate by dropping one conjunct at a time, samples
  real values from the column with cardinality-aware SQLite probes, and proposes a minimal literal edit.
- **Measured effect:** FILL (hypothesis H2: recovery at least 80% and ERP at least 70% on injected
  literal faults; SFPR reduced on natural failures)
  - `E4 gap2.literal.A3.summary.erp_pct` and `recovery_pct`, against `gap2.literal.B1`
  - `E4 gap2.natural.A3.summary.sfpr_before_pct`, `sfpr_after_pct`, `sfpr_rel_reduction_pct`

## Entry 4 - Over-correction of true empty results (Gap 3)

- **Date:** FILL
- **Observed failure / limitation:** If every empty result is treated as a bug, a question such as
  "Which customers ordered in 1999?" is "fixed" by loosening filters until some rows appear.
- **Change:** `GuardedVerifier` decides whether an empty result is expected before any literal is
  edited; edits are applied only on a MISMATCH verdict. `enable_verifier=False` is the unguarded
  ablation (A2).
- **Measured effect:** FILL (hypothesis H3: guarded EOCR at most 10%, recovery within 5 points of A2)
  - `E4 gap3.stress.A2.summary.eocr_pct` vs `E4 gap3.stress.A3.summary.eocr_pct`
  - recovery cost of the guard: `E4 gap2.literal.A2.summary.recovery_pct` vs `...A3...`

## Entry 5 - Loop guard rejected case-only repairs

- **Date:** 2026-11-07
- **Observed failure / limitation:** On the synthetic question "customers who live in Austin" the
  first query used `city = 'austin'` and returned 0 rows. The trajectory showed `probe_and_heal`
  producing `city = 'Austin'`, followed immediately by `stop` with reason `no_progress`. The agent
  compared lowercased SQL (`agent.py`, `seen.add(...lower())` and `if ... .lower() in seen`), so the
  repaired query looked identical to the one already tried. This blocked every case-only literal edit,
  which is the main Gap 2 fault.
- **Change:** removed `.lower()` from both comparisons in `SQLHealAgent.run` (whitespace is still
  collapsed). Added `test_agent_repairs_case_mismatch_literal`, which fails on the old code.
- **Measured effect (mock):** E3 EX 83.3% to 100.0%; E4 literal-fault recovery for A3 50.0% to 100.0%;
  the new test and `test_agent_metrics_end_to_end` fail before the change and pass after it.
  Real BIRD effect: FILL (`E4 gap2.literal.A3.summary.recovery_pct`).

## Entry 6 - One EX rule for every system

- **Date:** 2026-11-07
- **Observed failure / limitation:** `B0Baseline` and `B1FixerBaseline` set `execution_match` only when
  both gold and prediction are SUCCESS, while `SQLHealAgent` also counts EMPTY equal to EMPTY. A correct
  empty answer was therefore marked wrong for the baselines and right for the agent, which biases
  E1/E2/E3 and every Gap 3 comparison.
- **Change:** `eval/metrics.py` provides `execution_match`; `experiments/run_experiments.py` rescores
  every system's final result with it before computing any metric.
- **Measured effect (mock):** B0 EX 33.3% to 50.0% (the gold-empty question is now counted correct).
  Real BIRD effect: FILL (compare `E1 arms.B0.natural.summary.ex_pct` with the value printed by
  `eval/run_bird.py --system b0`; the difference is the number of gold-empty questions that B0 answered
  with an empty result).

## Entry 7 - Fault injector and sqlglot version (open)

- **Date:** 2026-11-07
- **Observed failure / limitation:** With sqlglot 30.30.0 the FROM argument is named `from_`, so
  `FaultInjector` F3 (`parsed.args.get("from")`) finds nothing and appends a second FROM instead of
  replacing the table. F7 does the same with ORDER BY / LIMIT when the gold query already has them.
  `split_clauses` rejects the duplicate clause, so these repairs cannot be scored for RDR.
- **Change (proposed, not yet applied):** in `faults.py` use
  `parsed.args.get("from_") or parsed.args.get("from")`, make F7 replace the existing LIMIT/ORDER BY,
  and pin the sqlglot version in `requirements.txt`.
- **Measured effect (mock):** 6 of 22 repaired fault cases were excluded from RDR
  (`rdr_excluded_queries`): all five F3 cases and one F7 case. After the fix this value should be 0;
  FILL the post-fix value from `E4 gap1.arms.A1.runs[0].rdr_excluded_queries`.



## Entry 8 - Rebuild of the Python environment

- **Date:** 2026-11-09
- **Observed failure / limitation:** The original virtual environment used MSYS2 Python 3.12.7 and Pydantic 1.10.26. This caused compatibility problems with project code relying on Pydantic 2 serialization methods, including `.model_dump()`. Installing the required Pydantic core package in that environment was also unsuccessful.
- **Change:** Preserved the old environment as `.venv_msys2_backup/`, created a new `.venv` using official Windows CPython 3.12, and reinstalled the project in editable mode. Installed pytest and python-dotenv for testing.
- **Measured effect:** Python 3.12.10, Pydantic 2.14.3 and pydantic-core 2.50.4 were installed. The `TrajectoryLogger` serialization check passed, and `python -m pip check` reported no broken requirements.

## Entry 9 - Test-suite verification after environment repair

- **Date:** 2026-11-09
- **Observed failure / limitation:** The project needed a complete test run after replacing the Python environment. One test encountered a Windows SQLite temporary-file error before the sandbox behavior could be tested.
- **Change:** Installed pytest in the new environment and executed the complete test suite using `python -m pytest -q`.
- **Measured effect:** 57 tests passed and 1 failed. The remaining failure was `test_sandbox_read_only_and_typed_states` in `tests/test_sqlheal.py`, at `sqlite3.connect(tmp.name)`, with `sqlite3.OperationalError: unable to open database file`. The failure was associated with the temporary-file/SQLite setup on this Windows environment; it was not treated as a successful test.

## Entry 10 - Initial real BIRD runs using local Llama 3.1 8B

- **Date:** 2026-11-09
- **Observed failure / limitation:** Earlier smoke tests and environment problems had prevented reliable progress toward the Phase 2 BIRD evaluation. The experiments needed to be executed with the actual local model and the correctly nested BIRD database directory.
- **Change:** Recreated the Windows virtual environment, verified the BIRD database directory, and ran E1, E2 and E3 using `llama3.1:8b`, temperature 0, seed 42 and one run. E1 used a 50-question sample; E2 and E3 used 40-question samples.
- **Measured effect:** The experiment runner completed and saved:
  - E1: `results/final50/E1/results.json`
  - E2: `results/final40/E2/results.json`
  - E3: `results/final40/E3/results.json`

  E2 recorded 40 natural questions, 82.5% final execution success, 45.0% execution accuracy, 62.5% hallucination recovery rate and 0% repair-drift rate.

  E3 recorded 40 natural questions, 92.5% final execution success, 42.5% execution accuracy, 66.67% hallucination recovery rate, 25.0% empty-result precision and 33.33% empty-result recovery among eligible candidates.

  These are one-run preliminary results. E1, E2 and E3 used different sample sizes, so their execution-accuracy values should not be presented as a controlled head-to-head comparison. The JSON files remain the source of truth for all reported metrics.

## Entry 11 - E4, E5 and E6 smoke validation using local inference

- **Date:** 2026-11-09
- **Observed failure / limitation:** The gap-analysis, ablation and efficiency experiment pipelines had not yet been verified end-to-end in the repaired Windows environment with the local model.
- **Change:** Executed E4, E5 and E6 with `llama3.1:8b`, seed 42 and one run, using a 20-question natural sample and the experiment-specific fault, literal and stress subsets.
- **Measured effect:**
  - E4 completed its fault, natural, literal and stress experiment arms and saved `results/smoke_ollama/E4/results.json`.
  - E5 completed A1, A2 and A3 across natural, fault, literal and stress subsets and saved `results/smoke_ollama/E5/results.json`.
  - E6 completed B0, B1 and A3 on the natural subset and saved `results/smoke_ollama/E6/results.json`.

  The E5 smoke run recorded the following preliminary values:

  | System | Natural EX (%) | Fault HRR (%) | Fault RDR (%) | Literal recovery (%) | Literal ERP (%) | Stress EOCR (%) |
  |---|---:|---:|---:|---:|---:|---:|
  | A1 | 25.0 | 94.7 | 4.4 | 0.0 | N/A | 0.0 |
  | A2 | 30.0 | 96.0 | 4.4 | 100.0 | 100.0 | 25.0 |
  | A3 | 30.0 | 96.0 | 4.4 | 100.0 | 100.0 | 25.0 |

  E6 recorded the following preliminary average latency measurements:

  | System | Average latency (s/query) | Average LLM calls |
  |---|---:|---:|
  | B0 | 0.055 | 1.00 |
  | B1 | 0.134 | 1.50 |
  | A3 | 0.122 | 1.55 |

  These are smoke-test results from one run, not final research results. E6 latency requires further inspection because model-client caching may affect the measured timings. The reported USD-equivalent token costs in the JSON are based on configured API pricing and must not be described as actual local Ollama inference costs.

## Entry 12 - Gemini request interval adjustment

- **Date:** 2026-11-09
- **Observed failure / limitation:** The Gemini API has a request-per-minute limit. Running experiments too close to that limit risks rate-limit errors and interrupted experiment runs.
- **Change:** Increased `min_interval_sec` in `configs/gemini.json` from 5 to 6 seconds.
- **Measured effect:** The configuration now requests a minimum six-second interval between requests. This is a preventive configuration change; no improvement in experiment accuracy or completed Gemini runs is claimed.