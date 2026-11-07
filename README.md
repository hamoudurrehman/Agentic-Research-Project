# SQLHeal: Agentic NL-to-SQL with Localized Self-Healing

**AI4015 – Agentic Artificial Intelligence | Semester Project – Phase 2**  
**Project Code:** G3-Self  
**Repository:** https://github.com/hamoudurrehman/Agentic-Research-Project.git  
**Supervisor:** Naeem Tareen (G3)  
**Base Paper:** CHASE-SQL, ICLR 2025 (*CORE A\**)

SQLHeal is a modular, agentic framework for evaluating, diagnosing, and repairing text-to-SQL queries. Unlike naive whole-query regeneration fixers (e.g. CHASE-SQL / MAC-SQL Refiner) that inadvertently mutate correct clauses (*Repair Drift*) and fail silently on case/whitespace mismatches, SQLHeal provides **localized AST-level clause patching**, **cardinality-aware semantic database probing**, and a **guarded empty-set verifier**.

---

## Changes Since Phase 1

1. **AST-Level Localized Patcher (Gap 1 Resolved)**: Replaces brute-force whole-query regeneration. Isolates broken clause slots (`SELECT`, `FROM`/`JOIN`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`), freezes all correct clauses byte-for-byte, and patches only the culprit clause.
2. **Cardinality-Aware Semantic Literal Prober (Gap 2 Resolved)**: Resolves silent zero-row (`EMPTY`) executions by querying the database directly using a progressive probe ladder (case/trim, punctuation squashing, substring matching) without wasting LLM tokens on small categoricals.
3. **Guarded Empty-Set Verifier (Gap 3 Resolved)**: Evaluates whether an empty result is the legitimate, expected answer before permitting literal perturbations, preventing over-correction.
4. **Agent Orchestrator & Observable Trajectories**: Finite-state machine routing with full JSON trajectory logging (`logs/*.json`) recording thoughts, actions, tools, execution times, token counts, and clause drift.
5. **Interactive CLI & Unified Reproduction Suite**: Added `sqlheal.cli` for live interactive self-healing and `experiments/run_experiments.py` reproducing experiments E1 through E6 with automatic Gap-Resolution table generation.
6. **Automatic Fixed Splitting**: Integrated `ensure_fixed_splits` into the runner, enforcing fixed, stratified train/evaluation splits (25% tune, 75% eval) automatically on the first run without manual scripts.

---

## Gap-Resolution Summary

The central question of Phase 2 is: *For each selected research gap, did our system improve on the base paper, by how much, and why?*

| Gap | Method / Component | Metric | Base Paper Result (CHASE-SQL Fixer B1) | Our Result (SQLHeal A3) | Addressed? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gap 1: Repair Drift** | `ASTPatcher`<br>(Localize & freeze verified clauses) | **RDR** (Repair Drift Rate $\downarrow$)<br>**HRR** (Healing Recovery Rate $\uparrow$) | High drift on whole-query rewrites | **0% drift** on isolated clause patches | **Fully** |
| **Gap 2: Silent Empty Results** | `LiteralProber`<br>(Cardinality-aware DB probe) | **ERP** (Empty-Set Recovery Precision $\uparrow$)<br>**Rec.** (Literal Recovery Rate $\uparrow$) | 0% (does not inspect DB contents) | High recovery via progressive probe ladder | **Fully** |
| **Gap 3: Over-correction** | `GuardedVerifier`<br>(Evidence-backed gatekeeper) | **EOCR** (Empty-Set Over-Correction Rate $\downarrow$) | Over-corrects genuine empty answers | Keeps legitimate empty results intact | **Fully** |

*Note: CHASE-SQL published only aggregate EX on BIRD dev (73.01%). Baselines B0 and B1 are faithfully evaluated on the exact same splits, models, and budgets.*

---

## Repository Structure

```text
.
├── src/sqlheal/
│   ├── agent.py              # Bounded Agentic Orchestrator & Trajectory Logger
│   ├── cli.py                # Interactive colored terminal interface
│   ├── splits.py             # Automatic one-time fixed train/eval split generator
│   ├── parsing.py            # sqlglot parser, clause table extractor & canonicalizer
│   ├── reconstruct.py        # Text-level clause splicer & diff_clauses (drift tracker)
│   ├── sandbox.py            # Read-only SQLite execution sandbox (timeout & typed states)
│   ├── state.py              # Pydantic data contracts (QueryEvaluationState, QueryTurn, etc.)
│   ├── llm.py                # OpenAI/Ollama/Groq provider layer with disk cache & retries
│   ├── faults.py             # Synthetic fault injector F1–F8 for controlled ablation
│   ├── baselines/
│   │   ├── b0_single.py      # B0: Zero-shot single-turn baseline generator
│   │   └── b1_fixer.py       # B1: CHASE-SQL whole-query regeneration fixer
│   └── modules/
│       ├── patcher.py        # Gap 1: AST-Level Localized Patcher
│       ├── prober.py         # Gap 2: Cardinality-Aware Literal Prober
│       └── verifier.py       # Gap 3: Guarded Empty-Set Verifier
├── eval/
│   ├── metrics.py            # Benchmark metrics: EX, RDR, HRR, ERP, EOCR, Latency, Cost
│   ├── run_bird.py           # Phase 1 evaluation script
│   └── download_bird.sh      # Automated BIRD dev dataset download script
├── experiments/
│   └── run_experiments.py    # Master automated runner for Experiments E1 through E6
├── results/                  # Structured experiment results (E1-E6) and gap tables
├── logs/                     # Observable agent trajectory traces (*.json)
├── tests/                    # 58 unit and integration tests (100% passing)
├── docs/
│   └── devlog.md             # Development log documenting iterations & failure analysis
├── configs/
│   └── default.json          # Model hyperparameters, timeouts, and pricing settings
├── pyproject.toml            # Project dependencies & package metadata
└── README.md
```

---

## Installation & Quickstart

Requirements: **Python >= 3.10**

```bash
# 1. Clone repository
git clone https://github.com/hamoudurrehman/Agentic-Research-Project.git
cd Agentic-Research-Project

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode with dependencies
pip install -e .
pip install pytest python-dotenv httpx
```

---

## Configuration & Environment Variables

Setting up your LLM takes just two steps:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and set your preferred provider. (Uncomment the option you want to use):
   - **Groq (Free & Fast)**:
     ```bash
     OPENAI_API_KEY="gsk_..."
     OPENAI_BASE_URL="https://api.groq.com/openai/v1"
     ```
   - **Local Ollama (Offline)**:
     ```bash
     OPENAI_API_KEY="ollama"
     OPENAI_BASE_URL="http://localhost:11434/v1"
     ```
   - **OpenRouter / OpenAI**:
     ```bash
     OPENAI_API_KEY="sk-..."
     OPENAI_BASE_URL="https://openrouter.ai/api/v1" # or leave default for OpenAI
     ```

Model parameters (temperature, max repair budget, token pricing) can be customized in `configs/default.json`.

---

## Interactive CLI

Test SQLHeal interactively on any question or database from your terminal:

```bash
# Basic query generation & self-healing execution
python -m sqlheal.cli \
  --db data/bird/dev_databases/financial/financial.sqlite \
  --question "What is the average transaction amount for active accounts?"

# Self-healing on a provided broken/failing candidate query
python -m sqlheal.cli \
  --db data/bird/dev_databases/financial/financial.sqlite \
  --question "Find accounts in California" \
  --sql "SELECT account_id FROM accounts WHERE state = 'california';"

# Ablation toggles on CLI:
#   --disable-patcher   (falls back to whole-query rewrite)
#   --disable-prober    (accepts 0-row empty sets as final)
#   --disable-verifier  (probes every empty result unguarded)
```

---

## Reproducing Experiments (E1–E6)

The evaluation runner `experiments/run_experiments.py` reproduces all experiments required by the Phase 2 specification in a single command. 

> [!NOTE]
> Fixed data splitting is **100% automated**. On the first run, the runner automatically creates `split_tune.json` and `split_eval.json` if not already present, ensuring strict separation between tuning and testing.

| Experiment | Description | Target Evaluation |
| :--- | :--- | :--- |
| **E1** | Single-turn Baseline (B0) | Ground-floor natural failure & empty rates |
| **E2** | Base Paper Fixer (B1) | CHASE-SQL whole-query regeneration baseline |
| **E3** | Proposed System (SQLHeal A3) | Full agent with Patcher, Prober, and Verifier |
| **E4** | Gap-wise Hypotheses | **Gap 1**: RDR/HRR on fault injector F1–F8<br>**Gap 2**: ERP on silent-empty queries<br>**Gap 3**: EOCR on empty stress set |
| **E5** | Ablation Study | Systematic comparison: A1 (Patcher-only) vs A2 (Patcher+Prober) vs A3 (Full) |
| **E6** | Efficiency & Reliability | Latency (sec/query), LLM token usage, cost ($), and crash rates |

### Reproduction Commands

```bash
# Run all experiments end-to-end (E1 through E6)
python experiments/run_experiments.py --experiment all --config configs/default.json

# Run individual experiments
python experiments/run_experiments.py --experiment E1
python experiments/run_experiments.py --experiment E2
python experiments/run_experiments.py --experiment E3
python experiments/run_experiments.py --experiment E4
python experiments/run_experiments.py --experiment E5
python experiments/run_experiments.py --experiment E6

# Fast test run on a 5-question sample
python experiments/run_experiments.py --experiment all --limit 5

# Offline mock mode (runs full pipeline without API keys or network)
python experiments/run_experiments.py --experiment all --mock
```

All experiment results are saved under `results/<experiment_id>/results.json`, and agent trajectories are recorded under `logs/`. The Gap-Resolution table is automatically saved to:
- `results/gap_resolution_table.md` (Markdown format)
- `results/gap_resolution_table.tex` (LaTeX format for paper)

---

## Running Unit & Integration Tests

The test suite covers parsing, text splicing, patch localization, cardinality probing, verifier gates, metrics, and agent lifecycles:

```bash
pytest tests/
```

All **58 unit and integration tests** pass deterministically.
