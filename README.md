# SQLHeal: Clause-level Diagnosis and Targeted Self-Healing for Text-to-SQL

**AI4015 – Agentic Artificial Intelligence | Semester Project, Phase 1**  
**Project Code:** G3-Self  
**Repository:** https://github.com/hamoudurrehman/Agentic-Research-Project.git  

SQLHeal provides a modular framework for evaluating, diagnosing, and repairing text-to-SQL queries. This repository contains the Phase 1 Baseline implementation (B0 single-turn and B1 CHASE-SQL-style whole-query regeneration fixer) evaluated on BIRD dev alongside clause canonicalisation, SQLite execution sandboxing, and fault injection (F1–F8).

---

## Repository Structure

```text
sqlheal/
  README.md             install, data download, how to run B0 and B1
  pyproject.toml        dependencies (sqlglot, openai-compatible client, pytest)
  data/                 BIRD dev (SQLite), stress set, fault-injected set (git-ignored)
  src/sqlheal/
    parsing.py          sqlglot parse, clause table, canonical forms
    sandbox.py          read-only SQLite execution, typed state
    state.py            query state contract (pydantic models)
    llm.py              provider layer, caching, retries
    baselines/          b0_single.py, b1_fixer.py
    modules/            patcher.py, prober.py, verifier.py (Phase 2)
    faults.py           fault injector F1-F8
  eval/                 run_bird.py, metrics.py, report.py, download_bird.sh
  tests/                parser, sandbox, fault injector, metric unit tests
  configs/              model and budget settings
```

---

## Installation

Requirements: Python >= 3.10

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install -e .
pip install pytest python-dotenv
```

---

## Download BIRD dev Data

> [!IMPORTANT]
> Because the official BIRD development dataset and SQLite databases are large. Anyone cloning this repository must run the automated setup script before evaluation.

Run the automated download and setup script:
```bash
./eval/download_bird.sh
```

This script will automatically:
1. Download the official BIRD development dataset archive (`dev.zip`).
2. Extract and organize `dev.json` (1,534 questions) and all 11 SQLite databases into `data/bird/dev_databases/<db_id>/<db_id>.sqlite`.
3. Clean up archive artifacts (`.zip`, `__MACOSX`, `.DS_Store`).

Directory layout after running:
```text
data/
  bird/
    dev.json
    dev_tables.json
    dev_databases/
      california_schools/
        california_schools.sqlite
      financial/
        financial.sqlite
      ...
```

---

## Configuration

SQLHeal supports any OpenAI-compatible provider (Groq, OpenRouter, DeepSeek, OpenAI) or local models (Ollama, vLLM).

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` with your API key:
   ```bash
   # Example: Groq (100% Free tier, ultra-fast)
   OPENAI_API_KEY="gsk_..."
   OPENAI_BASE_URL="https://api.groq.com/openai/v1"

   # Example: OpenRouter (free open-weight models)
   # OPENAI_API_KEY="sk-or-v1-..."
   # OPENAI_BASE_URL="https://openrouter.ai/api/v1"

   # Example: Local Ollama (Free, offline)
   # OPENAI_API_KEY="ollama"
   # OPENAI_BASE_URL="http://localhost:11434/v1"
   ```

3. Configure model parameters in `configs/default.json`:
   ```json
   {
     "model": "llama-3.1-8b-instant",
     "temperature": 0.0,
     "max_repair_turns": 2,
     "timeout_sec": 30.0
   }
   ```

---

## How to Run B0 and B1

### 1. Running B0 (Single-turn Zero-shot Generator)
```bash
python eval/run_bird.py --system b0 --config configs/default.json --output_dir results/b0
```
*(To test on a subset first, add `--limit 5`)*

### 2. Running B1 (Iterative Whole-Query Regeneration Fixer)
```bash
python eval/run_bird.py --system b1 --config configs/default.json --output_dir results/b1
```

### 3. Generating the Comparison Report
```bash
python eval/report.py --b0 results/b0/results.json --b1 results/b1/results.json
```

---

## What the Phase 1 Baseline Demonstrates

1. **End-to-End Pipeline**: Question and database schema go in, the LLM generator returns SQL, `sqlglot` prints the structured clause table, the SQLite sandbox executes in read-only mode, and B1 repairs errors/empty returns via whole-query regeneration.
2. **Outcome State Recording**: Per-question outcomes (`ExecutionStatus`), Execution Accuracy (`EX`), LLM calls, prompt/completion tokens, and turn histories are logged.
3. **Fault Injection & Metrics**: Controlled fault injection (F1–F8) introduces known clause-level mutations, allowing calculation of Relative Degradation / Repair Rate (`RDR`).
4. **Unit Tests**: Full unit test suite checks clause canonicalisation, sandbox isolation, fault injector F1–F8, and metric formulas.

---

## Initial Baseline Results

The table is filled from the first full B0 and B1 runs on BIRD dev (1,534 questions). It also provides the failure-rate report requested by the supervisor.

| Model | System | ERROR rate (%) | EMPTY rate (%) | EX (%) | Questions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [Frontier Model] | B0 | | | | 1,534 |
| [Frontier Model] | B1 | | | | 1,534 |
| Llama-3.1-8B-Instruct | B0 | | | | 1,534 |
| Llama-3.1-8B-Instruct | B1 | | | | 1,534 |

---

## Running Unit Tests

Run pytest across all components:
```bash
pytest
```
