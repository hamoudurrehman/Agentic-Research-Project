
from experiments.run_experiments import (
    load_bird_items,
    make_literal_candidate,
    make_absent_gold
)
from sqlheal.sandbox import SQLiteSandbox
from sqlheal.state import ExecutionStatus

data = "data/bird/dev/dev_20240627/dev.json"
db_dir = "data/bird/dev/dev_20240627/dev_databases/dev_databases"

# Fixed sample of 20 questions, using seed 42.
items, missing = load_bird_items(data, db_dir, 20, 42)

sandbox = SQLiteSandbox()

literal_count = 0
stress_count = 0

for index, item in enumerate(items, start=1):
    print(f"Checking question {index}/{len(items)}...", flush=True)

    db = item["db_path"]
    gold = item["gold_sql"]

    result = sandbox.execute(db, gold)

    if result.status == ExecutionStatus.SUCCESS:

        for kind in ("case", "space"):
            candidate = make_literal_candidate(gold, kind)

            if candidate:
                test = sandbox.execute(db, candidate)

                if test.status == ExecutionStatus.EMPTY:
                    literal_count += 1
                    break

        absent = make_absent_gold(gold, str(item["question_id"]))

        if absent:
            candidate, _ = absent
            test = sandbox.execute(db, candidate)

            if test.status == ExecutionStatus.EMPTY:
                stress_count += 1

    elif result.status == ExecutionStatus.EMPTY:
        stress_count += 1

print("\n========== E4 DIAGNOSTIC ==========")
print("Usable natural questions:", len(items))
print("Missing databases:", missing)
print("Eligible literal-fault questions:", literal_count)
print("Eligible empty-stress questions:", stress_count)
