import os
import json
import sqlite3

# Ensure target directories exist
target_dir = os.path.join("data", "bird")
db_dir = os.path.join(target_dir, "dev_databases", "financial")
os.makedirs(db_dir, exist_ok=True)

# 1. Create a local SQLite database for BIRD dev testing
sqlite_path = os.path.join(db_dir, "financial.sqlite")
conn = sqlite3.connect(sqlite_path)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS financial_records;")
cursor.execute("""
    CREATE TABLE financial_records (
        record_id INTEGER PRIMARY KEY,
        account_id INTEGER,
        district_id INTEGER,
        operation_type TEXT,
        amount INTEGER,
        balance INTEGER
    );
""")

records = [
    (1, 101, 1, 'Credit in Cash', 5000, 25000),
    (2, 102, 2, 'Withdrawal in Cash', 1200, 8000),
    (3, 103, 1, 'Credit in Cash', 3400, 15000),
    (4, 104, 3, 'Remittance to Another Bank', 850, 4200),
    (5, 105, 2, 'Credit in Cash', 11000, 45000),
    (6, 106, 1, 'Withdrawal in Cash', 900, 3100),
    (7, 107, 4, 'Credit in Cash', 4000, 12000),
    (8, 108, 1, 'Credit in Cash', 1500, 9500),
    (9, 109, 2, 'Remittance to Another Bank', 2100, 6700),
    (10, 110, 3, 'Credit in Cash', 7500, 18000)
]
cursor.executemany("INSERT INTO financial_records VALUES (?,?,?,?,?,?)", records)
conn.commit()
conn.close()
print(f"Created SQLite database at: {sqlite_path}")

# 2. Create sample BIRD dev.json structure
dev_data = [
    {
        "question_id": 1,
        "db_id": "financial",
        "question": "What is the total cash credit amount in district 1?",
        "evidence": "operation_type is 'Credit in Cash'",
        "SQL": "SELECT SUM(amount) FROM financial_records WHERE district_id = 1 AND operation_type = 'Credit in Cash';"
    },
    {
        "question_id": 2,
        "db_id": "financial",
        "question": "List all account IDs with a balance over 10000.",
        "evidence": "balance > 10000",
        "SQL": "SELECT account_id FROM financial_records WHERE balance > 10000;"
    },
    {
        "question_id": 3,
        "db_id": "financial",
        "question": "What is the average withdrawal amount for district 2?",
        "evidence": "operation_type is 'Withdrawal in Cash'",
        "SQL": "SELECT AVG(amount) FROM financial_records WHERE district_id = 2 AND operation_type = 'Withdrawal in Cash';"
    },
    {
        "question_id": 4,
        "db_id": "financial",
        "question": "How many records are in district 4?",
        "evidence": "",
        "SQL": "SELECT COUNT(*) FROM financial_records WHERE district_id = 4;"
    },
    {
        "question_id": 5,
        "db_id": "financial",
        "question": "What is the maximum balance recorded in district 3?",
        "evidence": "",
        "SQL": "SELECT MAX(balance) FROM financial_records WHERE district_id = 3;"
    }
]

# Duplicate/expand to 20 test cases so --limit 20 works smoothly
full_sample = []
for i in range(4):
    for item in dev_data:
        new_item = dict(item)
        new_item["question_id"] = len(full_sample) + 1
        full_sample.append(new_item)

json_path = os.path.join(target_dir, "dev.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(full_sample, f, indent=2)

print(f"Created dev.json ({len(full_sample)} questions) at: {json_path}")
print("BIRD sample environment setup complete!")