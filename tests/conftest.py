import sqlite3, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class StubLLM:
    """Scripted LLM: returns queued replies in order and records prompts."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def generate(self, messages, **kw):
        self.calls.append(messages)
        return {"content": self.replies.pop(0), "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE client(id INTEGER, name TEXT, gender TEXT, country TEXT);
    INSERT INTO client VALUES (1,'Ann','F','USA'),(2,'Bob','M','USA'),(3,'Cy','M','New York ');
    CREATE TABLE product(id INTEGER, category TEXT);
    INSERT INTO product VALUES (1,'Men''s Shoes'),(2,'Women''s Shoes'),(3,'Hats');
    CREATE TABLE empty_t(id INTEGER, v TEXT);
    """)
    for i in range(40):
        con.execute("INSERT INTO product VALUES (?,?)", (100 + i, f"Misc Item {i}"))
    con.commit(); con.close()
    return str(path)


@pytest.fixture
def stub():
    return StubLLM
