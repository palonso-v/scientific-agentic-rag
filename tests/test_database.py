
import sqlite3

import pytest

from app import database


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_assistant.db"

    monkeypatch.setattr(database, "DB_PATH", db_path)

    database.initialize_database()

    return db_path


def test_database_initialization(test_db):
    assert test_db.exists()

    with sqlite3.connect(test_db) as connection:
        cursor = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'requests'
            """
        )
        assert cursor.fetchone() is not None


def test_save_request(test_db):
    request_id = database.save_request(
        question="What is PointNet?",
        answer="A neural network for point clouds.",
        latency_ms=1200.0,
    )

    rows = database.get_requests()

    assert request_id == 1
    assert len(rows) == 1
    assert rows[0][1] == "What is PointNet?"
    assert rows[0][2] == "A neural network for point clouds."
    assert rows[0][3] == 1200.0


def test_get_slow_requests(test_db):
    database.save_request(
        "Fast question",
        "Fast answer",
        500.0,
    )

    database.save_request(
        "Slow question",
        "Slow answer",
        2500.0,
    )

    database.save_request(
        "Medium question",
        "Medium answer",
        1500.0,
    )

    rows = database.get_slow_requests(1000.0)

    assert len(rows) == 2
    assert rows[0] == ("Slow question", 2500.0)
    assert rows[1] == ("Medium question", 1500.0)
