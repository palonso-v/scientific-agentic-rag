import pytest

from app import database
from app.database import update_request_answer, delete_request


@pytest.fixture(autouse=True)
def clean_database():
    database.initialize_database()

    with database.get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                TRUNCATE TABLE feedback, requests
                RESTART IDENTITY CASCADE
                """
            )

    yield


def test_database_initialization():
    with database.get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = 'requests'
                )
                """
            )

            exists = cursor.fetchone()[0]

    assert exists is True


def test_save_request():
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


def test_get_slow_requests():
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


def test_update_request():
    request_id = database.save_request(
        question="What is PointNet?",
        answer="Original answer",
        latency_ms=100.0,
    )

    updated = update_request_answer(
        request_id,
        "Improved answer",
    )

    assert updated is True

    rows = database.get_requests()

    assert len(rows) == 1
    assert rows[0][2] == "Improved answer"


def test_delete_request():
    request_id = database.save_request(
        question="What is DGCNN?",
        answer="A graph neural network.",
        latency_ms=200.0,
    )

    deleted = delete_request(request_id)

    assert deleted is True
    assert database.get_requests() == []


def test_feedback_join():
    request_id = database.save_request(
        question="What is DGCNN?",
        answer="A graph neural network.",
        latency_ms=300.0,
    )

    database.save_feedback(
        request_id=request_id,
        rating=0,
        comment="Not enough detail.",
    )

    results = database.get_negative_feedback()

    assert len(results) == 1
    assert results[0][0] == "What is DGCNN?"
    assert results[0][1] == "A graph neural network."
    assert results[0][2] == 0
    assert results[0][3] == "Not enough detail."