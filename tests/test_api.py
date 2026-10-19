import asyncio
import time
import httpx
import pytest

from unittest.mock import patch
from fastapi.testclient import TestClient
from openai import APIError

from app import database
from app.api import app
from shared.cache import clear_cache


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_question():
    response = client.post(
        "/ask",
        json={"question": "Hi"},
    )

    assert response.status_code == 422


def test_ask():
    class FakeStatus:
        value = "answered"

    class FakeResult:
        status = FakeStatus()
        answer = "34.32 ± 1.46%"

    fake_sources = [
        {
            "id": "ModelNet40E_p11_c1",
            "paper": "ModelNet40E.pdf",
            "page": 11,
        }
    ]

    fake_usage = {
        "input_tokens": 1000,
        "output_tokens": 100,
        "total_tokens": 1100,
        "estimated_cost_usd": 0.0012,
        "rerank": {
            "total_tokens": 700,
            "latency_seconds": 1.2,
            "retrieval_seconds": 0.3,
        },
        "generation": {
            "total_tokens": 400,
            "latency_seconds": 0.8,
        },
    }

    with patch(
        "app.api.answer_question",
        return_value=(FakeResult(), fake_sources, fake_usage),
    ):
        response = client.post(
            "/ask",
            json={
                "question": (
                    "What is PointNet accuracy under heavy noise?"
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "answered"
    assert "34.32" in data["answer"]
    assert data["sources"][0]["page"] == 11
    assert data["latency_seconds"] >= 0

    rows = database.get_requests()

    assert len(rows) == 1
    assert rows[0][1] == "What is PointNet accuracy under heavy noise?"
    assert rows[0][2] == "34.32 ± 1.46%"
    assert rows[0][3] >= 0

    with database.get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    cache_hit,
                    input_tokens,
                    output_tokens,
                    estimated_cost_usd
                FROM requests
                WHERE id = %s
                """,
                (rows[0][0],),
            )

            metrics = cursor.fetchone()

    assert metrics[0] is False
    assert metrics[1] == 1000
    assert metrics[2] == 100
    assert metrics[3] == pytest.approx(0.0012)


def test_openai_failure_returns_503():

    fake_request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
    )

    with patch(
        "app.api.answer_question",
        side_effect=APIError(
            "Temporary upstream failure",
            request=fake_request,
            body=None,
        ),
    ):
        response = client.post(
            "/ask",
            json={
                "question": (
                    "What is PointNet accuracy under heavy noise?"
                )
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "AI service temporarily unavailable."
    }


@pytest.mark.anyio
async def test_concurrency_limit():

    active = 0
    max_active = 0

    class FakeStatus:
        value = "answered"

    class FakeResult:
        status = FakeStatus()
        answer = "Fake answer"

    fake_sources = []

    fake_usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "estimated_cost_usd": 0.0001,
        "rerank": {
            "total_tokens": 80,
            "latency_seconds": 0.1,
            "retrieval_seconds": 0.1,
        },
        "generation": {
            "total_tokens": 40,
            "latency_seconds": 0.1,
        },
    }

    def fake_answer_question(*args, **kwargs):
        nonlocal active, max_active

        active += 1
        max_active = max(max_active, active)

        time.sleep(0.2)

        active -= 1

        return FakeResult(), fake_sources, fake_usage

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as async_client:

        with patch(
            "app.api.answer_question",
            side_effect=fake_answer_question,
        ):
            responses = await asyncio.gather(
                *[
                    async_client.post(
                        "/ask",
                        json={
                            "question": "Test concurrent request"
                        },
                    )
                    for _ in range(6)
                ]
            )

    assert all(
        response.status_code == 200
        for response in responses
    )

    assert max_active <= 3


@pytest.fixture(autouse=True)
def reset_test_environment():
    clear_cache()

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

    clear_cache()


def test_identical_question_uses_cache():

    class FakeStatus:
        value = "answered"

    class FakeResult:
        status = FakeStatus()
        answer = "Cached answer"

    fake_sources = []

    fake_usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "estimated_cost_usd": 0.0001,
        "rerank": {
            "total_tokens": 80,
            "latency_seconds": 0.1,
            "retrieval_seconds": 0.1,
        },
        "generation": {
            "total_tokens": 40,
            "latency_seconds": 0.1,
        },
    }

    with patch(
        "app.api.answer_question",
        return_value=(
            FakeResult(),
            fake_sources,
            fake_usage,
        ),
    ) as mock_answer:

        response1 = client.post(
            "/ask",
            json={
                "question": "What is PointNet accuracy?"
            },
        )

        response2 = client.post(
            "/ask",
            json={
                "question": "What is PointNet accuracy?"
            },
        )

    assert response1.status_code == 200
    assert response2.status_code == 200

    assert mock_answer.call_count == 1



def test_stats_endpoint():
    # Insert two known records into our temporary database
    database.save_request(
        question="What is PointNet?",
        answer="A neural network for point clouds.",
        latency_ms=1000.0,
        cache_hit=False,
        input_tokens=100,
        output_tokens=20,
        estimated_cost_usd=0.001,
    )

    database.save_request(
        question="What is DGCNN?",
        answer="A graph-based neural network.",
        latency_ms=200.0,
        cache_hit=True,
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0.0,
    )

    # Call the real FastAPI endpoint
    response = client.get("/stats")

    assert response.status_code == 200

    data = response.json()

    assert data["total_requests"] == 2
    assert data["average_latency_ms"] == 600.0
    assert data["total_input_tokens"] == 100
    assert data["total_output_tokens"] == 20
    assert data["estimated_total_cost_usd"] == pytest.approx(0.001)

    assert len(data["cache_statistics"]) == 2