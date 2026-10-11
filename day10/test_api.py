
from unittest.mock import patch

from fastapi.testclient import TestClient

from day10.api import app

import httpx
from openai import APIError

import asyncio
import time

import pytest

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
        "day10.api.answer_question",
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


def test_openai_failure_returns_503():

    fake_request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
    )

    with patch(
        "day10.api.answer_question",
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
            "day10.api.answer_question",
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
def reset_cache():
    clear_cache()
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
        "day10.api.answer_question",
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