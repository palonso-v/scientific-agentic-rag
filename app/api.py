
import asyncio
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import answer_question

import json
import uuid

from openai import APIError

from shared.cache import get_cached_answer, set_cached_answer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Scientific Literature Assistant",
    version="0.1.0",
)

MAX_CONCURRENT_QUESTIONS = 3

question_semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_QUESTIONS
)


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
    )


class Source(BaseModel):
    id: str
    paper: str
    page: int


class QuestionResponse(BaseModel):
    status: str
    answer: str
    sources: list[Source]
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        cached = get_cached_answer(request.question)

        if cached is not None:
            result, sources, usage = cached
            cache_hit = True

        else:
            async with question_semaphore:
                result, sources, usage = await asyncio.to_thread(
                    answer_question,
                    request.question,
                    None,
                    True,
                )

            set_cached_answer(
                request.question,
                (result, sources, usage),
            )

            cache_hit = False

    except APIError:
        logger.exception(
            json.dumps({
                "event": "openai_api_failed",
                "request_id": request_id,
            })
        )

        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable.",
        )

    except Exception:
        logger.exception(
            json.dumps({
                "event": "question_failed",
                "request_id": request_id,
            })
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process the question.",
        )

    latency = time.perf_counter() - start

    logger.info(
        json.dumps({
            "event": "question_processed",
            "request_id": request_id,
            "status": result.status.value,
            "latency_seconds": round(latency, 3),
            "source_count": len(sources),

            "llm_input_tokens": usage["input_tokens"],
            "llm_output_tokens": usage["output_tokens"],
            "llm_total_tokens": usage["total_tokens"],

            "rerank_tokens": usage["rerank"]["total_tokens"],
            "generation_tokens": usage["generation"]["total_tokens"],

            "estimated_cost_usd": round(usage["estimated_cost_usd"],6,),

            "rerank_seconds": usage["rerank"]["latency_seconds"],
            "generation_seconds": usage["generation"]["latency_seconds"],

            "retrieval_seconds": usage["rerank"]["retrieval_seconds"],

            "cache_hit": cache_hit,
        })
    )

    return QuestionResponse(
        status=result.status.value,
        answer=result.answer,
        sources=[Source(**source) for source in sources],
        latency_seconds=round(latency, 3),
    )
