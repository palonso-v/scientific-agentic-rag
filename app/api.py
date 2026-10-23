
import asyncio
import logging
import time
import json
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import APIError

from shared.cache import get_cached_answer, set_cached_answer

from app.rag import answer_question
from app.database import initialize_database, save_request
from app.analytics import get_usage_summary, get_cache_statistics
from app.langgraph_agent import run_langgraph_agent

from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(initialize_database)
    yield

app = FastAPI(
    title="Scientific Literature Assistant",
    version="0.1.0",
    lifespan=lifespan,
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


class AgentResponse(BaseModel):
    answer: str
    actions: list[str]
    steps: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    summary, cache_rows = await asyncio.gather(
        asyncio.to_thread(get_usage_summary),
        asyncio.to_thread(get_cache_statistics),
    )

    return {
        "total_requests": summary[0],
        "average_latency_ms": summary[1],
        "total_input_tokens": summary[2] or 0,
        "total_output_tokens": summary[3] or 0,
        "estimated_total_cost_usd": summary[4] or 0,
        "cache_statistics": [
            {
                "cache_hit": bool(cache_hit),
                "request_count": count,
                "average_latency_ms": avg_latency,
            }
            for cache_hit, count, avg_latency in cache_rows
        ],
    }


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

    try:
        db_id = await asyncio.to_thread(
            save_request,
            question=request.question,
            answer=result.answer,
            latency_ms=latency * 1000,
            cache_hit=cache_hit,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            estimated_cost_usd=usage["estimated_cost_usd"],
        )

    except Exception:
        logger.exception(
            json.dumps({
                "event": "database_write_failed",
                "request_id": request_id,
            })
        )
        db_id = None

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

            "db_request_id": db_id,
        })
    )

    return QuestionResponse(
        status=result.status.value,
        answer=result.answer,
        sources=[Source(**source) for source in sources],
        latency_seconds=round(latency, 3),
    )


@app.post("/agent", response_model=AgentResponse)
async def agent(request: QuestionRequest):
    request_id = str(uuid.uuid4())

    try:
        async with question_semaphore:
            result = await asyncio.to_thread(
                run_langgraph_agent,
                request.question,
                request_id,
            )

    except APIError:
        logger.exception(
            json.dumps({
                "event": "agent_openai_api_failed",
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
                "event": "agent_failed",
                "request_id": request_id,
            })
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process the agent request.",
        )

    logger.info(
        json.dumps({
            "event": "agent_processed",
            "request_id": request_id,
            "actions": result["actions"],
            "steps": result["steps"],
            "error": result.get("error"),
        })
    )

    return AgentResponse(
        answer=result["answer"],
        actions=result["actions"],
        steps=result["steps"],
    )