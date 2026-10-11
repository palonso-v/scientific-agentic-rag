
from pydantic import BaseModel, Field
from day06.search_collection import search, client

import time


class RerankResult(BaseModel):
    selected_ids: list[str] = Field(
        description="Passage IDs ordered from most to least useful"
    )


def rerank(question: str, candidates: list[dict], k: int = 3, return_usage: bool = False):
    passages = "\n\n".join(
        f"ID: {c['id']}\n"
        f"Paper: {c['paper']}, page: {c['page']}\n"
        f"TEXT:\n{c['text']}"
        for c in candidates
    )

    rerank_start = time.perf_counter()

    response = client.responses.parse(
        model="gpt-5.4-mini",
        instructions="""
        You are a scientific evidence reranker.

        Select passages that contain the information needed
        to answer the question.

        Prioritize direct numerical evidence, especially tables,
        over passages that merely mention the same terminology.

        Do not answer the question.
        Do not invent passage IDs.
        Return up to the three most useful passage IDs.
        """,
        input=f"QUESTION:\n{question}\n\nPASSAGES:\n{passages}",
        text_format=RerankResult,
    )

    rerank_seconds = time.perf_counter() - rerank_start

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency_seconds": round(rerank_seconds, 3),
    }

    selected = response.output_parsed.selected_ids

    by_id = {c["id"]: c for c in candidates}

    # Validate model-generated IDs and preserve its ranking.
    valid = []
    seen = set()

    for chunk_id in selected:
        if chunk_id in by_id and chunk_id not in seen:
            valid.append(by_id[chunk_id])
            seen.add(chunk_id)

        if len(valid) >= k:
            break

    if return_usage:
        return valid, usage

    return valid


def search_with_reranking(
    question: str,
    k: int = 3,
    candidate_k: int = 20,
    return_usage: bool = False,
):
    retrieval_start = time.perf_counter()

    candidates = search(question, k=candidate_k,)

    retrieval_seconds = time.perf_counter() - retrieval_start

    if return_usage:
        results, usage = rerank(
            question,
            candidates,
            k=k,
            return_usage=True,
        )

        usage["retrieval_seconds"] = round(
            retrieval_seconds,
            3,
        )

        return results, usage

    return rerank(
        question,
        candidates,
        k=k,
        return_usage=False,
    )