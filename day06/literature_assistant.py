from day06.search_collection import search, ensure_collection_loaded
from day08.rerank import search_with_reranking

from shared.openai_client import client

from enum import Enum
from pydantic import BaseModel, Field

import time


class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 4.50


def expand_context(results, neighbors=1, max_chunks=8):
    _, chunks, _ = ensure_collection_loaded()

    chunk_positions = {
        chunk["id"]: i
        for i, chunk in enumerate(chunks)
    }

    selected = {}

    for result in results:
        position = chunk_positions[result["id"]]

        # Always retain the retrieved passage.
        selected[result["id"]] = chunks[position]

        # Include adjacent chunks only if they belong
        # to the same paper and page.
        for offset in range(-neighbors, neighbors + 1):
            j = position + offset

            if not 0 <= j < len(chunks):
                continue

            candidate = chunks[j]

            if (
                candidate["paper"] == result["paper"]
                and candidate["page"] == result["page"]
            ):
                selected[candidate["id"]] = candidate

        if len(selected) >= max_chunks:
            break

    return list(selected.values())[:max_chunks]


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT = "insufficient_evidence"


class LiteratureAnswer(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[str] = Field(
        description="IDs of passages supporting the answer."
    )


INSTRUCTIONS = """
You are a scientific literature assistant.

Answer using only the supplied passages.

Rules:
- Cite the passage IDs supporting factual claims.
- Never invent numerical measurements.
- Distinguish experimental findings from hypotheses.
- Do not claim causation unless the passages establish it.
- If the evidence is insufficient, say so.
- If evidence is insufficient, return an empty citations list.
"""


def build_context(passages):
    return "\n\n".join(
        (
            f'<passage id="{p["id"]}" '
            f'paper="{p["paper"]}" '
            f'page="{p["page"]}">\n'
            f'{p["text"]}\n'
            f'</passage>'
        )
        for p in passages
    )



def answer_question(question, history=None, return_usage=False, candidate_k=20):
    retrieved, rerank_usage = search_with_reranking(question, k=3, candidate_k=candidate_k, return_usage=True)
    evidence = expand_context(
        retrieved,
        neighbors=1,
        max_chunks=8
    )

    context = build_context(evidence)

    ######################

    #print("\n========== RETRIEVED ==========")
    #for item in retrieved:
    #    print(item["id"], item["paper"], item["page"])

    #print("\n========== FINAL EVIDENCE ==========")
    #for item in evidence:
    #    print(item["id"], item["paper"], item["page"])

    #print("\n========== CONTEXT SENT TO MODEL ==========")
    #print(context)
    #print("==========================================\n")

    ######################

    recent_history = history or []

    history_text = "\n\n".join(
        f"User: {turn['question']}\n"
        f"Assistant: {turn['answer']}"
        for turn in recent_history
    )

    generation_start = time.perf_counter()

    response = client.responses.parse(
        model="gpt-5.4-mini",
        instructions=INSTRUCTIONS + """
        Conversation history is provided only to understand
        references and maintain conversational continuity.

        It is not scientific evidence.

        All factual claims must be supported by the
        retrieved passages. If a previous answer conflicts
        with the passages, use the passages.
        """,
        input=(
            f"Recent conversation:\n{history_text}\n\n"
            f"Retrieved evidence:\n{context}\n\n"
            f"Standalone question: {question}"
        ),
        text_format=LiteratureAnswer
    )

    generation_seconds = time.perf_counter() - generation_start

    generation_usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency_seconds": round(generation_seconds, 3),
    }

    result = response.output_parsed

    if result is None:
        raise RuntimeError("No structured answer returned.")

    allowed_ids = {p["id"] for p in evidence}

    if not set(result.citations).issubset(allowed_ids):
        raise ValueError("Invalid citations.")

    if result.status == AnswerStatus.INSUFFICIENT:
        if result.citations:
            raise ValueError(
                "Insufficient-evidence answers cannot cite passages."
            )

    elif not result.citations:
        raise ValueError("Answered response has no citations.")

    source_lookup = {p["id"]: p for p in evidence}

    sources = [
        {
            "id": citation,
            "paper": source_lookup[citation]["paper"],
            "page": source_lookup[citation]["page"]
        }
        for citation in dict.fromkeys(result.citations)
    ]

    usage = {
        "input_tokens": (
            rerank_usage["input_tokens"]
            + generation_usage["input_tokens"]
        ),
        "output_tokens": (
            rerank_usage["output_tokens"]
            + generation_usage["output_tokens"]
        ),
        "total_tokens": (
            rerank_usage["total_tokens"]
            + generation_usage["total_tokens"]
        ),
        "rerank": rerank_usage,
        "generation": generation_usage,
    }

    estimated_cost_usd = (
        usage["input_tokens"] / 1_000_000
        * INPUT_PRICE_PER_MILLION
        +
        usage["output_tokens"] / 1_000_000
        * OUTPUT_PRICE_PER_MILLION
    )

    usage["estimated_cost_usd"] = estimated_cost_usd

    if return_usage:
        return result, sources, usage

    return result, sources



if __name__ == "__main__":
    while True:
        question = input("\nAsk about your papers: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            answer, sources = answer_question(question)

            print(f"\nStatus: {answer.status.value}")
            print(f"\n{answer.answer}")

            if sources:
                print("\nSources:")

                for source in sources:
                    print(
                        f"  [{source['id']}] "
                        f"{source['paper']}, "
                        f"page {source['page']}"
                    )

        except Exception as exc:
            print(f"Unable to answer: {exc}")
