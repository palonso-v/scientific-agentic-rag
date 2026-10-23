from pathlib import Path
import random

import app.retrieval as retrieval

from app.rerank import rerank
from evals.retrieval_cases import RETRIEVAL_CASES


EVAL_INDEX = (
    Path(__file__).resolve().parent
    / "retrieval_index"
)

RANDOM_SEED = 42

CANDIDATE_K_VALUES = [
    3,
    5,
    10,
    20,
]

RERANK_K = 3

# Keep these consistent with app/rag.py.
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 4.50


def configure_eval_index():
    retrieval.INDEX = EVAL_INDEX

    retrieval.vectors = None
    retrieval.chunks = []
    retrieval.model = None
    retrieval.bm25_index = None


def split_cases():
    cases = list(RETRIEVAL_CASES)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(cases)

    split_index = int(
        len(cases) * 0.7
    )

    development = cases[:split_index]
    test = cases[split_index:]

    return development, test


def first_relevant_rank(
    results: list[dict],
    expected_sources: list[str],
):
    expected = set(expected_sources)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result["id"] in expected:
            return rank

    return None


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    return 1.0 / rank


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
):
    return (
        input_tokens
        / 1_000_000
        * INPUT_PRICE_PER_MILLION
        +
        output_tokens
        / 1_000_000
        * OUTPUT_PRICE_PER_MILLION
    )


def build_candidate_cache(cases):
    """
    Run hybrid retrieval once per question using the
    maximum candidate depth.

    The resulting candidates are reused for all candidate_k
    values so that the experiment changes only the number
    of passages passed to the reranker.
    """

    cache = {}

    max_candidate_k = max(
        CANDIDATE_K_VALUES
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"[retrieval {index}/{len(cases)}] "
            f"{case['id']}"
        )

        cache[case["id"]] = (
            retrieval.hybrid_search(
                case["question"],
                k=max_candidate_k,
                dense_k=max_candidate_k,
                bm25_k=max_candidate_k,
            )
        )

    return cache


def evaluate_case(
    case,
    candidate_k: int,
    full_candidates: list[dict],
):
    """
    Slice the precomputed retrieval result and run
    the LLM reranker on only candidate_k passages.
    """

    candidates = full_candidates[
        :candidate_k
    ]

    reranked, usage = rerank(
        case["question"],
        candidates,
        k=RERANK_K,
        return_usage=True,
    )

    rank = first_relevant_rank(
        reranked,
        case["expected_sources"],
    )

    cost = estimate_cost(
        usage["input_tokens"],
        usage["output_tokens"],
    )

    return {
        "id": case["id"],
        "rank": rank,
        "input_tokens": usage[
            "input_tokens"
        ],
        "output_tokens": usage[
            "output_tokens"
        ],
        "total_tokens": usage[
            "total_tokens"
        ],
        "cost_usd": cost,
        "latency_seconds": usage[
            "latency_seconds"
        ],
    }


def evaluate_candidate_k(
    cases,
    candidate_k: int,
    candidate_cache,
):
    results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"[rerank {index}/{len(cases)}] "
            f"{case['id']} "
            f"(candidate_k={candidate_k})"
        )

        full_candidates = (
            candidate_cache[
                case["id"]
            ]
        )

        result = evaluate_case(
            case,
            candidate_k,
            full_candidates,
        )

        results.append(result)

    return results


def summarize(results):
    count = len(results)

    hit_at_3 = sum(
        result["rank"] is not None
        and result["rank"] <= 3
        for result in results
    ) / count

    mrr = sum(
        reciprocal_rank(
            result["rank"]
        )
        for result in results
    ) / count

    avg_input_tokens = sum(
        result["input_tokens"]
        for result in results
    ) / count

    avg_output_tokens = sum(
        result["output_tokens"]
        for result in results
    ) / count

    avg_total_tokens = sum(
        result["total_tokens"]
        for result in results
    ) / count

    avg_cost = sum(
        result["cost_usd"]
        for result in results
    ) / count

    avg_latency = sum(
        result["latency_seconds"]
        for result in results
    ) / count

    return {
        "hit@3": hit_at_3,
        "mrr": mrr,
        "avg_input_tokens": (
            avg_input_tokens
        ),
        "avg_output_tokens": (
            avg_output_tokens
        ),
        "avg_total_tokens": (
            avg_total_tokens
        ),
        "avg_cost_usd": avg_cost,
        "avg_latency_seconds": (
            avg_latency
        ),
    }


def print_metrics(
    candidate_k,
    metrics,
):
    print(
        f"\ncandidate_k = "
        f"{candidate_k}"
    )

    print(
        f"  Hit@3: "
        f"{metrics['hit@3']:.3f}"
    )

    print(
        f"  MRR: "
        f"{metrics['mrr']:.3f}"
    )

    print(
        f"  Avg input tokens: "
        f"{metrics['avg_input_tokens']:.1f}"
    )

    print(
        f"  Avg output tokens: "
        f"{metrics['avg_output_tokens']:.1f}"
    )

    print(
        f"  Avg total tokens: "
        f"{metrics['avg_total_tokens']:.1f}"
    )

    print(
        f"  Avg cost: "
        f"${metrics['avg_cost_usd']:.6f}"
    )

    print(
        f"  Avg rerank latency: "
        f"{metrics['avg_latency_seconds']:.3f}s"
    )


def select_candidate_k(
    development_cases,
):
    print(
        "\n=============================="
    )
    print(
        "CANDIDATE-K TUNING — DEVELOPMENT"
    )
    print(
        "=============================="
    )

    candidate_cache = (
        build_candidate_cache(
            development_cases
        )
    )

    evaluations = {}

    for candidate_k in CANDIDATE_K_VALUES:
        results = evaluate_candidate_k(
            development_cases,
            candidate_k,
            candidate_cache,
        )

        metrics = summarize(results)

        evaluations[candidate_k] = (
            metrics
        )

        print_metrics(
            candidate_k,
            metrics,
        )

    # Selection rule:
    #
    # 1. maximize Hit@3
    # 2. maximize MRR
    # 3. if quality is tied, prefer fewer input tokens
    best_k = max(
        CANDIDATE_K_VALUES,
        key=lambda candidate_k: (
            evaluations[
                candidate_k
            ]["hit@3"],
            evaluations[
                candidate_k
            ]["mrr"],
            -evaluations[
                candidate_k
            ]["avg_input_tokens"],
        ),
    )

    return best_k, evaluations


def evaluate_test_configuration(
    test_cases,
    candidate_k,
    candidate_cache,
):
    print(
        "\n=============================="
    )
    print(
        f"HELD-OUT TEST — "
        f"candidate_k={candidate_k}"
    )
    print(
        "=============================="
    )

    results = evaluate_candidate_k(
        test_cases,
        candidate_k,
        candidate_cache,
    )

    metrics = summarize(results)

    print_metrics(
        candidate_k,
        metrics,
    )

    return metrics


def main():
    configure_eval_index()

    development_cases, test_cases = (
        split_cases()
    )

    print(
        f"Development cases: "
        f"{len(development_cases)}"
    )

    print(
        f"Test cases: "
        f"{len(test_cases)}"
    )

    best_k, development_metrics = (
        select_candidate_k(
            development_cases
        )
    )

    print(
        "\n=============================="
    )
    print(
        "SELECTED CANDIDATE DEPTH"
    )
    print(
        "=============================="
    )

    print(
        f"candidate_k = {best_k}"
    )

    # Build retrieval candidates once for
    # the held-out test split.
    test_candidate_cache = (
        build_candidate_cache(
            test_cases
        )
    )

    # Evaluate selected configuration.
    selected_metrics = (
        evaluate_test_configuration(
            test_cases,
            best_k,
            test_candidate_cache,
        )
    )

    # Compare against the original
    # candidate_k=20 baseline.
    if best_k != 20:
        baseline_metrics = (
            evaluate_test_configuration(
                test_cases,
                20,
                test_candidate_cache,
            )
        )

        print(
            "\n=============================="
        )
        print(
            "HELD-OUT COMPARISON"
        )
        print(
            "=============================="
        )

        print(
            f"Selected candidate_k: "
            f"{best_k}"
        )

        print(
            f"Selected Hit@3: "
            f"{selected_metrics['hit@3']:.3f}"
        )

        print(
            f"Baseline Hit@3: "
            f"{baseline_metrics['hit@3']:.3f}"
        )

        print(
            f"Selected MRR: "
            f"{selected_metrics['mrr']:.3f}"
        )

        print(
            f"Baseline MRR: "
            f"{baseline_metrics['mrr']:.3f}"
        )

        token_reduction = (
            1
            - selected_metrics[
                "avg_input_tokens"
            ]
            / baseline_metrics[
                "avg_input_tokens"
            ]
        )

        cost_reduction = (
            1
            - selected_metrics[
                "avg_cost_usd"
            ]
            / baseline_metrics[
                "avg_cost_usd"
            ]
        )

        print(
            f"Input-token reduction: "
            f"{token_reduction * 100:.1f}%"
        )

        print(
            f"Cost reduction: "
            f"{cost_reduction * 100:.1f}%"
        )


if __name__ == "__main__":
    main()