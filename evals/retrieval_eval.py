from pathlib import Path
import random

import app.retrieval as retrieval
from evals.retrieval_cases import RETRIEVAL_CASES


EVAL_INDEX = (
    Path(__file__).resolve().parent
    / "retrieval_index"
)

RANDOM_SEED = 42

BM25_WEIGHTS = [
    1.0,
    1.1,
    1.25,
    1.5,
]


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


def hit_at_k(rank, k):
    return (
        rank is not None
        and rank <= k
    )


def evaluate_case(
    case,
    retrieval_k=10,
    dense_weight=1.0,
    bm25_weight=1.0,
):
    question = case["question"]
    expected_sources = case["expected_sources"]

    dense_results = retrieval.dense_search(
        question,
        k=retrieval_k,
    )

    bm25_results = retrieval.bm25_search(
        question,
        k=retrieval_k,
    )

    hybrid_results = (
        retrieval.reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=retrieval_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
    )

    dense_rank = first_relevant_rank(
        dense_results,
        expected_sources,
    )

    bm25_rank = first_relevant_rank(
        bm25_results,
        expected_sources,
    )

    hybrid_rank = first_relevant_rank(
        hybrid_results,
        expected_sources,
    )

    return {
        "id": case["id"],
        "dense_rank": dense_rank,
        "bm25_rank": bm25_rank,
        "hybrid_rank": hybrid_rank,
    }


def evaluate_cases(
    cases,
    dense_weight=1.0,
    bm25_weight=1.0,
):
    return [
        evaluate_case(
            case,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
        for case in cases
    ]


def summarize(results, method):
    rank_key = f"{method}_rank"

    ranks = [
        result[rank_key]
        for result in results
    ]

    count = len(ranks)

    hit_1 = sum(
        hit_at_k(rank, 1)
        for rank in ranks
    ) / count

    hit_3 = sum(
        hit_at_k(rank, 3)
        for rank in ranks
    ) / count

    hit_5 = sum(
        hit_at_k(rank, 5)
        for rank in ranks
    ) / count

    mrr = sum(
        reciprocal_rank(rank)
        for rank in ranks
    ) / count

    return {
        "hit@1": hit_1,
        "hit@3": hit_3,
        "hit@5": hit_5,
        "mrr": mrr,
    }


def print_summary(name, metrics):
    print(f"\n{name}")

    print(
        f"  Hit@1: {metrics['hit@1']:.3f}"
    )

    print(
        f"  Hit@3: {metrics['hit@3']:.3f}"
    )

    print(
        f"  Hit@5: {metrics['hit@5']:.3f}"
    )

    print(
        f"  MRR:   {metrics['mrr']:.3f}"
    )


def select_best_bm25_weight(
    development_cases,
):
    print("\n==============================")
    print("WEIGHT TUNING — DEVELOPMENT")
    print("==============================")

    best_weight = None
    best_metrics = None

    for bm25_weight in BM25_WEIGHTS:
        results = evaluate_cases(
            development_cases,
            dense_weight=1.0,
            bm25_weight=bm25_weight,
        )

        metrics = summarize(
            results,
            "hybrid",
        )

        print(
            f"\nDense weight = 1.0 "
            f"| BM25 weight = {bm25_weight}"
        )

        print(
            f"  Hit@1: {metrics['hit@1']:.3f}"
        )
        print(
            f"  MRR:   {metrics['mrr']:.3f}"
        )

        if best_metrics is None:
            best_weight = bm25_weight
            best_metrics = metrics
            continue

        current_score = (
            metrics["mrr"],
            metrics["hit@1"],
        )

        best_score = (
            best_metrics["mrr"],
            best_metrics["hit@1"],
        )

        if current_score > best_score:
            best_weight = bm25_weight
            best_metrics = metrics

    return best_weight


def print_case_results(results):
    for result in results:
        print(f"\n{result['id']}")

        print(
            "  Dense rank:",
            result["dense_rank"],
        )

        print(
            "  BM25 rank:",
            result["bm25_rank"],
        )

        print(
            "  Hybrid rank:",
            result["hybrid_rank"],
        )


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

    best_bm25_weight = (
        select_best_bm25_weight(
            development_cases
        )
    )

    print("\n==============================")
    print("SELECTED WEIGHTS")
    print("==============================")

    print("Dense weight: 1.0")
    print(
        f"BM25 weight: {best_bm25_weight}"
    )

    # -----------------------------
    # Final held-out evaluation
    # -----------------------------

    test_results = evaluate_cases(
        test_cases,
        dense_weight=1.0,
        bm25_weight=best_bm25_weight,
    )

    print("\n==============================")
    print("HELD-OUT TEST CASES")
    print("==============================")

    print_case_results(
        test_results
    )

    dense_metrics = summarize(
        test_results,
        "dense",
    )

    bm25_metrics = summarize(
        test_results,
        "bm25",
    )

    hybrid_metrics = summarize(
        test_results,
        "hybrid",
    )

    print("\n==============================")
    print("HELD-OUT TEST RESULTS")
    print("==============================")

    print_summary(
        "Dense",
        dense_metrics,
    )

    print_summary(
        "BM25",
        bm25_metrics,
    )

    print_summary(
        "Weighted Hybrid RRF",
        hybrid_metrics,
    )


if __name__ == "__main__":
    main()