
import json
from pathlib import Path

from day06.literature_assistant import answer_question
from day09.eval_cases import EVAL_CASES


def evaluate_case(case):
    result, sources = answer_question(case["question"])

    actual_status = result.status.value.lower()
    answer = result.answer

    status_ok = actual_status == case["expected_status"]

    values_ok = all(
        value in answer
        for value in case["expected_values"]
    )

    actual_source_ids = {
        source["id"] for source in sources
    }

    if case["expected_status"] == "answered":
        sources_ok = bool(
            actual_source_ids
            & set(case["expected_sources"])
        )
    else:
        sources_ok = len(sources) == 0

    passed = status_ok and values_ok and sources_ok

    return {
        "id": case["id"],
        "passed": passed,
        "status_ok": status_ok,
        "values_ok": values_ok,
        "sources_ok": sources_ok,
        "actual_status": actual_status,
        "answer": answer,
        "sources": sources,
    }


def main():
    results = []

    for case in EVAL_CASES:
        print(f"\nEvaluating: {case['id']}")
        evaluation = evaluate_case(case)
        results.append(evaluation)

        print(f"Status: {evaluation['actual_status']}")
        print(f"Answer: {evaluation['answer']}")
        print(
            "Result:",
            "PASS" if evaluation["passed"] else "FAIL"
        )

    output_path = Path("day09/eval_results.json")
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    passed = sum(result["passed"] for result in results)

    print(f"\nFINAL SCORE: {passed}/{len(results)}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
