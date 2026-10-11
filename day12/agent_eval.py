from day12.dynamic_agent import run_agent

EVAL_CASES = [
    {
        "name": "literature_and_calculation",
        "question": (
            "According to the papers, what is DGCNN accuracy "
            "under clean and heavy noise on ModelNet40-E, "
            "and calculate the accuracy drop?"
        ),
        "must_include_actions": [
            "literature",
            "calculator",
            "finish",
        ],
        "max_steps": 4,
        "expected_text": "52.86",
        "expected_outcome": "answered",
    },
    {
        "name": "literature_only",
        "question": (
            "According to the papers, what is DGCNN accuracy "
            "under heavy noise on ModelNet40-E?"
        ),
        "must_include_actions": [
            "literature",
            "finish",
        ],
        "max_steps": 3,
        "expected_text": "36.83",
        "expected_outcome": "answered",
    },
    {
        "name": "missing_evidence",
        "question": (
            "According to the papers, what is DGCNN's "
            "training time on an NVIDIA H100?"
        ),
        "must_include_actions": [
            "literature",
            "finish",
        ],
        "max_steps": 3,
        "expected_text": None,
        "expected_outcome": "insufficient_evidence",
    },
]


def evaluate_case(case):

    run = run_agent(
        case["question"]
    )

    required_actions_present = all(
        action in run.actions
        for action in case["must_include_actions"]
    )

    within_step_budget = (
        run.steps <= case["max_steps"]
    )

    # Evaluate the final answer
    if case["expected_outcome"] == "answered":

        answer_correct = (
            case["expected_text"] in run.answer
        )

    elif case["expected_outcome"] == "insufficient_evidence":

        lower_answer = run.answer.lower()

        answer_correct = any(
            phrase in lower_answer
            for phrase in [
                "not report",
                "not available",
                "no evidence",
                "not specify",
                "couldn't find",
                "could not find",
            ]
        )

    passed = (
        required_actions_present
        and within_step_budget
        and answer_correct
    )

    return {
        "name": case["name"],
        "passed": passed,
        "actions": run.actions,
        "steps": run.steps,
        "required_actions_present":
            required_actions_present,
        "within_step_budget":
            within_step_budget,
        "answer_correct":
            answer_correct,
    }


if __name__ == "__main__":

    passed_count = 0

    for case in EVAL_CASES:

        print(
            f"\nRunning: {case['name']}"
        )

        result = evaluate_case(case)

        print(result)

        if result["passed"]:
            passed_count += 1

    print(
        f"\nPassed: "
        f"{passed_count}/{len(EVAL_CASES)}"
    )