from dataclasses import dataclass, field

from app.planner import ExecutionPlan, create_plan

from app.rag import answer_question

from typing import Literal

from pydantic import BaseModel

from shared.openai_client import client


MAX_PLAN_STEPS = 5

@dataclass
class WorkflowState:
    question: str
    plan: ExecutionPlan
    results: list[str] = field(default_factory=list)


def execute_plan(question: str):

    plan = create_plan(question)

    validate_plan(plan)

    state = WorkflowState(
        question=question,
        plan=plan,
    )

    for step in plan.steps:

        print(f"\nExecuting: {step.action}")
        print(f"Instruction: {step.instruction}")

        if step.action == "literature":

            result, sources = answer_question(
                step.instruction,
            )

            step_result = result.answer

            state.results.append(
                f"Literature result:\n{step_result}"
            )

            if result.status.value == "insufficient_evidence":
                break

        elif step.action == "calculator":

            if not state.results:
                raise ValueError(
                    "Calculator step requires previous results."
                )

            calculation_request = extract_calculation(
                step.instruction,
                previous_results,
            )

            value = calculate(calculation_request)

            step_result = (
                f"Calculation result: {value}"
            )

            state.results.append(step_result)

        previous_results = "\n\n".join(
            state.results
        )

        print("Previous results:")
        print(
            previous_results
            if previous_results
            else "(none)"
        )

    final_answer = synthesize_answer(
        state.question,
        state.results,
    )

    return final_answer


class CalculationRequest(BaseModel):
    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
    ]

    a: float
    b: float


def extract_calculation(
    instruction: str,
    previous_results: str,
) -> CalculationRequest:

    response = client.responses.parse(
        model="gpt-5.4-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Convert the requested calculation into "
                    "structured calculator arguments. "
                    "Use numerical values only from the provided "
                    "previous results. "
                    "Do not invent missing values."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Instruction:\n{instruction}\n\n"
                    f"Previous results:\n{previous_results}"
                ),
            },
        ],
        text_format=CalculationRequest,
    )

    return response.output_parsed



def calculate(request: CalculationRequest) -> float:

    if request.operation == "add":
        return request.a + request.b

    if request.operation == "subtract":
        return request.a - request.b

    if request.operation == "multiply":
        return request.a * request.b

    if request.operation == "divide":
        if request.b == 0:
            raise ValueError("Cannot divide by zero.")

        return request.a / request.b

    raise ValueError(
        f"Unsupported operation: {request.operation}"
    )



def synthesize_answer(
    question: str,
    results: list[str],
) -> str:

    evidence = "\n\n".join(results)

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Answer the user's original question using only "
                    "the workflow results provided. "
                    "Do not invent information. "
                    "Combine the intermediate results into one concise, "
                    "clear answer. "
                    "Preserve important numerical values and distinguish "
                    "percentage points from relative percentages."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question:\n{question}\n\n"
                    f"Workflow results:\n{evidence}"
                ),
            },
        ],
    )

    return response.output_text


def validate_plan(plan: ExecutionPlan):

    if not plan.steps:
        raise ValueError(
            "Planner returned an empty plan."
        )

    if len(plan.steps) > MAX_PLAN_STEPS:
        raise ValueError(
            f"Plan contains too many steps: "
            f"{len(plan.steps)}"
        )


if __name__ == "__main__":

    question = (
        "According to the papers, what is DGCNN accuracy "
        "under clean and heavy noise on ModelNet40-E, "
        "and calculate the accuracy drop?"
    )

    question = (
        "According to the papers, what is DGCNN's training time "
        "on an NVIDIA H100, and calculate how much faster it is "
        "than PointNet?"
    )

    final_answer = execute_plan(question)

    print("\n========== FINAL ANSWER ==========")
    print(final_answer)