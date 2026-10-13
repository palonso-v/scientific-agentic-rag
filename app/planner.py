from typing import Literal

from pydantic import BaseModel

from shared.openai_client import client


class PlanStep(BaseModel):
    action: Literal[
        "literature",
        "calculator",
        "conversation",
    ]

    instruction: str


class ExecutionPlan(BaseModel):
    steps: list[PlanStep]


def create_plan(question: str) -> ExecutionPlan:

    response = client.responses.parse(
        model="gpt-5.4-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "Create a minimal execution plan for the user's request. "
                    "Available actions are: "
                    "'literature' for retrieving information from scientific papers; "
                    "'calculator' for numerical calculations; "
                    "'conversation' for ordinary conversation. "
                    "Use only the steps necessary to solve the request. "
                    "Order the steps according to their dependencies. "
                    "Do not perform the tasks yourself."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        text_format=ExecutionPlan,
    )

    return response.output_parsed


if __name__ == "__main__":

    question = (
        "According to the papers, what is DGCNN accuracy "
        "under clean and heavy noise on ModelNet40-E, "
        "and calculate the accuracy drop?"
    )

    plan = create_plan(question)

    for i, step in enumerate(plan.steps, start=1):
        print(
            f"{i}. {step.action}: "
            f"{step.instruction}"
        )