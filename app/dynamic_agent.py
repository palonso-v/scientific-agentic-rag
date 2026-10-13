from typing import Literal

from pydantic import BaseModel

from shared.openai_client import client

from app.rag import answer_question
from app.workflow import (
    extract_calculation,
    calculate,
    synthesize_answer,
)

MAX_AGENT_STEPS = 5


class AgentDecision(BaseModel):
    action: Literal[
        "literature",
        "calculator",
        "finish",
    ]

    instruction: str


def decide_next_action(
    question: str,
    observations: list[str],
) -> AgentDecision:

    history = "\n\n".join(observations)

    if not history:
        history = "(none)"

    response = client.responses.parse(
        model="gpt-5.4-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "You are controlling a research workflow. "
                    "Decide exactly one next action.\n\n"

                    "Available actions:\n"
                    "- literature: retrieve information from scientific papers.\n"
                    "- calculator: perform a numerical calculation using "
                    "values already found in previous observations.\n"
                    "- finish: stop when the user's request has been fully answered.\n\n"

                    "Choose literature when more evidence is required. "
                    "Choose calculator only when the required numerical values "
                    "are already present in the observations. "
                    "Choose finish only when enough information has been gathered "
                    "to answer the original request."
                    "Do not repeat a literature search that has already failed "
                    "unless you can make a meaningfully different request. "
                    "If the available evidence is insufficient and there is no "
                    "useful alternative strategy, choose finish rather than "
                    "repeating the same action."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question:\n{question}\n\n"
                    f"Observations so far:\n{history}"
                ),
            },
        ],
        text_format=AgentDecision,
    )

    return response.output_parsed


def run_agent(question: str):

    observations = []
    actions = []

    for step_number in range(
        1,
        MAX_AGENT_STEPS + 1,
    ):

        decision = decide_next_action(
            question,
            observations,
        )

        actions.append(decision.action)

        print(
            f"\nStep {step_number}: "
            f"{decision.action}"
        )
        print(
            f"Instruction: {decision.instruction}"
        )

        if decision.action == "literature":

            try:
                result, sources = answer_question(
                    decision.instruction,
                )

                observations.append(
                    f"Literature result:\n{result.answer}"
                )

                if (
                    result.status.value
                    == "insufficient_evidence"
                ):
                    continue

            except ValueError as exc:

                observations.append(
                    "Literature search failed to provide sufficient usable evidence. "
                    f"Tool message: {exc}"
                )

                continue

        elif decision.action == "calculator":

            if not observations:
                raise ValueError(
                    "Calculator requires previous observations."
                )

            previous_results = "\n\n".join(
                observations
            )

            request = extract_calculation(
                decision.instruction,
                previous_results,
            )

            value = calculate(request)

            observations.append(
                f"Calculation result: {value}"
            )

        elif decision.action == "finish":

            answer = synthesize_answer(
                question,
                observations,
            )

            return AgentRun(
                answer=answer,
                actions=actions,
                steps=step_number,
            )

    raise RuntimeError(
        f"Agent reached the maximum of "
        f"{MAX_AGENT_STEPS} steps without "
        f"completing the task."
    )


class AgentRun(BaseModel):
    answer: str
    actions: list[str]
    steps: int


if __name__ == "__main__":

    question = (
        "According to the papers, what is DGCNN accuracy "
        "under clean and heavy noise on ModelNet40-E, "
        "and calculate the accuracy drop?"
    )

    run = run_agent(question)

    print("\n========== FINAL ANSWER ==========")
    print(run.answer)

    print("\n========== TRACE ==========")
    print("Actions:", run.actions)
    print("Steps:", run.steps)