from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.dynamic_agent import (
    AgentDecision,
    decide_next_action,
)
from app.rag import answer_question
from app.workflow import (
    extract_calculation,
    calculate,
    synthesize_answer,
)


MAX_AGENT_STEPS = 5


class AgentState(TypedDict):
    question: str
    observations: list[str]
    actions: list[str]
    next_action: str | None
    instruction: str | None
    answer: str | None
    steps: int


def decision_node(state: AgentState) -> dict:
    decision: AgentDecision = decide_next_action(
        question=state["question"],
        observations=state["observations"],
    )

    return {
        "next_action": decision.action,
        "instruction": decision.instruction,
        "actions": state["actions"] + [decision.action],
        "steps": state["steps"] + 1,
    }


def literature_node(state: AgentState) -> dict:
    instruction = state["instruction"]

    if instruction is None:
        raise ValueError(
            "Literature node requires an instruction."
        )

    try:
        literature_instruction = (
            f"{instruction}\n\n"
            "Retrieve and report only information explicitly supported "
            "by the source documents. Do not perform arithmetic, "
            "calculations, comparisons, or derive new numerical values. "
            "Return the source values needed for later reasoning."
        )

        result, sources = answer_question(
            literature_instruction,
        )

        observation = (
            f"Literature result:\n{result.answer}"
        )

    except ValueError as exc:
        observation = (
            "Literature search failed to provide "
            "sufficient usable evidence. "
            f"Tool message: {exc}"
        )

    return {
        "observations": (
            state["observations"] + [observation]
        )
    }


def calculator_node(state: AgentState) -> dict:
    if not state["observations"]:
        raise ValueError(
            "Calculator requires previous observations."
        )

    instruction = state["instruction"]

    if instruction is None:
        raise ValueError(
            "Calculator node requires an instruction."
        )

    previous_results = "\n\n".join(
        state["observations"]
    )

    request = extract_calculation(
        instruction,
        previous_results,
    )

    value = calculate(request)

    return {
        "observations": (
            state["observations"]
            + [f"Calculation result: {value}"]
        )
    }


def finish_node(state: AgentState) -> dict:
    answer = synthesize_answer(
        state["question"],
        state["observations"],
    )

    return {
        "answer": answer,
    }


def route_decision(
    state: AgentState,
) -> Literal[
    "literature",
    "calculator",
    "finish",
]:
    if state["steps"] >= MAX_AGENT_STEPS:
        return "finish"

    action = state["next_action"]

    if action == "literature":
        return "literature"

    if action == "calculator":
        return "calculator"

    return "finish"


builder = StateGraph(AgentState)

builder.add_node(
    "decide",
    decision_node,
)

builder.add_node(
    "literature",
    literature_node,
)

builder.add_node(
    "calculator",
    calculator_node,
)

builder.add_node(
    "finish",
    finish_node,
)

builder.add_edge(
    START,
    "decide",
)

builder.add_conditional_edges(
    "decide",
    route_decision,
    {
        "literature": "literature",
        "calculator": "calculator",
        "finish": "finish",
    },
)

builder.add_edge(
    "literature",
    "decide",
)

builder.add_edge(
    "calculator",
    "decide",
)

builder.add_edge(
    "finish",
    END,
)

agent_graph = builder.compile()


def run_langgraph_agent(
    question: str,
):
    initial_state: AgentState = {
        "question": question,
        "observations": [],
        "actions": [],
        "next_action": None,
        "instruction": None,
        "answer": None,
        "steps": 0,
    }

    final_state = agent_graph.invoke(
        initial_state
    )

    return final_state


if __name__ == "__main__":
    question = (
        "According to the document, what were the carbon dioxide "
        "concentrations in 2015 and 2025, and calculate the increase?"
    )

    result = run_langgraph_agent(
        question
    )

    print(
        "\n========== FINAL ANSWER =========="
    )
    print(result["answer"])

    print(
        "\n========== TRACE =========="
    )
    print(
        "Actions:",
        result["actions"],
    )
    print(
        "Steps:",
        result["steps"],
    )