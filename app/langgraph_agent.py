from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.dynamic_agent import (
    AgentDecision,
    decide_next_action,
)
from app.rag import answer_question
from app.workflow import (
    extract_calculation,
    synthesize_answer,
)

from app.mcp_client import call_calculator_sync

from langfuse.langchain import CallbackHandler
from langfuse import get_client


langfuse = get_client()

MAX_AGENT_STEPS = 5

class AgentState(TypedDict):
    question: str
    observations: list[str]
    actions: list[str]
    next_action: str | None
    instruction: str | None
    answer: str | None
    steps: int
    error: str | None


def decision_node(state: AgentState) -> dict:
    observations = list(state["observations"])

    if state["error"] is not None:
        observations.append(
            f"Previous tool error: {state['error']}"
        )

    decision: AgentDecision = decide_next_action(
        question=state["question"],
        observations=observations,
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
        return {
            "error": "Literature node received no instruction.",
            "observations": (
                state["observations"]
                + ["Literature tool failed: missing instruction."]
            ),
        }

    literature_instruction = (
        f"{instruction}\n\n"
        "Retrieve and report only information explicitly supported "
        "by the source documents. Do not perform arithmetic, "
        "calculations, comparisons, or derive new numerical values. "
        "Return the source values needed for later reasoning."
    )

    try:
        with langfuse.start_as_current_observation(
            name="literature-rag",
            as_type="retriever",
            input={
                "instruction": literature_instruction,
            },
        ) as span:

            result, sources, usage = answer_question(
                literature_instruction,
                None,
                True,
            )

            span.update(
                output={
                    "answer": result.answer,
                    "source_count": len(sources),
                },
                metadata={
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "estimated_cost_usd": usage["estimated_cost_usd"],
                },
            )

        observation = (
            f"Literature result:\n{result.answer}"
        )

        return {
            "observations": (
                state["observations"] + [observation]
            ),
            "error": None,
        }

    except Exception as exc:
        error_message = (
            f"Literature tool failed: {exc}"
        )

        return {
            "observations": (
                state["observations"] + [error_message]
            ),
            "error": error_message,
        }


def calculator_node(state: AgentState) -> dict:
    instruction = state["instruction"]

    if (
        instruction is None
        or state["next_action"] != "calculator"
    ):
        instruction = state["question"]

    if instruction is None:
        error_message = (
            "Calculator tool failed: missing instruction."
        )

        return {
            "observations": (
                state["observations"] + [error_message]
            ),
            "error": error_message,
        }

    if not state["observations"]:
        error_message = (
            "Calculator tool failed: no previous observations available."
        )

        return {
            "observations": (
                state["observations"] + [error_message]
            ),
            "error": error_message,
        }

    try:
        previous_results = "\n\n".join(
            state["observations"]
        )

        request = extract_calculation(
            instruction,
            previous_results,
        )

        with langfuse.start_as_current_observation(
            name="mcp-calculator",
            as_type="tool",
            input={
                "operation": request.operation,
                "a": request.a,
                "b": request.b,
            },
        ) as span:

            try:
                value = call_calculator_sync(
                    operation=request.operation,
                    a=request.a,
                    b=request.b,
                )

                span.update(
                    output={
                        "result": value,
                    },
                    metadata={
                        "protocol": "MCP",
                    },
                )

            except Exception as exc:
                span.update(
                    level="ERROR",
                    status_message=str(exc),
                    metadata={
                        "protocol": "MCP",
                    },
                )

                raise

        return {
            "observations": (
                state["observations"]
                + [f"Calculation result: {value}"]
            ),
            "error": None,
        }

    except Exception as exc:
        error_message = (
            f"Calculator tool failed: {exc}"
        )

        return {
            "observations": (
                state["observations"] + [error_message]
            ),
            "error": error_message,
        }

def finish_node(state: AgentState) -> dict:
    synthesis_question = (
        f"{state['question']}\n\n"
        "When producing the final answer, use only facts and "
        "calculation results explicitly present in the observations. "
        "Do not perform arithmetic or calculations yourself. "
        "If the user requested a calculation but the calculator tool "
        "failed and no explicit calculation result is present in the "
        "observations, state that the calculation could not be completed."
    )

    answer = synthesize_answer(
        synthesis_question,
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

    # If the user requested a calculation, do not allow the
    # agent to finish until the calculator has produced a result.
    calculation_requested = any(
        phrase in state["question"].lower()
        for phrase in (
            "calculate",
            "calculation",
            "difference",
            "increase",
            "decrease",
            "sum",
            "multiply",
            "divide",
            "percentage",
            "percent change",
        )
    )

    calculation_completed = any(
        observation.startswith("Calculation result:")
        for observation in state["observations"]
    )

    calculator_failed = any(
        observation.startswith("Calculator tool failed:")
        for observation in state["observations"]
    )

    if (
        calculation_requested
        and not calculation_completed
        and not calculator_failed
        and state["observations"]
    ):
        return "calculator"

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
    request_id: str | None = None,
):
    initial_state: AgentState = {
        "question": question,
        "observations": [],
        "actions": [],
        "next_action": None,
        "instruction": None,
        "answer": None,
        "steps": 0,
        "error": None,
    }

    langfuse_handler = CallbackHandler()

    config = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_tags": [
                "langgraph",
                "scientific-rag",
            ],
            "question": question,
        },
    }

    if request_id is not None:
        config["metadata"]["request_id"] = request_id

    final_state = agent_graph.invoke(
        initial_state,
        config=config,
    )

    return final_state


if __name__ == "__main__":
    question = (
        "According to the document, what were the carbon dioxide "
        "concentrations in 2015 and 2025, and calculate the increase?"
    )

    result = run_langgraph_agent(question)

    print("\n========== FINAL ANSWER ==========")
    print(result["answer"])

    print("\n========== TRACE ==========")
    print("Actions:", result["actions"])
    print("Steps:", result["steps"])

    print("\n========== OBSERVATIONS ==========")

    for i, observation in enumerate(
        result["observations"],
        start=1,
    ):
        print(f"\nObservation {i}:")
        print(observation)

    print("\n========== ERROR ==========")
    print(result["error"])