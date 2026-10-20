from types import SimpleNamespace

from app import langgraph_agent


def make_decision(action, instruction=None):
    return SimpleNamespace(
        action=action,
        instruction=instruction,
    )


def test_literature_calculator_finish(monkeypatch):
    decisions = iter([
        make_decision(
            "literature",
            "Find the CO2 concentrations in 2015 and 2025.",
        ),
        make_decision(
            "calculator",
            "Calculate the increase from 400 to 420.",
        ),
        make_decision(
            "finish",
            None,
        ),
    ])

    def fake_decide_next_action(question, observations):
        return next(decisions)

    def fake_answer_question(question):
        result = SimpleNamespace(
            answer=(
                "The document reports 400 ppm in 2015 "
                "and 420 ppm in 2025."
            )
        )

        return result, []

    def fake_extract_calculation(instruction, previous_results):
        return SimpleNamespace(
            operation="subtract",
            a=420,
            b=400,
        )

    def fake_call_calculator_sync(operation, a, b):
        return 20

    def fake_synthesize_answer(question, observations):
        return (
            "Carbon dioxide increased from 400 ppm "
            "to 420 ppm, an increase of 20 ppm."
        )

    monkeypatch.setattr(
        langgraph_agent,
        "decide_next_action",
        fake_decide_next_action,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "answer_question",
        fake_answer_question,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "extract_calculation",
        fake_extract_calculation,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "call_calculator_sync",
        fake_call_calculator_sync,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "synthesize_answer",
        fake_synthesize_answer,
    )

    result = langgraph_agent.run_langgraph_agent(
        (
            "What were the carbon dioxide concentrations "
            "in 2015 and 2025, and what was the increase?"
        )
    )

    assert result["actions"] == [
        "literature",
        "calculator",
        "finish",
    ]

    assert result["steps"] == 3

    assert result["answer"] == (
        "Carbon dioxide increased from 400 ppm "
        "to 420 ppm, an increase of 20 ppm."
    )

    assert len(result["observations"]) == 2

    assert "400 ppm" in result["observations"][0]
    assert "420 ppm" in result["observations"][0]
    assert "20" in result["observations"][1]


def test_insufficient_evidence_then_finish(monkeypatch):
    decisions = iter([
        make_decision(
            "literature",
            "Find evidence for the requested value.",
        ),
        make_decision(
            "finish",
            None,
        ),
    ])

    def fake_decide_next_action(question, observations):
        return next(decisions)

    def fake_answer_question(question):
        result = SimpleNamespace(
            answer=(
                "The supplied document does not contain "
                "the requested information."
            )
        )

        return result, []

    def fake_synthesize_answer(question, observations):
        return (
            "The requested information cannot be determined "
            "from the supplied document."
        )

    monkeypatch.setattr(
        langgraph_agent,
        "decide_next_action",
        fake_decide_next_action,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "answer_question",
        fake_answer_question,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "synthesize_answer",
        fake_synthesize_answer,
    )

    result = langgraph_agent.run_langgraph_agent(
        "What was the global average temperature in 2024?"
    )

    assert result["actions"] == [
        "literature",
        "finish",
    ]

    assert result["steps"] == 2

    assert len(result["observations"]) == 1

    assert (
        "does not contain"
        in result["observations"][0]
    )


def test_unknown_action_falls_back_to_finish(monkeypatch):
    def fake_decide_next_action(question, observations):
        return make_decision(
            "unknown_action",
            None,
        )

    def fake_synthesize_answer(question, observations):
        return "Finished safely."

    monkeypatch.setattr(
        langgraph_agent,
        "decide_next_action",
        fake_decide_next_action,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "synthesize_answer",
        fake_synthesize_answer,
    )

    result = langgraph_agent.run_langgraph_agent(
        "Test unknown action."
    )

    assert result["actions"] == [
        "unknown_action",
    ]

    assert result["steps"] == 1
    assert result["answer"] == "Finished safely."


def test_max_steps_forces_finish(monkeypatch):
    def fake_decide_next_action(question, observations):
        return make_decision(
            "literature",
            "Search again.",
        )

    def fake_synthesize_answer(question, observations):
        return "Stopped because the maximum number of steps was reached."

    monkeypatch.setattr(
        langgraph_agent,
        "decide_next_action",
        fake_decide_next_action,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "synthesize_answer",
        fake_synthesize_answer,
    )

    initial_state = {
        "question": "Keep searching.",
        "observations": [],
        "actions": [],
        "next_action": None,
        "instruction": None,
        "answer": None,
        "steps": langgraph_agent.MAX_AGENT_STEPS - 1,
        "error": None,
    }

    result = langgraph_agent.agent_graph.invoke(
        initial_state
    )

    assert result["steps"] == langgraph_agent.MAX_AGENT_STEPS

    assert result["actions"] == [
        "literature",
    ]

    assert result["answer"] == (
        "Stopped because the maximum number of steps was reached."
    )


def test_calculator_failure_is_stored_in_state(monkeypatch):
    decisions = iter([
        make_decision(
            "calculator",
            "Calculate the increase.",
        ),
        make_decision(
            "finish",
            None,
        ),
    ])

    def fake_decide_next_action(question, observations):
        return next(decisions)

    def fake_extract_calculation(
        instruction,
        previous_results,
    ):
        return SimpleNamespace(
            operation="subtract",
            a=420,
            b=400,
        )

    def fake_call_calculator_sync(operation, a, b):
        raise RuntimeError("Calculator unavailable")

    def fake_synthesize_answer(
        question,
        observations,
    ):
        return "Unable to complete the calculation."

    monkeypatch.setattr(
        langgraph_agent,
        "decide_next_action",
        fake_decide_next_action,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "extract_calculation",
        fake_extract_calculation,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "call_calculator_sync",
        fake_call_calculator_sync,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "synthesize_answer",
        fake_synthesize_answer,
    )

    initial_state = {
        "question": "Calculate the increase.",
        "observations": [
            "Values are 400 and 420."
        ],
        "actions": [],
        "next_action": None,
        "instruction": None,
        "answer": None,
        "steps": 0,
        "error": None,
    }

    result = langgraph_agent.agent_graph.invoke(
        initial_state
    )

    assert result["actions"] == [
        "calculator",
        "finish",
    ]

    assert (
        "Calculator tool failed"
        in result["observations"][-1]
    )

    assert result["answer"] == (
        "Unable to complete the calculation."
    )


def test_calculator_node_uses_mcp(monkeypatch):
    decisions = iter([
        make_decision(
            "literature",
            "Find the CO2 concentrations in 2015 and 2025.",
        ),
        make_decision(
            "calculator",
            "Calculate the increase from 400 to 420.",
        ),
        make_decision(
            "finish",
            None,
        ),
    ])

    def fake_decide_next_action(question, observations):
        return next(decisions)

    def fake_answer_question(question):
        result = SimpleNamespace(
            answer=(
                "The document reports 400 ppm in 2015 "
                "and 420 ppm in 2025."
            )
        )

        return result, []

    def fake_extract_calculation(
        instruction,
        previous_results,
    ):
        return SimpleNamespace(
            operation="subtract",
            a=420,
            b=400,
        )

    mcp_calls = []

    def fake_call_calculator_sync(
        operation,
        a,
        b,
    ):
        mcp_calls.append(
            {
                "operation": operation,
                "a": a,
                "b": b,
            }
        )

        return 20.0

    def fake_synthesize_answer(
        question,
        observations,
    ):
        return "The increase was 20 ppm."

    monkeypatch.setattr(
        langgraph_agent,
        "decide_next_action",
        fake_decide_next_action,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "answer_question",
        fake_answer_question,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "extract_calculation",
        fake_extract_calculation,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "call_calculator_sync",
        fake_call_calculator_sync,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "synthesize_answer",
        fake_synthesize_answer,
    )

    result = langgraph_agent.run_langgraph_agent(
        (
            "According to the document, what were the "
            "carbon dioxide concentrations in 2015 and "
            "2025, and calculate the increase?"
        )
    )

    assert mcp_calls == [
        {
            "operation": "subtract",
            "a": 420,
            "b": 400,
        }
    ]

    assert result["actions"] == [
        "literature",
        "calculator",
        "finish",
    ]

    assert "Calculation result: 20.0" in result["observations"]


def test_mcp_failure_is_preserved_in_state(monkeypatch):
    decisions = iter([
        make_decision(
            "calculator",
            "Calculate the increase.",
        ),
        make_decision(
            "finish",
            None,
        ),
    ])

    def fake_decide_next_action(question, observations):
        return next(decisions)

    def fake_extract_calculation(
        instruction,
        previous_results,
    ):
        return SimpleNamespace(
            operation="subtract",
            a=420,
            b=400,
        )

    def fake_call_calculator_sync(
        operation,
        a,
        b,
    ):
        raise RuntimeError(
            "MCP calculator unavailable"
        )

    def fake_synthesize_answer(
        question,
        observations,
    ):
        return (
            "The source values were available, but the "
            "calculation could not be completed."
        )

    monkeypatch.setattr(
        langgraph_agent,
        "decide_next_action",
        fake_decide_next_action,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "extract_calculation",
        fake_extract_calculation,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "call_calculator_sync",
        fake_call_calculator_sync,
    )

    monkeypatch.setattr(
        langgraph_agent,
        "synthesize_answer",
        fake_synthesize_answer,
    )

    initial_state = {
        "question": "Calculate the increase.",
        "observations": [
            "The values are 400 ppm and 420 ppm."
        ],
        "actions": [],
        "next_action": None,
        "instruction": None,
        "answer": None,
        "steps": 0,
        "error": None,
    }

    result = langgraph_agent.agent_graph.invoke(
        initial_state
    )

    assert result["actions"] == [
        "calculator",
        "finish",
    ]

    assert (
        "Calculator tool failed: MCP calculator unavailable"
        in result["observations"][-1]
    )

    assert result["error"] == (
        "Calculator tool failed: "
        "MCP calculator unavailable"
    )

    assert result["answer"] == (
        "The source values were available, but the "
        "calculation could not be completed."
    )