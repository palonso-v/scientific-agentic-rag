from typing import Literal

from mcp.server.mcpserver import MCPServer

from app.workflow import (
    CalculationRequest,
    calculate,
)


mcp = MCPServer(
    "scientific-rag-tools",
    instructions=(
        "Tools for the Scientific Agentic RAG Assistant."
    ),
)


@mcp.tool()
def calculate_values(
    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
    ],
    a: float,
    b: float,
) -> float:
    """
    Perform a basic arithmetic operation on two numbers.
    """

    request = CalculationRequest(
        operation=operation,
        a=a,
        b=b,
    )

    return calculate(request)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8001,
        stateless_http=True,
        json_response=True,
    )