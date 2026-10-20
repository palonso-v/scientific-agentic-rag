import asyncio
from typing import Literal

from mcp import Client


MCP_SERVER_URL = "http://127.0.0.1:8001/mcp"


async def call_calculator(
    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
    ],
    a: float,
    b: float,
) -> float:
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "calculate_values",
            {
                "operation": operation,
                "a": a,
                "b": b,
            },
        )

        if result.is_error:
            error_text = "\n".join(
                getattr(block, "text", str(block))
                for block in result.content
            )

            raise RuntimeError(
                f"MCP calculator tool failed: {error_text}"
            )

        if not result.content:
            raise RuntimeError(
                "MCP calculator returned no content."
            )

        text = result.content[0].text

        return float(text)


def call_calculator_sync(
    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
    ],
    a: float,
    b: float,
) -> float:
    return asyncio.run(
        call_calculator(
            operation,
            a,
            b,
        )
    )


async def main():
    result = await call_calculator(
        "subtract",
        420,
        400,
    )

    print("Result:", result)


if __name__ == "__main__":
    asyncio.run(main())