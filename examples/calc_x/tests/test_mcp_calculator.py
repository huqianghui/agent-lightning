# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import os

import openai
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _get_required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(f"Set one of these environment variables: {', '.join(names)}")


def _create_openai_client() -> tuple[openai.OpenAI | openai.AzureOpenAI, str]:
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return (
            openai.AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=_get_required_env("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"),
                api_version=_get_required_env("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION"),
            ),
            os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("OPENAI_MODEL", "gpt-4.1-nano")),
        )

    return (
        openai.OpenAI(
            base_url=_get_required_env("OPENAI_API_BASE", "OPENAI_BASE_URL"),
            api_key=os.environ["OPENAI_API_KEY"],
        ),
        os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
    )


async def main():
    # 1. Initialize OpenAI client
    client, model = _create_openai_client()

    # 2. Prepare MCP stdio connection to the calculator server
    server_params = StdioServerParameters(
        command="uvx",
        args=["mcp-server-calculator"],
    )

    # 3. Ask the LLM to calculate an expression via a function call
    chat_resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 31415926 * 11415789?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string", "description": "The expression to calculate"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
    )
    print(chat_resp)

    # 4. Extract the expression argument
    func_call = chat_resp.choices[0].message.tool_calls[0]  # type: ignore
    expr = json.loads(func_call.function.arguments)["expression"]  # type: ignore

    # 5. Connect to the MCP server and invoke the 'calculate' tool
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session initialized.")
            result = await session.call_tool("calculate", arguments={"expression": expr})
            # The structured result is under `.structuredContent`
            value = result.structuredContent["result"]  # type: ignore

    # 6. Print out the result
    print(f"{expr} = {value}")


if __name__ == "__main__":
    asyncio.run(main())
