import os

from dotenv import load_dotenv
from openai import OpenAI

from src.agent.prompts import FEW_SHOT_EXAMPLE, SYSTEM_PROMPT
from src.agent.tool_schema import TOOLS, execute_tool

load_dotenv()


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or api_key == "your_key_here":
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env.")

    return OpenAI(api_key=api_key)


def run_agent(user_query: str) -> str:
    client = get_client()

    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    max_steps = int(os.getenv("MAX_AGENT_STEPS", "5"))

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip() + "\n\n" + FEW_SHOT_EXAMPLE.strip(),
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content or ""

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            tool_result = execute_tool(tool_name, tool_args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    return "Agent stopped after reaching MAX_AGENT_STEPS."


if __name__ == "__main__":
    print(run_agent("Analyze ASML"))