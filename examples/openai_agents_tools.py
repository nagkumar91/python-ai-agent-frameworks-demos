import asyncio
import logging
import os
import random
from datetime import datetime

import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from dotenv import load_dotenv
from rich.logging import RichHandler

# Setup logging with rich
logging.basicConfig(level=logging.DEBUG, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("weekend_planner")

# Disable tracing since we're not connected to a supported tracing provider
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
client = openai.AsyncOpenAI(base_url=os.environ["OLLAMA_ENDPOINT"], api_key="none")
MODEL_NAME = os.environ["OLLAMA_MODEL"]


@function_tool
def get_weather(city: str) -> str:
    logger.info(f"Getting weather for {city}")
    if random.random() < 0.05:
        return {
            "city": city,
            "temperature": 72,
            "description": "Sunny",
        }
    else:
        return {
            "city": city,
            "temperature": 60,
            "description": "Rainy",
        }


@function_tool
def get_activities(city: str, date: str) -> list:
    logger.info(f"Getting activities for {city} on {date}")
    return [
        {"name": "Hiking", "location": city},
        {"name": "Beach", "location": city},
        {"name": "Museum", "location": city},
    ]


@function_tool
def get_current_date() -> str:
    """Gets the current date and returns as a string in format YYYY-MM-DD."""
    logger.info("Getting current date")
    return datetime.now().strftime("%Y-%m-%d")


@function_tool
def python(input: str) -> str:
    """Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
    When you send a message containing python code to python, it will be executed in a stateless docker container, and the stdout of that process will be returned to you. You have to use print statements to access the output."""
    print(f"Executing python code:\n{input}")


agent = Agent(
    name="Weekend Planner",
    instructions="You help users do data science",
    tools=[python],
    # TODO: Figure out what exactly the tool description should look like
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)


async def main():
    result = await Runner.run(agent, input="hii make me a bar chart with three bars of 60 50 40")
    print(result.final_output)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    asyncio.run(main())
