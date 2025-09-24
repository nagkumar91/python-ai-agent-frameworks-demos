import logging
import os
import random
from datetime import datetime

import azure.identity
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from rich import print
from rich.logging import RichHandler
from langchain_azure_ai.callbacks.tracers import AzureAIInferenceTracer

# Setup logging with rich
logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("weekend_planner")

load_dotenv(override=True)

# Determine the actual endpoint based on API_HOST
def get_endpoint_url():
    api_host = os.getenv("API_HOST", "github")
    if api_host == "azure":
        return os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    elif api_host == "github":
        return "https://models.inference.ai.azure.com"
    elif api_host == "ollama":
        return os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1")
    else:
        return "https://api.openai.com/v1"

# Configure Azure OpenAI tracing with proper values
azure_tracer = AzureAIInferenceTracer(
    connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING"),
    enable_content_recording=os.getenv("OTEL_RECORD_CONTENT", "true").lower() == "true",
    name="Weekend Planner Agent",
    id="weekend_planner_007",
    endpoint=get_endpoint_url(),
    scope="Activity Planning"
)


API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(
        azure.identity.DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    model = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        azure_ad_token_provider=token_provider,
    )
elif API_HOST == "github":
    model = ChatOpenAI(
        model=os.getenv("GITHUB_MODEL", "gpt-4o"),
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ.get("GITHUB_TOKEN"),
    )
elif API_HOST == "ollama":
    model = ChatOpenAI(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"),
        api_key="none",
    )
else:
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


@tool
def get_weather(city: str, date: str) -> dict:
    """Returns weather data for a given city and date."""
    logger.info(f"Getting weather for {city} on {date}")
    if random.random() < 0.05:
        return {
            "temperature": 72,
            "description": "Sunny",
        }
    else:
        return {
            "temperature": 60,
            "description": "Rainy",
        }


@tool
def get_activities(city: str, date: str) -> list:
    """Returns a list of activities for a given city and date."""
    logger.info(f"Getting activities for {city} on {date}")
    return [
        {"name": "Hiking", "location": city},
        {"name": "Beach", "location": city},
        {"name": "Museum", "location": city},
    ]


@tool
def get_current_date() -> str:
    """Gets the current date from the system and returns as a string in format YYYY-MM-DD."""
    logger.info("Getting current date")
    return datetime.now().strftime("%Y-%m-%d")


agent = create_agent(
    model=model,
    prompt="You help users plan their weekends and choose the best activities for the given weather. If an activity would be unpleasant in the weather, don't suggest it. Include the date of the weekend in your response.",
    tools=[get_weather, get_activities, get_current_date],
)


def main():
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "hii what can I do this weekend in San Francisco?"}]},
        config={"callbacks": [azure_tracer]}
    )
    latest_message = response["messages"][-1]
    print(latest_message.content)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()