import asyncio
import logging
import os
import random
from datetime import datetime

import azure.identity
import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from dotenv import load_dotenv
from rich.logging import RichHandler

# Import OpenTelemetry components
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.openai_agents import OpenAIAgentsInstrumentor

load_dotenv(override=True)
# Try to import Azure Monitor exporter
try:
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
except ImportError:
    AzureMonitorTraceExporter = None
    print("Warning: Azure Monitor exporter not installed. Install with: pip install azure-monitor-opentelemetry-exporter")

# Setup logging with rich
logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("weekend_planner")


def _configure_otel() -> None:
    """Configure OpenTelemetry with Azure Monitor or console export."""
    conn = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")
    resource = Resource.create({"service.name": "weekend-planner-service"})
    
    tp = TracerProvider(resource=resource)
    
    if conn and AzureMonitorTraceExporter:
        tp.add_span_processor(BatchSpanProcessor(AzureMonitorTraceExporter.from_connection_string(conn)))
        # tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))  # Uncomment for debugging
        print("[otel] Azure Monitor trace exporter configured")
    else:
        tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        print("[otel] Console span exporter configured")
        if not conn:
            print("[otel] To send traces to Application Insights, set APPLICATION_INSIGHTS_CONNECTION_STRING environment variable")

    trace.set_tracer_provider(tp)


# Configure OpenTelemetry with Azure Monitor
_configure_otel()

# Instrument OpenAI Agents - this will automatically trace all agent operations
OpenAIAgentsInstrumentor().instrument()

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models

API_HOST = os.getenv("API_HOST", "github")
if API_HOST == "github":
    client = openai.AsyncOpenAI(base_url="https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])
    MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")
elif API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = openai.AsyncAzureOpenAI(
        api_version=os.environ["AZURE_OPENAI_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
    )
    MODEL_NAME = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
elif API_HOST == "ollama":
    client = openai.AsyncOpenAI(base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"), api_key="none")
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


agent = Agent(
    name="Weekend Planner",
    instructions="You help users plan their weekends and choose the best activities for the given weather. If an activity would be unpleasant in the weather, don't suggest it. Include the date of the weekend in your response.",
    tools=[get_weather, get_activities, get_current_date],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)


async def main():
    # Create a root span for the weekend planning session
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("weekend_planning_session") as span:
        user_request = "hii what can I do this weekend in Seattle?"
        
        # Add custom attributes for observability
        span.set_attribute("user.request", user_request)
        span.set_attribute("api.host", API_HOST)
        span.set_attribute("model.name", MODEL_NAME)
        span.set_attribute("agent.name", "Weekend Planner")
        span.set_attribute("target.city", "Seattle")
        
        try:
            result = await Runner.run(agent, input=user_request)
            print(result.final_output)
            
            # Add result to span
            span.set_attribute("agent.response", result.final_output[:500] if result.final_output else "")
            span.set_attribute("request.success", True)
            
            # Log the weather result if it was fetched
            logger.info(f"Weekend planning completed successfully")
        except Exception as e:
            # Record error in span
            span.record_exception(e)
            span.set_attribute("request.success", False)
            logger.error(f"Error during weekend planning: {e}")
            raise


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    try:
        asyncio.run(main())
    finally:
        # Ensure all spans are flushed before exit
        trace.get_tracer_provider().shutdown()