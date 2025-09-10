import asyncio
import os

import azure.identity
import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from agents.extensions.visualization import draw_graph
from dotenv import load_dotenv

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


def _configure_otel() -> None:
    """Configure OpenTelemetry with Azure Monitor or console export."""
    conn = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")
    resource = Resource.create({"service.name": "multi-agent-weather-service"})
    
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
    client = openai.AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="none")
    MODEL_NAME = "llama3.1:latest"


@function_tool
def get_weather(city: str) -> str:
    return {
        "city": city,
        "temperature": 72,
        "description": "Sunny",
    }


agent = Agent(
    name="Weather agent",
    instructions="You can only provide weather information.",
    tools=[get_weather],
)

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    tools=[get_weather],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    tools=[get_weather],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)


async def main():
    # Create a root span for better tracing context
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("weather_service_request") as span:
        # Add custom attributes for better observability
        user_request = "Hola, ¿cómo estás? ¿Puedes darme el clima para San Francisco CA?"
        span.set_attribute("user.request", user_request)
        span.set_attribute("api.host", API_HOST)
        span.set_attribute("model.name", MODEL_NAME)
        
        try:
            result = await Runner.run(triage_agent, input=user_request)
            print(result.final_output)
            
            # Add result to span for observability
            span.set_attribute("agent.response", result.final_output[:500] if result.final_output else "")
            span.set_attribute("request.success", True)
        except Exception as e:
            # Record error in span
            span.record_exception(e)
            span.set_attribute("request.success", False)
            raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Ensure all spans are flushed before exit
        trace.get_tracer_provider().shutdown()