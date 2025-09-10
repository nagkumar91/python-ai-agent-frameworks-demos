"""OpenAI Agents framework + MCP HTTP example.

Prerequisite:
Start the local MCP server defined in `mcp_server_basic.py` on port 8000:
    python examples/mcp_server_basic.py
"""

import asyncio
import logging
import os

import azure.identity
import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.mcp.server import MCPServerStreamableHttp
from agents.model_settings import ModelSettings
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

logging.basicConfig(level=logging.WARNING)


def _configure_otel() -> None:
    """Configure OpenTelemetry with Azure Monitor or console export."""
    conn = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")
    resource = Resource.create({"service.name": "mcp-hotel-finder-service"})
    
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


mcp_server = MCPServerStreamableHttp(name="weather", params={"url": "http://localhost:8000/mcp/"})

agent = Agent(
    name="Assistant",
    instructions="Use the tools to achieve the task",
    mcp_servers=[mcp_server],
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    model_settings=ModelSettings(tool_choice="required"),
)


async def main():
    # Create a root span for the MCP operation
    tracer = trace.get_tracer(__name__)
    
    try:
        await mcp_server.connect()
        
        with tracer.start_as_current_span("mcp_hotel_search") as span:
            message = "Find me a hotel in San Francisco for 2 nights starting from 2024-01-01. I need a hotel with free WiFi and a pool."
            
            # Add custom attributes for observability
            span.set_attribute("user.request", message)
            span.set_attribute("mcp.server", "weather")
            span.set_attribute("mcp.url", "http://localhost:8000/mcp/")
            span.set_attribute("api.host", API_HOST)
            span.set_attribute("model.name", MODEL_NAME)
            
            try:
                result = await Runner.run(starting_agent=agent, input=message)
                print(result.final_output)
                
                # Add result to span
                span.set_attribute("agent.response", result.final_output[:500] if result.final_output else "")
                span.set_attribute("request.success", True)
            except Exception as e:
                # Record error in span
                span.record_exception(e)
                span.set_attribute("request.success", False)
                raise
        
    finally:
        await mcp_server.cleanup()
        # Ensure all spans are flushed before exit
        trace.get_tracer_provider().shutdown()


if __name__ == "__main__":
    asyncio.run(main())