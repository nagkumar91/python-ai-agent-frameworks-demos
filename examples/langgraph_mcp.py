"""Custom LangGraph state graph + MCP HTTP example.

Prerequisite:
Start the local MCP server defined in `mcp_server_basic.py` on port 8000:
    .venv/bin/python examples/mcp_server_basic.py
"""

import asyncio
import logging
import os

import azure.identity
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer

load_dotenv(override=True)


# Configure Azure OpenAI tracing with proper values
azure_tracer = AzureAIOpenTelemetryTracer(
    connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING"),
    enable_content_recording=os.getenv("OTEL_RECORD_CONTENT", "true").lower() == "true",
    name="Weather MCP Graph Agent",
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
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"),
        api_key="none",
    )
else:
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


async def setup_agent():
    client = MultiServerMCPClient(
        {
            "weather": {
                "url": "http://localhost:8000/mcp/",
                "transport": "streamable_http",
            }
        }
    )
    tools = await client.get_tools()

    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    config = {"callbacks": [azure_tracer]}

    weather_response = await graph.ainvoke(
        {
            "messages": (
                "What's the weather like in San Francisco today? "
                "Should I bring an umbrella?"
            )
        },
        config=config,
    )
    print(weather_response["messages"][-1].content)

    image_bytes = graph.get_graph().draw_mermaid_png()
    with open("examples/images/langgraph_mcp_http_graph.png", "wb") as f:
        f.write(image_bytes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(setup_agent())
