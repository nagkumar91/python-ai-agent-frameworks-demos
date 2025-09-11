import os
import logging

import azure.identity
import rich
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_azure_ai.callbacks.tracers import AzureOpenAITracingCallback

# Setup the client to use either Azure OpenAI or GitHub Models
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
azure_tracer = AzureOpenAITracingCallback(
    connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING"),
    enable_content_recording=os.getenv("OTEL_RECORD_CONTENT", "true").lower() == "true",
    name="GitHub Issue Analyzer Agent",
    id="github_analyzer_011",
    endpoint=get_endpoint_url(),
    scope="Development Tools"
)

API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(
        azure.identity.DefaultAzureCredential(), 
        "https://cognitiveservices.azure.com/.default"
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
        api_key=os.environ.get("GITHUB_TOKEN")
    )
elif API_HOST == "ollama":
    model = ChatOpenAI(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"),
        api_key="none"
    )
else:
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


async def setup_agent():
    client = MultiServerMCPClient(
        {
            "github": {
                "url": "https://api.githubcopilot.com/mcp/",
                "transport": "streamable_http",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
                },
            }
        }
    )
    config = {"callbacks": [azure_tracer]}
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    stale_prompt_path = os.path.join(os.path.dirname(__file__), "staleprompt.md")
    with open(stale_prompt_path) as f:
        stale_prompt = f.read()
    final_text = ""
    async for event in agent.astream_events(
        {"messages": stale_prompt + " Find one issue from Azure-samples python-ai-agent-frameworks-demos that is potentially closeable."}, 
        version="v2", 
        config=config
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # The event corresponding to a stream of new content (tokens or chunks of text)
            if chunk := event.get("data", {}).get("chunk"):
                final_text += chunk.content  # Append the new content to the accumulated text

        elif kind == "on_tool_start":
            # The event signals that a tool is about to be called
            rich.print("Called ", event["name"])  # Show which tool is being called
            rich.print("Tool input: ")
            rich.print(event["data"].get("input"))  # Display the input data sent to the tool

        elif kind == "on_tool_end":
            if output := event["data"].get("output"):
                # The event signals that a tool has finished executing
                rich.print("Tool output: ")
                rich.print(output.content)

    rich.print("Final response:")
    rich.print(final_text)


if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(setup_agent())