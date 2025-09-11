"""LangChain v1 MCP tools example (ported from LangGraph version).

This script demonstrates how to use LangChain v1 agent syntax with MCP tools
exposed by the GitHub MCP endpoint. It preserves the Azure OpenAI vs GitHub
model selection logic from the original LangGraph based example.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import azure.identity
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from rich.logging import RichHandler

logging.basicConfig(level=logging.WARNING, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("lang_triage")

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(
        azure.identity.DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    base_model = AzureChatOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
        azure_ad_token_provider=token_provider,
    )
elif API_HOST == "github":
    base_model = ChatOpenAI(
        model=os.getenv("GITHUB_MODEL", "gpt-4o"),
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ.get("GITHUB_TOKEN"),
    )
elif API_HOST == "ollama":
    base_model = ChatOpenAI(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"),
        api_key="none",
    )
else:
    base_model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


class ToolCallLimitMiddleware(AgentMiddleware):
    def __init__(self, limit) -> None:
        super().__init__()
        self.limit = limit

    def modify_model_request(self, request: ModelRequest, state: AgentState) -> ModelRequest:
        tool_call_count = sum(1 for msg in state["messages"] if isinstance(msg, AIMessage) and msg.tool_calls)
        if tool_call_count >= self.limit:
            logger.info("Tool call limit of %d reached, disabling further tool calls.", self.limit)
            request.tools = []
        return request


async def main():
    client = MultiServerMCPClient(
        {
            "github": {
                "url": "https://api.githubcopilot.com/mcp/",
                "transport": "streamable_http",
                "headers": {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN', '')}"},
            }
        }
    )

    tools = await client.get_tools()
    tools = [t for t in tools if t.name in ("list_issues", "search_code", "search_issues", "search_pull_requests")]
    agent = create_agent(base_model, tools, middleware=[ToolCallLimitMiddleware(limit=5)])

    stale_prompt_path = Path(__file__).parent / "staleprompt.md"
    with stale_prompt_path.open("r", encoding="utf-8") as f:
        stale_prompt = f.read()

    user_content = stale_prompt + " Find one issue from Azure-samples azure-search-openai-demo that can be closed."

    async for step in agent.astream({"messages": [HumanMessage(content=user_content)]}, stream_mode="updates"):
        for step_name, step_data in step.items():
            last_message = step_data["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                tool_name = last_message.tool_calls[0]["name"]
                tool_args = last_message.tool_calls[0]["args"]
                logger.info(f"Calling tool '{tool_name}' with args: {tool_args}")
            elif isinstance(last_message, ToolMessage):
                logger.info(f"Got tool result: {step_data['messages'][-1].content[0:200]}...")
            else:
                logger.info(f"Response: {step_data['messages'][-1].content}")


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    asyncio.run(main())
