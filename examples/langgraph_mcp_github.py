import os

import azure.identity
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.prebuilt import create_react_agent
from rich import print

# Setup the client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        openai_api_version=os.environ["AZURE_OPENAI_VERSION"],
        azure_ad_token_provider=token_provider,
    )
else:
    model = ChatOpenAI(model=os.getenv("GITHUB_MODEL", "gpt-4o"), base_url="https://models.inference.ai.azure.com", api_key=os.environ["GITHUB_TOKEN"])


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

    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    stale_prompt_path = os.path.join(os.path.dirname(__file__), "staleprompt.md")
    with open(stale_prompt_path) as f:
        stale_prompt = f.read()
    async for event in agent.astream_events({"messages": stale_prompt + " Find one issue from Azure-samples azure-search-openai-demo that is potentially closeable."}, version="v2"):
        print(event)


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(setup_agent())
