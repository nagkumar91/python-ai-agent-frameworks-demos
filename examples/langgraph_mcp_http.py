# https://github.com/JRAlexander/IntroToAgents1-Oxford/blob/main/intro-langgraph/time-travel.ipynb

import os

import azure.identity
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.prebuilt import create_react_agent  # REACT

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
            "itinerary": {
                # Make sure you start your itinerary server on port 8000
                "url": "http://localhost:8000/mcp/",
                "transport": "streamable_http",
            }
        }
    )

    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    hotel_response = await agent.ainvoke({"messages": "Find me a hotel in San Francisco for 2 nights starting from 2024-01-01. I need a hotel with free WiFi and a pool."})
    print(hotel_response["messages"][-1].content)


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(setup_agent())
