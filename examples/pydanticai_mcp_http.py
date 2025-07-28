import asyncio
import logging
import os

import azure.identity
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

if API_HOST == "github":
    client = AsyncOpenAI(api_key=os.environ["GITHUB_TOKEN"], base_url="https://models.inference.ai.azure.com")
    model = OpenAIModel(os.getenv("GITHUB_MODEL", "gpt-4o"), provider=OpenAIProvider(openai_client=client))
elif API_HOST == "azure":
    token_provider = azure.identity.get_bearer_token_provider(azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = AsyncAzureOpenAI(
        api_version=os.environ["AZURE_OPENAI_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
    )
    model = OpenAIModel(os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"], provider=OpenAIProvider(openai_client=client))
else:
    raise ValueError(f"Unknown API_HOST: {API_HOST}. Set it to either 'github' or 'azure'.")

server = MCPServerStreamableHTTP("http://localhost:8000/mcp")

agent: Agent[None, str] = Agent(model, system_prompt="You are a travel planning agent. You can help users find hotels.", output_type=str, toolsets=[server])


async def main():
    result = await agent.run("Find me a hotel in San Francisco for 2 nights starting from 2024-01-01. I need a hotel with free WiFi and a pool.")
    print(result.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
