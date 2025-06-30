import asyncio
import logging
import os
import shutil

import azure.identity
import openai
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.mcp import MCPServer, MCPServerStdio
from dotenv import load_dotenv
from rich.logging import RichHandler

# Setup logging with rich
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("weekend_planner")

# Disable tracing since we're not connected to a supported tracing provider
set_tracing_disabled(disabled=True)

# Setup the OpenAI client to use either Azure OpenAI or GitHub Models
load_dotenv(override=True)
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


async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Assistant",
        instructions="Use the Playwright tool to interact with the LinkedIn website.",
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    )

    message = "Open the browser and keep the browser open - pause/sleep until the user is ready to continue."
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)

    message = "Open LinkedIn invitations manager at https://www.linkedin.com/mynetwork/invitation-manager/ It is an infinitely scrolling page. Look at each invitation. If it is from a recruiter, ignore it. If it is from someone with a technical role who has a mutual connection, accept it. Stop after processing 10 invitations and report whose invitations you accepted and which you rejected. Provide links to the profiles of each person processed."
    print(f"Running: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)


async def main():
    # Documentation: First open up the browser using the npx MCP server (like with Claude/Copilot) and login

    # Then run the MCP-based example
    async with MCPServerStdio(
        name="Playwright Server, via npx",
        params={
            "command": "npx",
            "args": ["@playwright/mcp@latest"],
        },
    ) as server:
        await run(server)


if __name__ == "__main__":
    # Let's make sure the user has npx installed
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")

    asyncio.run(main())
