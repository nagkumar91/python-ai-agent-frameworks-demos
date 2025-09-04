"""LangChain v1 style music-playing agent example.

Updated to follow the LangChain v1 quickstart patterns:
* Uses `create_agent` instead of building a manual `StateGraph`.
* Uses a system prompt string + tools list directly.
* Demonstrates provider‑agnostic model initialization (Azure OpenAI, GitHub Models, Ollama).
* Adds optional structured output via a dataclass.

Docs reference: https://docs.langchain.com/oss/python/langchain-quickstart
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import azure.identity
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv(override=True)
API_HOST = os.getenv("API_HOST", "github")

# For LangChain v1 we can often just map to provider-prefixed model names.
# We keep explicit branches to respect existing env vars the repo uses.
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
        temperature=0,
    )
elif API_HOST == "github":
    model = ChatOpenAI(
        model=os.getenv("GITHUB_MODEL", "gpt-4o"),
        base_url="https://models.inference.ai.azure.com",
        api_key=os.environ.get("GITHUB_TOKEN"),
        temperature=0,
    )
elif API_HOST == "ollama":
    model = ChatOpenAI(
        model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
        base_url=os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434/v1"),
        api_key="none",
        temperature=0,
    )
else:  # Fallback to OpenAI public (requires OPENAI_API_KEY env)
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
system_prompt = """You are a helpful music assistant.

TOOLS (pick exactly ONE to call per user request):
- play_song_on_spotify: play a song on Spotify
- play_song_on_apple: play a song on Apple Music

RULES:
1. Call at most ONE tool. Never call more than one tool for a single user request.
2. Choose the single most appropriate platform (Spotify or Apple) based on any explicit user preference or request; if none is given, choose one reasonably and proceed.
5. Provide a concise cheerful confirmation.
6. If uncertain which platform to choose, default to SPOTIFY.
"""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def play_song_on_spotify(song: str) -> str:
    """Play a song on Spotify"""
    # In real code: call Spotify API
    return f"Played '{song}' on Spotify (platform=spotify)."


@tool
def play_song_on_apple(song: str) -> str:
    """Play a song on Apple Music"""
    # In real code: call Apple Music API
    return f"Played '{song}' on Apple Music (platform=apple)."


# ---------------------------------------------------------------------------
# Structured output response
# ---------------------------------------------------------------------------
class Platform(str, Enum):
    SPOTIFY = "spotify"
    APPLE = "apple"


@dataclass
class PlaySongResponse:
    platform: Platform
    confirmation: str


# ---------------------------------------------------------------------------
# Memory / checkpointer
# ---------------------------------------------------------------------------
checkpointer = InMemorySaver()


# ---------------------------------------------------------------------------
# Create agent (LangChain v1 helper)
# ---------------------------------------------------------------------------
agent = create_agent(
    model=model,
    prompt=system_prompt,
    tools=[play_song_on_spotify, play_song_on_apple],
    response_format=PlaySongResponse,  # comment out if you prefer free-form text
    checkpointer=checkpointer,
)


def main():
    config = {"configurable": {"thread_id": "1"}}
    user_message = "Can you play Taylor Swift's most popular song?"
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]}, config=config)

    # The agent returns a dict with messages + structured_response (if used)
    structured = result.get("structured_response")
    if structured:
        print(structured)
    else:
        # Fallback: print last message content
        final_messages = result.get("messages", [])
        if final_messages:
            print(final_messages[-1].content)


if __name__ == "__main__":  # pragma: no cover
    main()
